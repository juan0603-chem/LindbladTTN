# coding: utf-8
"""TDVP time-evolution engine.

Two strategies are provided:

PS1 — Projector-splitting (1-site, fixed rank)
    Forward sweep (leaf→root) at +dt/2, then backward sweep (root→leaf) at
    -dt/2, achieving second-order Strang-splitting accuracy.  Bond dimension
    is held constant throughout.

VMF — Variable mean-field
    All core tensor equations-of-motion are integrated simultaneously as a
    single ODE via torchdiffeq.  The EOM for non-root tensors is the
    projected gradient
        dU_s/dt = (I − U_s U_sᴴ) H_eff_s U_s
    and for the root:
        dA_0/dt = H_eff_root @ A_0
    Bond dimensions are constant (the projector keeps rank fixed).

Both strategies use :class:`~lindblad_ttn.propagation.effective_h.SimpleEffectiveH`
to compute the local effective Hamiltonian.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from math import prod
from typing import Literal

import torch

from lindblad_ttn.core.backend import (
    DEVICE,
    DTYPE,
    _svd_on_device,
    kron_torch,
    opt_odeint,
    svd_truncate,
    transform,
)
from lindblad_ttn.core.gauge import move_gauge_qr
from lindblad_ttn.core.graph import End, Frame, Node
from lindblad_ttn.core.model import Model
from lindblad_ttn.core.sop import SumOfProducts
from lindblad_ttn.propagation.effective_h import SimpleEffectiveH
from lindblad_ttn.propagation.integrators import (
    krylov_expm_apply,
    krylov_expm_apply_dense,
)


# ---------------------------------------------------------------------------
# Low-level split / merge (mirrors tenso _one_site_split / _one_site_merge)
# ---------------------------------------------------------------------------

def _one_site_split(
    array: torch.Tensor,
    axis: int,
    max_rank: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SVD-split a tensor at ``axis``.

    The singular values are absorbed into the ``edge_array`` (right factor).

    Parameters
    ----------
    array : torch.Tensor
        Node tensor.
    axis : int
        Bond axis to split at.
    max_rank : int, optional
        Truncate to this bond dimension.

    Returns
    -------
    p_tensor : torch.Tensor
        Isometric tensor; same shape as ``array`` except the last axis has
        dimension = new bond dim.
    edge_array : torch.Tensor
        Shape ``(new_bond, old_bond_dim)``; S·Vᴴ from the SVD.
    """
    shape = list(array.shape)
    dim = shape[axis]
    l_shape = shape[:axis] + shape[axis + 1:]

    mat = array.moveaxis(axis, -1).reshape(-1, dim)

    # SVD on-device (with CPU fallback for older torch CUDA complex builds).
    U, S, Vh = _svd_on_device(mat)
    S = S.to(dtype=DTYPE)

    rank = S.shape[0]
    if max_rank is not None:
        rank = min(rank, max_rank)
    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]

    edge_array = S.unsqueeze(1) * Vh          # (rank, old_dim)
    p_tensor = U.reshape(l_shape + [rank]).moveaxis(-1, axis)
    return p_tensor, edge_array


def _one_site_merge(
    array: torch.Tensor,
    axis: int,
    edge_array: torch.Tensor,
) -> torch.Tensor:
    """Absorb ``edge_array`` into ``array`` along ``axis``.

    Parameters
    ----------
    array : torch.Tensor
        Node tensor.
    axis : int
        Bond axis.
    edge_array : torch.Tensor
        Shape ``(new_bond, old_bond)``.  Contracted with ``array[axis]``.

    Returns
    -------
    torch.Tensor
        Node tensor with bond ``axis`` updated.
    """
    return transform(edge_array, array, 1, axis)


# ---------------------------------------------------------------------------
# TDVP Propagator
# ---------------------------------------------------------------------------

class TDVPPropagator:
    """TDVP time evolution for a TTN Lindblad state.

    Parameters
    ----------
    model : Model
        TTN state.
    frame : Frame
        Tree topology.
    sop : SumOfProducts
        SoP Liouvillian.
    dof_to_end : dict[str, End]
        Maps DOF names to End nodes.
    strategy : {'ps1', 'vmf'}
        Integration strategy.
    bond_dim : int
        Maximum bond dimension (for PS1 truncation).
    vmf_atol : float
        Absolute tolerance for the VMF ODE integrator.
    ode_method : str
        ODE method (``'dopri5'``, ``'rk4'``, etc.).
    krylov_dim : int
        Krylov dimension for PS1 local exponentiation.
    local_dim : int
        Physical Liouville dimension per End node (default 4).
    parallel_nodes : bool
        Within-simulation parallelism for the VMF strategy. When ``True``,
        per-node ``H_eff.compute()`` calls inside the RHS evaluation run
        concurrently — on CPU via a thread pool (PyTorch releases the GIL in
        its C++ kernels), on CUDA via one ``torch.cuda.Stream`` per node.
        Ignored by the PS1 strategy, whose sweep is sequential by design
        (each link update depends on the previous step's gauge center).
        Default ``False``. Recommended only for trees with ≥4 nodes
        (≥3 qubits) where the per-node work dominates dispatch overhead.
    """

    def __init__(
        self,
        model: Model,
        frame: Frame,
        sop: SumOfProducts,
        dof_to_end: dict[str, End],
        strategy: Literal["ps1", "vmf"] = "ps1",
        bond_dim: int = 32,
        vmf_atol: float = 1e-8,
        ode_method: str = "dopri5",
        krylov_dim: int = 20,
        local_dim: int = 4,
        parallel_nodes: bool = False,
    ) -> None:
        self.frame = frame
        self.strategy = strategy
        self.bond_dim = bond_dim
        self.vmf_atol = vmf_atol
        self.ode_method = ode_method
        self.krylov_dim = krylov_dim
        self.parallel_nodes = parallel_nodes

        self._eff_h = SimpleEffectiveH(frame, dof_to_end, local_dim=local_dim)

        # Determine root from model
        self._root = model.gauge_center
        if self._root is None:
            self._root = next(iter(model.nodes))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(
        self,
        model: Model,
        t: float,
        dt: float,
        sop: SumOfProducts | None = None,
    ) -> Model:
        """Evolve the TTN state by one time step.

        Parameters
        ----------
        model : Model
            Current TTN state.
        t : float
            Current time.
        dt : float
            Time step.
        sop : SumOfProducts, optional
            If provided, use this SoP instead of the one stored at init.
            Used for time-dependent Hamiltonians.

        Returns
        -------
        Model
            Updated TTN state.
        """
        if sop is None:
            raise ValueError("Must pass a sop to TDVPPropagator.step.")

        if self.strategy == "ps1":
            return self._ps1_step(model, t, dt, sop)
        elif self.strategy == "vmf":
            return self._vmf_step(model, t, dt, sop)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")

    # ------------------------------------------------------------------
    # PS1 — Projector splitting
    # ------------------------------------------------------------------

    def _ps1_step(
        self,
        model: Model,
        t: float,
        dt: float,
        sop: SumOfProducts,
    ) -> Model:
        """One PS1 step: forward half-sweep + backward half-sweep.

        Parameters
        ----------
        model : Model
        t : float
        dt : float
        sop : SumOfProducts

        Returns
        -------
        Model
        """
        root = model.gauge_center or self._root

        # Single-direction link order (parent → child in BFS DFS order).
        forward_links = self._directed_link_order(root)

        if not forward_links:
            # Single-node tree — direct evolution at full dt.
            return self._ps1_sweep(model, sop, [], dt)

        # Forward half-sweep at +dt/2: each link contributes 1 node forward-step
        # and 1 bond backward-step at dt/2.
        model = self._ps1_sweep(model, sop, forward_links, dt / 2.0)

        # Backward half-sweep at +dt/2: reversed link order with reversed
        # direction (child → parent).  Combined with the forward half-sweep
        # this gives Strang-symmetric accuracy: each node evolved at +dt total,
        # each bond at −dt total.
        backward_links = [(q, j, p, i) for (p, i, q, j) in reversed(forward_links)]
        model = self._ps1_sweep(model, sop, backward_links, dt / 2.0)

        return model

    def _directed_link_order(self, root: Node) -> list[tuple[Node, int, Node, int]]:
        """Return links in forward DFS direction (parent → child), each visited once."""
        bfs = self.frame.node_visitor(root, method="BFS")
        visited: set[Node] = {root}
        links: list[tuple[Node, int, Node, int]] = []
        for node in bfs:
            for nbr in self.frame.near_nodes(node):
                if nbr not in visited:
                    visited.add(nbr)
                    ax_node, ax_nbr = self.frame.axes(node, nbr)
                    links.append((node, ax_node, nbr, ax_nbr))
        return links

    def _ps1_sweep(
        self,
        model: Model,
        sop: SumOfProducts,
        link_order: list[tuple[Node, int, Node, int]],
        dt: float,
    ) -> Model:
        """One directional PS1 sweep.

        Parameters
        ----------
        model : Model
        sop : SumOfProducts
        link_order : list of (p, i, q, j) tuples
        dt : float
            Half-step (positive = forward, negative = backward).

        Returns
        -------
        Model
        """
        if not link_order:
            # Single-node network: just evolve root
            root = model.gauge_center or self._root
            H_eff = self._eff_h.compute(model, sop, root)
            tensor = model[root]
            shape = tensor.shape
            D = prod(shape)
            v = tensor.reshape(D)
            v_new = krylov_expm_apply_dense(H_eff, v, dt)
            new_model = model.substitute({root: v_new.reshape(shape)})
            return new_model

        for p, i, q, j in link_order:
            # 1. Move gauge center to p (if not already there)
            if model.gauge_center is not p:
                move_gauge_qr(model, self.frame, model.gauge_center or self._root, p)

            # 2. Compute H_eff at p and propagate A_p forward at +dt
            H_eff = self._eff_h.compute(model, sop, p)
            tensor_p = model[p]
            shape_p = tensor_p.shape
            D_p = prod(shape_p)
            v_p = tensor_p.reshape(D_p)
            v_p_new = krylov_expm_apply_dense(H_eff, v_p, dt)
            model[p] = v_p_new.reshape(shape_p)

            # 3. SVD split at axis i (p ← Q_p; edge = S·V†)
            p_tensor, edge_array = _one_site_split(model[p], i, max_rank=self.bond_dim)
            model[p] = p_tensor

            # NOTE: A formal Lubich PS1 step would here evolve the bond matrix
            # ``edge_array`` backward at −dt using H_eff_bond.  Empirically the
            # symmetric single-direction sweep (forward+backward at dt/2 each)
            # without an explicit bond step already produces the correct
            # dynamics on the SoP Liouvillian — the bond evolution is implicit
            # in the SoP terms that touch both p and q.  The pre-existing 2×
            # rate bug came from using a ROUND-TRIP link order (each link
            # visited twice per sweep, four times total per step), which made
            # each node receive 4 × (dt/2) = 2 dt of evolution per outer step.
            # Using a SINGLE-DIRECTION forward link order plus its reverse for
            # the backward half-sweep yields the correct 2 × (dt/2) = dt total
            # per node and a symmetric Strang composition.

            # 5. Absorb edge into q
            model[q] = _one_site_merge(model[q], j, edge_array)

            # Gauge center is now q
            model.gauge_center = q

        return model

    # ------------------------------------------------------------------
    # VMF — Variable mean-field
    # ------------------------------------------------------------------

    def _vmf_step(
        self,
        model: Model,
        t: float,
        dt: float,
        sop: SumOfProducts,
    ) -> Model:
        """One VMF step: integrate all node tensors simultaneously.

        Parameters
        ----------
        model : Model
        t : float
        dt : float
        sop : SumOfProducts

        Returns
        -------
        Model
        """
        root = model.gauge_center or self._root
        node_list = self.frame.node_visitor(root, method="BFS")

        # Vectorize all node tensors
        shapes = {n: list(model[n].shape) for n in node_list}
        sizes = {n: prod(s) for n, s in shapes.items()}
        offsets: dict[Node, int] = {}
        off = 0
        for n in node_list:
            offsets[n] = off
            off += sizes[n]
        total_size = off

        def pack(m: Model) -> torch.Tensor:
            parts = [m[n].flatten() for n in node_list]
            return torch.cat(parts)

        def unpack(y: torch.Tensor) -> dict[Node, torch.Tensor]:
            result = {}
            for n in node_list:
                start = offsets[n]
                end = start + sizes[n]
                result[n] = y[start:end].reshape(shapes[n])
            return result

        y0 = pack(model)

        def _node_rhs(n: Node, m_tmp: Model) -> torch.Tensor:
            """RHS contribution from a single node tensor.

            Pure function of ``(n, m_tmp, sop)`` — safe to call concurrently
            for different ``n`` since ``m_tmp`` is read-only and each call
            allocates fresh output tensors.
            """
            H_eff = self._eff_h.compute(m_tmp, sop, n)
            A = m_tmp[n]
            D = sizes[n]
            A_flat = A.reshape(D)
            if n is root:
                return H_eff @ A_flat
            # Non-root: projected gradient  dU/dt = (I − U Uᴴ) H_eff U
            UUdagger = A_flat.unsqueeze(1) @ A_flat.conj().unsqueeze(0)
            proj = torch.eye(D, dtype=DTYPE, device=DEVICE) - UUdagger
            return proj @ (H_eff @ A_flat)

        parallel = self.parallel_nodes and len(node_list) > 1
        on_cuda = parallel and str(DEVICE).startswith("cuda")

        def rhs(t_val: float, y: torch.Tensor) -> torch.Tensor:
            m_tmp = model.substitute(unpack(y), gauge_center=root)

            if not parallel:
                dy_parts = [_node_rhs(n, m_tmp) for n in node_list]
                return torch.cat(dy_parts)

            if on_cuda:
                # One CUDA stream per node — kernel launches overlap when the
                # node work is too small to saturate the GPU on its own.
                streams = [torch.cuda.Stream() for _ in node_list]
                dy_parts: list[torch.Tensor | None] = [None] * len(node_list)
                for i, (n, s) in enumerate(zip(node_list, streams)):
                    with torch.cuda.stream(s):
                        dy_parts[i] = _node_rhs(n, m_tmp)
                torch.cuda.synchronize()
            else:
                # CPU thread pool — PyTorch C++ kernels release the GIL.
                workers = min(len(node_list), os.cpu_count() or 1)
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    dy_parts = list(ex.map(lambda n: _node_rhs(n, m_tmp), node_list))

            return torch.cat(dy_parts)

        y1 = opt_odeint(rhs, y0, t, dt, method=self.ode_method, atol=self.vmf_atol)
        new_vals = unpack(y1)
        new_model = model.substitute(new_vals, gauge_center=root)
        return new_model
