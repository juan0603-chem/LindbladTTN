# coding: utf-8
"""LindbladTTN — the single user-facing class.

Two complementary constructor surfaces are provided:

Legacy spin-1/2 path (every site is a qubit)::

    sz = np.array([[1,0],[0,-1]], dtype=complex)
    sm = np.array([[0,0],[1,0]], dtype=complex)
    solver = LindbladTTN(
        H0=0.5 * sz, f=None, V=None,
        L_ops=[(0.1, sm)], n_sites=1, bond_dim=4,
    )

Heterogeneous path (M1+ — bosons, higher spins, mixed dimensions)::

    from lindblad_ttn.sites import spin_half_site, boson_site

    q = spin_half_site("q0")
    c = boson_site(8, "c0")

    solver = LindbladTTN(
        sites    = [q, c],
        H_terms  = [
            (omega_q, {q.name: q.sz}),
            (omega_c, {c.name: c.n}),
            (g,       {q.name: q.sp, c.name: c.a}),
            (g,       {q.name: q.sm, c.name: c.adag}),
        ],
        L_terms  = [(gamma, {q.name: q.sm}), (kappa, {c.name: c.a})],
        bond_dim = 16,
        topology = "tree",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import numpy as np
import torch
from tqdm import tqdm

from lindblad_ttn.core.backend import DEVICE, DTYPE, set_device, set_dtype, to_torch
from lindblad_ttn.core.graph import End, Frame, Node
from lindblad_ttn.core.model import Model
from lindblad_ttn.physics.frame_factory import LindbladFrameFactory
from lindblad_ttn.physics.liouville import expect_from_vec, trace_from_vec, vec, unvec
from lindblad_ttn.physics.liouville_nd import (
    permute_interleaved_to_rowmajor,
    permute_rowmajor_to_interleaved,
)
from lindblad_ttn.physics.liouvillian import LiouvillianSoP
from lindblad_ttn.physics.liouvillian_nd import LiouvillianSoPND
from lindblad_ttn.propagation.tdvp import TDVPPropagator
from lindblad_ttn.sites.base import Site
from lindblad_ttn.time_dependent import TimeDependentSoP


def _resolve_dtype(dtype: "str | torch.dtype") -> torch.dtype:
    if isinstance(dtype, str):
        key = dtype.lower()
        if key in ("complex64", "c64", "single"):
            return torch.complex64
        if key in ("complex128", "c128", "double"):
            return torch.complex128
        raise ValueError(f"Unknown dtype string: {dtype!r}")
    return dtype


HTermND = tuple[complex, dict[str, np.ndarray]]
LTermND = tuple[float, dict[str, np.ndarray]]
DriveND = tuple[Callable[[float], float], list[HTermND]]


@dataclass
class LindbladResult:
    """Results from :meth:`LindbladTTN.run`."""

    times: np.ndarray
    expect: dict[int, np.ndarray] = field(default_factory=dict)
    rho_final: np.ndarray = field(default_factory=lambda: np.array([]))
    norm: np.ndarray = field(default_factory=lambda: np.array([]))
    bond_dims: np.ndarray = field(default_factory=lambda: np.array([]))

    def save_txt(self, path: str = "lindblad_result.txt") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rho = np.asarray(self.rho_final)
        d = rho.shape[0]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("=" * 72 + "\n")
            fh.write("  LindbladTTN result\n")
            fh.write("=" * 72 + "\n\n")
            fh.write(f"Density matrix  rho_final  ({d} x {d})\n")
            fh.write("-" * 72 + "\n\n")

            def _fmt(M: np.ndarray, label: str) -> None:
                fh.write(f"  {label}\n")
                fh.write("         " + "".join(f"  [{j:3d}]" for j in range(d)) + "\n")
                for i in range(d):
                    row = "".join(f"  {M[i, j]:+.6f}" for j in range(d))
                    fh.write(f"  [{i:3d}] {row}\n")
                fh.write("\n")

            _fmt(rho.real, "Re(rho)")
            _fmt(rho.imag, "Im(rho)")
            _fmt(np.abs(rho), "|rho|  ")

            fh.write("-" * 72 + "\n")
            fh.write("  Summary\n")
            fh.write("-" * 72 + "\n")
            tr = np.trace(rho).real
            pur = np.real(np.trace(rho @ rho))
            fh.write(f"  Tr(rho)            = {tr:.10f}\n")
            fh.write(f"  Purity Tr(rho^2)   = {pur:.10f}\n")
            fh.write(f"  ||rho - rho†||     = {np.linalg.norm(rho - rho.conj().T):.2e}\n")
            eig = np.linalg.eigvalsh(rho).real
            fh.write(f"  min eigenvalue     = {eig.min():.4e}\n")
            fh.write(f"  max eigenvalue     = {eig.max():.4f}\n\n")

            if len(self.times) > 0:
                fh.write("-" * 72 + "\n")
                fh.write("  Time series\n")
                fh.write("-" * 72 + "\n")
                n_obs = len(self.expect)
                obs_h = "".join(f"  Re<O{k}>     Im<O{k}>  " for k in range(n_obs))
                fh.write(f"  {'t':>12}  {'Tr(rho)':>10}  {'bond_dim':>8}{obs_h}\n")
                fh.write("  " + "-" * (33 + 24 * n_obs) + "\n")
                for i, t in enumerate(self.times):
                    bd = int(self.bond_dims[i]) if i < len(self.bond_dims) else 0
                    nrm = float(self.norm[i]) if i < len(self.norm) else float("nan")
                    cols = "".join(
                        f"  {self.expect[k][i].real:+.6f}  {self.expect[k][i].imag:+.6f}"
                        for k in range(n_obs)
                    )
                    fh.write(f"  {t:>12.6f}  {nrm:>10.8f}  {bd:>8d}{cols}\n")
                fh.write("\n")
        print(f"[LindbladTTN] Result saved to {path}")


class LindbladTTN:
    """Lindblad master equation solver using Tree Tensor Networks and TDVP.

    Parameters
    ----------
    H0 : np.ndarray, optional
        Legacy: static Hamiltonian, shape ``(2^N, 2^N)``.  Used together with
        ``n_sites=`` for the spin-1/2 path.
    f : callable, optional
        Legacy: drive envelope ``f(t) -> float`` (single-drive shortcut).
    V : np.ndarray, optional
        Legacy: drive operator, shape ``(2^N, 2^N)``.
    drives : list of (callable, ndarray), optional
        Legacy: multiple drives ``H(t) = H₀ + Σᵢ fᵢ(t)·Vᵢ``.
    L_ops : list of (gamma, L)
        Legacy: jump operators.
    n_sites : int
        Legacy: number of qubits.
    sites : list of :class:`~lindblad_ttn.sites.Site`, optional
        New API: heterogeneous sites (spin-1/2, spin-S, boson, …).  When
        provided, ``H_terms`` and ``L_terms`` must use site-DOF dictionaries
        instead of full dense matrices.  Mutually exclusive with the legacy
        ``H0/V/L_ops/n_sites`` parameters.
    H_terms : list of (coeff, op_dict), optional
        New API: each entry contributes ``coeff · (⊗_s op_s)`` to the
        Hamiltonian.  Missing sites are identity.
    L_terms : list of (gamma, op_dict), optional
        New API: each entry contributes a Lindblad dissipator with rate
        ``gamma`` and jump operator ``L = ⊗_s op_s``.
    drives_nd : list of (callable, list[(coeff, op_dict)]), optional
        New API: time-dependent driving channels.  Each entry is
        ``(f_i, V_terms_i)`` where ``V_terms_i`` is a list of H-terms in the
        same format as ``H_terms``.
    bond_dim : int
        Max TTN bond dimension.
    topology : {'train', 'tree', 'cavity_centered'}
    cavity_dofs : list of str, optional
        Required when ``topology='cavity_centered'``.
    device, strategy, vmf_atol, ode_method, krylov_dim, dtype, num_threads,
    parallel_nodes : see the class docstring of the legacy interface.
    """

    def __init__(
        self,
        # Legacy spin-1/2 surface
        H0: np.ndarray | None = None,
        f: Callable[[float], float] | None = None,
        V: np.ndarray | None = None,
        L_ops: list[tuple[float, np.ndarray]] | None = None,
        n_sites: int | None = None,
        drives: list[tuple[Callable[[float], float], np.ndarray]] | None = None,
        # Heterogeneous-site surface (M1+)
        sites: Sequence[Site] | None = None,
        H_terms: Iterable[HTermND] | None = None,
        L_terms: Iterable[LTermND] | None = None,
        drives_nd: Sequence[DriveND] | None = None,
        # Shared
        bond_dim: int = 32,
        topology: Literal["train", "tree", "cavity_centered"] = "tree",
        cavity_dofs: Sequence[str] | None = None,
        device: str = "cpu",
        strategy: Literal["ps1", "vmf"] = "ps1",
        vmf_atol: float = 1e-8,
        ode_method: str = "dopri5",
        krylov_dim: int = 20,
        dtype: "str | torch.dtype" = "complex128",
        num_threads: int | None = None,
        parallel_nodes: bool = False,
    ) -> None:
        set_device(device)
        set_dtype(_resolve_dtype(dtype))
        if num_threads is not None:
            torch.set_num_threads(int(num_threads))

        # ------------------------------------------------------------------
        # Dispatch: heterogeneous-site path vs. legacy spin-1/2 path
        # ------------------------------------------------------------------
        using_sites = sites is not None
        using_legacy = (H0 is not None) or (n_sites is not None) or (L_ops is not None)

        if using_sites and using_legacy:
            raise ValueError(
                "Pass either `sites=`/`H_terms=`/`L_terms=` (new API) "
                "or `H0`/`n_sites`/`L_ops` (legacy spin-1/2), not both."
            )
        if not using_sites and not using_legacy:
            raise ValueError(
                "LindbladTTN requires either `sites=` (new API) or "
                "`n_sites=`/`H0` (legacy spin-1/2)."
            )

        self.bond_dim = bond_dim
        self.strategy = strategy
        self.vmf_atol = vmf_atol
        self.ode_method = ode_method
        self.krylov_dim = krylov_dim
        self.parallel_nodes = parallel_nodes
        self._topology = topology
        self._cavity_dofs = list(cavity_dofs) if cavity_dofs else None

        if using_sites:
            self._init_heterogeneous(
                sites=list(sites),
                H_terms=list(H_terms) if H_terms else [],
                L_terms=list(L_terms) if L_terms else [],
                drives_nd=list(drives_nd) if drives_nd else [],
            )
        else:
            self._init_legacy(
                H0=H0, f=f, V=V, L_ops=L_ops or [], n_sites=n_sites, drives=drives,
            )

    # ------------------------------------------------------------------
    # Initialisation paths
    # ------------------------------------------------------------------

    def _init_legacy(
        self,
        H0: np.ndarray | None,
        f: Callable[[float], float] | None,
        V: np.ndarray | None,
        L_ops: list[tuple[float, np.ndarray]],
        n_sites: int | None,
        drives: list[tuple[Callable[[float], float], np.ndarray]] | None,
    ) -> None:
        if n_sites is None or n_sites < 1:
            raise ValueError("Legacy path: n_sites must be ≥ 1.")
        self.n_sites = int(n_sites)
        self._site_dims = [2] * self.n_sites
        self._dof_names = [f"q{i}" for i in range(self.n_sites)]
        self._d = prod(self._site_dims)
        self._d_liouv = self._d * self._d

        if drives is not None and (f is not None or V is not None):
            raise ValueError("Pass either `drives=[(f, V), ...]` or `f, V`, not both.")
        if drives is None:
            if f is not None and V is None:
                raise ValueError("f provided but V is None.")
            if V is not None and f is None:
                raise ValueError("V provided but f is None.")
            self._drives = [(f, V)] if (f is not None and V is not None) else []
        else:
            self._drives = [(fn, np.asarray(V_i, dtype=complex)) for fn, V_i in drives]

        print(f"[LindbladTTN] Building SoP for {self.n_sites} qubit(s)…")
        Vs = [V_i for (_, V_i) in self._drives] if self._drives else None
        self._liouv = LiouvillianSoP(self.n_sites, H0, L_ops, Vs=Vs)
        print(f"  {self._liouv}")

        self._build_topology_and_td()

    def _init_heterogeneous(
        self,
        sites: list[Site],
        H_terms: list[HTermND],
        L_terms: list[LTermND],
        drives_nd: list[DriveND],
    ) -> None:
        if not sites:
            raise ValueError("sites list is empty.")
        # Validate operators against site dimensions
        seen_names: set[str] = set()
        site_by_name: dict[str, Site] = {}
        for s in sites:
            if s.name in seen_names:
                raise ValueError(f"Duplicate site name: {s.name!r}.")
            seen_names.add(s.name)
            site_by_name[s.name] = s

        def _check_op_dict(op_dict: dict[str, np.ndarray], context: str) -> None:
            for dof, op in op_dict.items():
                if dof not in site_by_name:
                    raise ValueError(
                        f"{context}: DOF {dof!r} not among site names "
                        f"{sorted(site_by_name)}."
                    )
                d = site_by_name[dof].dim
                op = np.asarray(op)
                if op.shape != (d, d):
                    raise ValueError(
                        f"{context}: operator on {dof!r} has shape {op.shape}, "
                        f"expected ({d}, {d})."
                    )

        for i, (_, od) in enumerate(H_terms):
            _check_op_dict(od, f"H_terms[{i}]")
        for i, (_, od) in enumerate(L_terms):
            _check_op_dict(od, f"L_terms[{i}]")
        for di, (_, vt) in enumerate(drives_nd):
            for ti, (_, od) in enumerate(vt):
                _check_op_dict(od, f"drives_nd[{di}].V_terms[{ti}]")

        self.n_sites = len(sites)
        self._sites = sites
        self._site_dims = [s.dim for s in sites]
        self._dof_names = [s.name for s in sites]
        self._d = prod(self._site_dims)
        self._d_liouv = self._d * self._d
        self._drives = []  # legacy slot, unused

        # Build SoP
        print(
            f"[LindbladTTN] Building heterogeneous SoP for "
            f"{self.n_sites} site(s), dims={self._site_dims}…"
        )
        V_terms_list = [list(vt) for (_, vt) in drives_nd] if drives_nd else None
        self._liouv_nd = LiouvillianSoPND(H_terms, L_terms, V_terms_list)
        print(f"  {self._liouv_nd}")

        # Per-drive callable list (for TimeDependentSoP wiring)
        self._drives_nd_callables = [fn for fn, _ in drives_nd]

        self._build_topology_and_td(heterogeneous=True)

    # ------------------------------------------------------------------
    # Common topology & time-dependence wiring
    # ------------------------------------------------------------------

    def _build_topology_and_td(self, heterogeneous: bool = False) -> None:
        factory = LindbladFrameFactory(dof_names=self._dof_names)
        if self._topology == "train":
            self._frame, self._root, self._dof_to_end = factory.train()
        elif self._topology == "tree":
            self._frame, self._root, self._dof_to_end = factory.balanced_tree()
        elif self._topology == "cavity_centered":
            if not self._cavity_dofs:
                raise ValueError("`cavity_dofs=` required for topology='cavity_centered'.")
            self._frame, self._root, self._dof_to_end = factory.cavity_centered(
                cavity_dofs=self._cavity_dofs,
            )
        else:
            raise ValueError(f"Unknown topology: {self._topology!r}")

        self._node_order = self._frame.node_visitor(self._root, method="BFS")

        if heterogeneous:
            sop_H0 = self._liouv_nd.build_sop_H0()
            sop_Vs = self._liouv_nd.build_sop_Vs()
            td_drives = list(zip(self._drives_nd_callables, sop_Vs))
        else:
            sop_H0 = self._liouv.build_sop_H0()
            td_drives = list(zip(
                (fn for fn, _ in self._drives),
                self._liouv.build_sop_Vs(),
            ))

        self._td_sop = TimeDependentSoP(
            sop_H0,
            drives=td_drives if td_drives else None,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        rho0: np.ndarray,
        t_span: tuple[float, float],
        dt: float,
        observables: list[np.ndarray] | None = None,
        save_every: int = 1,
        verbose: bool = True,
    ) -> LindbladResult:
        """Run the TDVP evolution."""
        if observables is None:
            observables = []

        rho0 = np.asarray(rho0, dtype=complex)
        if rho0.ndim == 1:
            rho0 = np.outer(rho0, rho0.conj())
        rho0 = rho0 / np.trace(rho0)

        v_row = vec(to_torch(rho0))
        v_inter = to_torch(permute_rowmajor_to_interleaved(v_row.cpu().numpy(), self._site_dims))
        model = self._initialize_model(v_inter)

        sop0 = self._td_sop.evaluate(t_span[0])
        propagator = TDVPPropagator(
            model=model,
            frame=self._frame,
            sop=sop0,
            dof_to_end=self._dof_to_end,
            strategy=self.strategy,
            bond_dim=self.bond_dim,
            vmf_atol=self.vmf_atol,
            ode_method=self.ode_method,
            krylov_dim=self.krylov_dim,
            local_dim=4,  # dead parameter inside; tensors carry their own dims
            parallel_nodes=self.parallel_nodes,
        )

        t = float(t_span[0])
        n_steps = max(1, int(round((float(t_span[1]) - t) / dt)))

        saved_times: list[float] = []
        saved_norms: list[float] = []
        saved_expects: list[list[complex]] = [[] for _ in observables]
        saved_bond_dims: list[int] = []
        obs_torch = [to_torch(O) for O in observables]

        iterator = range(n_steps)
        if verbose:
            iterator = tqdm(iterator, desc="TDVP", unit="step")

        for step_idx in iterator:
            sop_t = self._td_sop.evaluate_midpoint(t, dt)
            model = propagator.step(model, t, dt, sop=sop_t)
            t = float(t_span[0]) + (step_idx + 1) * dt

            if (step_idx + 1) % save_every == 0:
                v_inter_now = self._extract_vec(model).cpu().numpy()
                v_row_now = to_torch(permute_interleaved_to_rowmajor(v_inter_now, self._site_dims))
                saved_times.append(t)
                norm_t = float(trace_from_vec(v_row_now, self._d).real)
                saved_norms.append(norm_t)
                for i, O in enumerate(obs_torch):
                    saved_expects[i].append(expect_from_vec(v_row_now, O, self._d))
                saved_bond_dims.append(self._max_bond_dim(model))

        v_inter_final = self._extract_vec(model).cpu().numpy()
        v_row_final = permute_interleaved_to_rowmajor(v_inter_final, self._site_dims)
        rho_final = unvec(to_torch(v_row_final), self._d).cpu().numpy()

        return LindbladResult(
            times=np.array(saved_times),
            expect={i: np.array(saved_expects[i]) for i in range(len(observables))},
            rho_final=rho_final,
            norm=np.array(saved_norms),
            bond_dims=np.array(saved_bond_dims, dtype=int),
        )

    # ------------------------------------------------------------------
    # Per-site physical-dimension helpers
    # ------------------------------------------------------------------

    def _liouv_dim_of_dof(self, dof_name: str) -> int:
        """Liouville dimension d_s² for a given DOF."""
        idx = self._dof_names.index(dof_name)
        d = self._site_dims[idx]
        return d * d

    # ------------------------------------------------------------------
    # Model initialisation via MPS-style sweep
    # ------------------------------------------------------------------

    def _initialize_model(self, vec_rho0: torch.Tensor) -> Model:
        """Build TTN from vectorised ρ₀ (interleaved Liouville)."""
        if self.n_sites == 1:
            return Model({self._root: vec_rho0.clone()}, gauge_center=self._root)

        node_ends, node_children, node_parent = self._classify_nodes()
        valuation: dict[Node, torch.Tensor] = {}

        # `remaining` is the still-to-be-split tail of the interleaved vec.
        # For the root pass it is the full (∏d_s²,) tensor; for subsequent
        # nodes it is shape (bond_in, flat_rest).
        remaining: torch.Tensor = vec_rho0.clone()

        for node in self._node_order:
            ends = node_ends[node]
            children = node_children[node]
            parent = node_parent[node]

            n_ends = len(ends)
            d_phys = prod(self._end_liouv_dims(ends)) if n_ends else 1
            has_children = len(children) > 0

            if parent is None:
                # ---- ROOT ----
                d_rest = remaining.numel() // max(d_phys, 1)

                if not has_children:
                    valuation[node] = remaining.reshape(-1)
                    remaining = torch.ones(1, dtype=DTYPE, device=DEVICE)
                    continue

                mat = remaining.reshape(d_phys, d_rest)
                U, S, Vh = self._svd_split(mat, self.bond_dim)
                d_bond = U.shape[1]
                valuation[node] = self._assemble_tensor_root(U, d_phys, d_bond, ends)
                remaining = (S.unsqueeze(1) * Vh)
            else:
                # ---- NON-ROOT ----
                d_bond_in = remaining.shape[0]
                flat_rest = remaining.numel() // d_bond_in

                if not has_children or flat_rest <= d_phys:
                    if d_phys > 0 and flat_rest >= d_phys:
                        tensor = remaining.reshape(d_bond_in, d_phys)
                    else:
                        tensor = remaining.reshape(d_bond_in, -1)
                    valuation[node] = tensor
                    remaining = torch.ones(1, dtype=DTYPE, device=DEVICE)
                    continue

                d_rest = flat_rest // d_phys
                mat = remaining.reshape(d_bond_in * d_phys, d_rest)
                U, S, Vh = self._svd_split(mat, self.bond_dim)
                d_bond_out = U.shape[1]
                valuation[node] = self._assemble_tensor_nonroot(
                    U, d_bond_in, d_phys, d_bond_out, ends
                )
                remaining = (S.unsqueeze(1) * Vh)

        for node in self._node_order:
            if node not in valuation:
                shape = self._default_shape(
                    node, node_parent[node], node_children[node], node_ends[node]
                )
                t = torch.zeros(prod(shape), dtype=DTYPE, device=DEVICE)
                t[0] = 1.0
                valuation[node] = t.reshape(shape)

        return Model(valuation, gauge_center=self._root)

    def _end_liouv_dims(self, ends: list[str]) -> list[int]:
        """For each DOF name in ``ends``, return its Liouville dim d_s²."""
        return [self._liouv_dim_of_dof(dof) for dof in ends]

    def _svd_split(
        self, mat: torch.Tensor, max_bond: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cpu_mat = mat.cpu()
        U, S, Vh = torch.linalg.svd(cpu_mat, full_matrices=False)
        U = U.to(DEVICE)
        S = S.to(device=DEVICE, dtype=torch.float64)
        Vh = Vh.to(DEVICE)
        nat_rank = S.shape[0]
        trunc_rank = min(nat_rank, max_bond)
        target = min(nat_rank, max_bond)
        U_t = U[:, :trunc_rank]
        S_t = S[:trunc_rank]
        Vh_t = Vh[:trunc_rank, :]
        if trunc_rank < target:
            pad = target - trunc_rank
            U_t = torch.cat([U_t, torch.zeros(U_t.shape[0], pad, dtype=DTYPE, device=DEVICE)], dim=1)
            S_t = torch.cat([S_t, torch.zeros(pad, dtype=torch.float64, device=DEVICE)])
            Vh_t = torch.cat([Vh_t, torch.zeros(pad, Vh_t.shape[1], dtype=DTYPE, device=DEVICE)], dim=0)
        return U_t, S_t, Vh_t

    def _classify_nodes(self) -> tuple[
        dict[Node, list[str]],
        dict[Node, list[Node]],
        dict[Node, Node | None],
    ]:
        """Return (node_ends_by_DOF_name, node_children, node_parent)."""
        node_ends: dict[Node, list[str]] = {}
        node_children: dict[Node, list[Node]] = {}
        node_parent: dict[Node, Node | None] = {}

        bfs = self._node_order
        parent_map: dict[Node, Node | None] = {self._root: None}
        for node in bfs:
            for nbr in self._frame.near_nodes(node):
                if nbr not in parent_map:
                    parent_map[nbr] = node

        end_to_dof = {end: dof for dof, end in self._dof_to_end.items()}

        for node in bfs:
            ends: list[str] = []
            children: list[Node] = []
            for nbr in self._frame.near_points(node):
                if isinstance(nbr, End):
                    dof = end_to_dof.get(nbr)
                    if dof is not None:
                        ends.append(dof)
                elif isinstance(nbr, Node):
                    if parent_map.get(nbr) is node:
                        children.append(nbr)
            node_ends[node] = ends
            node_children[node] = children
            node_parent[node] = parent_map.get(node)

        return node_ends, node_children, node_parent

    def _assemble_tensor_root(
        self,
        U: torch.Tensor,
        d_phys: int,
        d_bond: int,
        ends: list[str],
    ) -> torch.Tensor:
        phys_shape = self._end_liouv_dims(ends)
        if len(phys_shape) <= 1:
            return U.reshape(d_phys, d_bond)
        return U.reshape(phys_shape + [d_bond])

    def _assemble_tensor_nonroot(
        self,
        U: torch.Tensor,
        d_bond_in: int,
        d_phys: int,
        d_bond_out: int,
        ends: list[str],
    ) -> torch.Tensor:
        phys_shape = self._end_liouv_dims(ends)
        if len(phys_shape) == 0:
            return U.reshape(d_bond_in, d_bond_out)
        if len(phys_shape) == 1:
            return U.reshape(d_bond_in, phys_shape[0], d_bond_out)
        return U.reshape([d_bond_in] + phys_shape + [d_bond_out])

    def _default_shape(
        self,
        node: Node,
        parent: Node | None,
        children: list[Node],
        ends: list[str],
    ) -> list[int]:
        shape: list[int] = []
        if parent is not None:
            shape.append(min(self.bond_dim, self._d_liouv))
        for dof in ends:
            shape.append(self._liouv_dim_of_dof(dof))
        for _ in children:
            shape.append(min(self.bond_dim, self._d_liouv))
        return shape if shape else [self._d_liouv]

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    def _extract_vec(self, model: Model) -> torch.Tensor:
        root = self._root
        bfs = self._frame.node_visitor(root, method="BFS")
        bottom_up = list(reversed(bfs))

        contracted: dict[Node, torch.Tensor] = {}
        for node in bottom_up:
            tensor = model[node].clone()
            parent_map = self._get_parent_map(bfs)
            children_of_node = [
                n for n in self._frame.near_nodes(node)
                if parent_map.get(n) is node
            ]

            def child_axis(c: Node) -> int:
                ax, _ = self._frame.axes(node, c)
                return ax if ax is not None else -1

            children_sorted = sorted(children_of_node, key=child_axis, reverse=True)
            for child in children_sorted:
                ax_node, ax_child = self._frame.axes(node, child)
                tensor = torch.tensordot(
                    tensor, contracted[child], dims=([ax_node], [ax_child])
                )
            contracted[node] = tensor

        flat = contracted[root].flatten()
        return self._reorder_by_dof_order(flat, model)

    def _reorder_by_dof_order(self, flat: torch.Tensor, model: Model) -> torch.Tensor:
        """Reorder a contracted root-down vector so DOFs appear in self._dof_names order.

        After bottom-up contraction the residual tensor's axes follow the order
        in which they were appended during BFS (Ends of the root first, then
        children's Ends).  If that order differs from ``self._dof_names``, we
        permute now so callers see a deterministic interleaved layout.
        """
        # Reconstruct the BFS-driven DOF appearance order.
        node_ends, _, _ = self._classify_nodes()
        bfs_order: list[str] = []
        for node in self._node_order:
            bfs_order.extend(node_ends[node])

        if bfs_order == self._dof_names:
            return flat

        # Each DOF carries Liouville dim d_s²; reshape and transpose.
        liouv_dims_bfs = [self._liouv_dim_of_dof(d) for d in bfs_order]
        # Map: position of dof in bfs_order → position in self._dof_names.
        target_positions = [bfs_order.index(d) for d in self._dof_names]
        return flat.reshape(liouv_dims_bfs).permute(target_positions).reshape(-1)

    def _get_parent_map(self, bfs: list[Node]) -> dict[Node, Node | None]:
        pm: dict[Node, Node | None] = {bfs[0]: None}
        for node in bfs:
            for nbr in self._frame.near_nodes(node):
                if nbr not in pm:
                    pm[nbr] = node
        return pm

    def _max_bond_dim(self, model: Model) -> int:
        max_d = 1
        for node in model.nodes:
            for nbr in self._frame.near_nodes(node):
                ax, _ = self._frame.axes(node, nbr)
                if ax is not None:
                    shape = model.shape(node)
                    if ax < len(shape):
                        max_d = max(max_d, shape[ax])
        return max_d

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def frame(self) -> Frame:
        return self._frame

    @property
    def site_dims(self) -> list[int]:
        return list(self._site_dims)

    @property
    def dof_names(self) -> list[str]:
        return list(self._dof_names)
