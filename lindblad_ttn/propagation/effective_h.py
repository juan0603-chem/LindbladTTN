# coding: utf-8
"""Effective local Hamiltonian for TDVP, plus a bond-effective Hamiltonian
for the Lubich-style 1-site projector-splitting backward step.

For each SoP term ``(coeff, {dof: op})``, the effective H at a NODE ``t`` is
built by a TWO-PASS environment contraction:

Pass 1 — bottom-up (leaves → root, skipping ``t``):
    For each non-target node ``n``, compute a marginal matrix that summarises
    the contribution of ``n``'s entire subtree to the bond connecting ``n``
    to its parent.  Store as ``env_sub[n]``.

Pass 2 — top-down (root → leaves):
    For each non-root node ``n``, compute a marginal matrix from the "above"
    side — everything reachable from n's parent (excluding n's subtree).
    Store as ``env_above[n]``.  Used only by ``compute_bond`` (the formal
    Lubich PS1 bond step).  ``compute`` does NOT consume env_above for the
    target itself: PS1 re-roots at the target so the parent's subtree is
    naturally absorbed into env_sub, and VMF gauges the root so non-root
    sites only need their child subtrees in H_eff.

H_eff at NODE target:
    For each axis of the target tensor, collect the environment on that axis:
    End axes get the local op (or identity), Node axes get ``env_sub`` (child
    side) or ``env_above`` (parent side).  H_eff = Kronecker product of
    per-axis matrices.

H_eff at BOND (between ``p`` and ``q``, after p's QR/SVD split):
    Each SoP term contributes ``c · E_p ⊗ E_q`` where ``E_p`` is p's full
    environment (with p's tensor in its post-split isometric form), contracted
    leaving the bond axis open; similarly for ``E_q``.
"""

from __future__ import annotations

from math import prod
from typing import Optional

import torch

from lindblad_ttn.core.backend import DEVICE, DTYPE, kron_torch, multitransform
from lindblad_ttn.core.graph import End, Frame, Node
from lindblad_ttn.core.model import Model
from lindblad_ttn.core.sop import SumOfProducts


class SimpleEffectiveH:
    """Effective Hamiltonian via two-pass environment contraction.

    Parameters
    ----------
    frame : Frame
    dof_to_end : dict[str, End]
    local_dim : int
        Legacy parameter; tensors carry their own per-axis dimensions.  Kept
        for backward compatibility.
    """

    def __init__(
        self,
        frame: Frame,
        dof_to_end: dict[str, End],
        local_dim: int = 4,
    ) -> None:
        self.frame = frame
        self.dof_to_end = dof_to_end
        self.local_dim = local_dim
        self._end_to_dof: dict[End, str] = {v: k for k, v in dof_to_end.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        model: Model,
        sop: SumOfProducts,
        target: Node,
    ) -> torch.Tensor:
        """Compute H_eff at NODE ``target`` as a dense (D, D) matrix."""
        root = self._select_root(model)
        D = prod(model[target].shape)
        H_eff = torch.zeros(D, D, dtype=DTYPE, device=DEVICE)
        for coeff, op_dict in sop.terms:
            term_matrix = self._term_node_env(model, op_dict, target, root)
            if term_matrix is not None:
                H_eff = H_eff + coeff * term_matrix
        return H_eff

    def compute_bond(
        self,
        model: Model,
        sop: SumOfProducts,
        p: Node,
        q: Node,
        ax_p: int,
        ax_q: int,
    ) -> torch.Tensor:
        """Compute H_eff on the bond between ``p`` and ``q``.

        Returns a ``(D_p · D_q, D_p · D_q)`` matrix where ``D_p = shape(p)[ax_p]``
        and ``D_q = shape(q)[ax_q]``.  Index convention: ``vec(R)[α·D_q + β]``
        with ``R`` the bond matrix.
        """
        root = self._select_root(model)
        D_p = model[p].shape[ax_p]
        D_q = model[q].shape[ax_q]
        H_bond = torch.zeros(D_p * D_q, D_p * D_q, dtype=DTYPE, device=DEVICE)

        for coeff, op_dict in sop.terms:
            E_p = self._term_node_env_on_bond(model, op_dict, p, ax_p, root)
            E_q = self._term_node_env_on_bond(model, op_dict, q, ax_q, root)
            H_bond = H_bond + coeff * kron_torch(E_p, E_q)

        return H_bond

    # ------------------------------------------------------------------
    # Single-SoP-term env computations
    # ------------------------------------------------------------------

    def _select_root(self, model: Model) -> Node:
        root = model.gauge_center
        if root is None:
            root = next(iter(model.nodes))
        return root

    def _term_node_env(
        self,
        model: Model,
        op_dict: dict[str, torch.Tensor],
        target: Node,
        root: Node,
    ) -> torch.Tensor:
        """Per-term H_eff at a NODE target (Kron over the target's axes)."""
        env_sub, env_above, bfs_nodes = self._compute_envs(model, op_dict, root)
        target_shape = list(model[target].shape)
        axis_ops_target: dict[int, torch.Tensor] = {}

        for nbr in self.frame.near_points(target):
            ax_target, _ = self.frame.axes(target, nbr)
            if ax_target is None:
                continue
            if isinstance(nbr, End):
                dof = self._end_to_dof.get(nbr)
                if dof is not None and dof in op_dict:
                    axis_ops_target[ax_target] = op_dict[dof]
            elif isinstance(nbr, Node):
                # NOTE: Do NOT include env_above[target] here.  Both PS1 (when
                # the gauge center is target, the env contraction effectively
                # re-roots the tree at target so the parent's subtree is
                # absorbed into env_sub) and VMF (where the root absorbs the
                # "free" direction and non-root sites only need their child
                # subtrees in H_eff because the parent's contribution is
                # mediated through the projected gradient at the root)
                # already get the parent's contribution implicitly.  Adding
                # env_above[target] explicitly double-counts and was the
                # source of a 2× rate bug in earlier drafts; it also broke
                # VMF trace conservation on multi-qubit problems with
                # bond_dim > local_dim².  ``env_above`` is still computed
                # because compute_bond() (used for the formal Lubich PS1 bond
                # step) needs it on non-target nodes.
                parent_of_target = self._find_parent(target, bfs_nodes)
                if nbr is parent_of_target:
                    pass
                else:
                    sub_info = env_sub.get(nbr)
                    if sub_info is not None:
                        ax_in_target, sub_marg = sub_info
                        axis_ops_target[ax_in_target] = sub_marg

        return _kron_over_axes(target_shape, axis_ops_target)

    def _term_node_env_on_bond(
        self,
        model: Model,
        op_dict: dict[str, torch.Tensor],
        target: Node,
        keep_ax: int,
        root: Node,
    ) -> torch.Tensor:
        """Per-term contraction of ``target``'s environment leaving ``keep_ax`` open.

        Returns a ``(D_keep, D_keep)`` matrix where ``D_keep`` is the bond
        dimension on ``keep_ax``.
        """
        env_sub, env_above, bfs_nodes = self._compute_envs(model, op_dict, root)
        tensor = model[target]
        axis_ops: dict[int, torch.Tensor] = {}

        for nbr in self.frame.near_points(target):
            ax_target, _ = self.frame.axes(target, nbr)
            if ax_target is None or ax_target == keep_ax:
                continue
            if isinstance(nbr, End):
                dof = self._end_to_dof.get(nbr)
                if dof is not None and dof in op_dict:
                    axis_ops[ax_target] = op_dict[dof]
            elif isinstance(nbr, Node):
                parent_of_target = self._find_parent(target, bfs_nodes)
                if nbr is parent_of_target:
                    above_info = env_above.get(target)
                    if above_info is not None:
                        ax_in_target, above_marg = above_info
                        axis_ops[ax_in_target] = above_marg
                else:
                    sub_info = env_sub.get(nbr)
                    if sub_info is not None:
                        ax_in_target, sub_marg = sub_info
                        axis_ops[ax_in_target] = sub_marg

        dressed = multitransform(axis_ops, tensor) if axis_ops else tensor
        return _contract_except(tensor, dressed, keep_ax)

    # ------------------------------------------------------------------
    # Unified env builder (used by both compute and compute_bond)
    # ------------------------------------------------------------------

    def _compute_envs(
        self,
        model: Model,
        op_dict: dict[str, torch.Tensor],
        root: Node,
    ) -> tuple[
        dict[Node, tuple[int, torch.Tensor]],
        dict[Node, tuple[int, torch.Tensor]],
        list[Node],
    ]:
        """Build env_sub and env_above for ALL nodes in the tree.

        Returns
        -------
        env_sub : dict
            For each non-root node ``n``, ``(ax_in_parent, marg)`` summarising
            ``n``'s subtree on the bond toward its parent.
        env_above : dict
            For each non-root node ``n``, ``(ax_in_self_toward_parent, marg)``
            summarising everything "above" ``n``.
        bfs_nodes : list[Node]
            BFS order rooted at ``root``.
        """
        bfs_nodes = self.frame.node_visitor(root, method="BFS")
        bottom_up = list(reversed(bfs_nodes))
        node_axes = self.frame.get_node_axes(root)

        env_sub: dict[Node, tuple[int, torch.Tensor]] = {}

        for node in bottom_up:
            tensor = model[node]
            parent_ax = node_axes.get(node)

            axis_ops: dict[int, torch.Tensor] = {}
            for nbr in self.frame.near_points(node):
                ax_node, _ = self.frame.axes(node, nbr)
                if ax_node is None:
                    continue
                if isinstance(nbr, End):
                    dof = self._end_to_dof.get(nbr)
                    if dof is not None and dof in op_dict:
                        axis_ops[ax_node] = op_dict[dof]
                elif isinstance(nbr, Node):
                    child_info = env_sub.get(nbr)
                    if child_info is not None:
                        ax_in_self, marg = child_info
                        axis_ops[ax_in_self] = marg

            dressed = multitransform(axis_ops, tensor) if axis_ops else tensor

            if parent_ax is not None:
                marg = _contract_except(tensor, dressed, parent_ax)
                parent_node = self._find_parent(node, bfs_nodes)
                if parent_node is not None:
                    ax_in_parent, _ = self.frame.axes(parent_node, node)
                    env_sub[node] = (ax_in_parent, marg)

        env_above: dict[Node, tuple[int, torch.Tensor]] = {}

        for node in bfs_nodes:
            for nbr in self.frame.near_nodes(node):
                parent_of_nbr = self._find_parent(nbr, bfs_nodes)
                if parent_of_nbr is not node:
                    continue  # nbr is node's parent, not child

                ax_node_to_nbr, ax_nbr_to_node = self.frame.axes(node, nbr)

                axis_ops_node: dict[int, torch.Tensor] = {}
                for node_nbr in self.frame.near_points(node):
                    ax_n, _ = self.frame.axes(node, node_nbr)
                    if ax_n is None:
                        continue
                    if node_nbr is nbr:
                        continue  # skip bond toward the child we're computing for
                    if isinstance(node_nbr, End):
                        dof = self._end_to_dof.get(node_nbr)
                        if dof is not None and dof in op_dict:
                            axis_ops_node[ax_n] = op_dict[dof]
                    elif isinstance(node_nbr, Node):
                        parent_of_node = self._find_parent(node, bfs_nodes)
                        if node_nbr is parent_of_node:
                            above_info = env_above.get(node)
                            if above_info is not None:
                                ax_in_node, above_marg = above_info
                                axis_ops_node[ax_in_node] = above_marg
                        else:
                            sib_info = env_sub.get(node_nbr)
                            if sib_info is not None:
                                ax_in_node, sib_marg = sib_info
                                axis_ops_node[ax_in_node] = sib_marg

                tensor_node = model[node]
                dressed_node = (
                    multitransform(axis_ops_node, tensor_node)
                    if axis_ops_node else tensor_node
                )
                marg_above = _contract_except(tensor_node, dressed_node, ax_node_to_nbr)
                env_above[nbr] = (ax_nbr_to_node, marg_above)

        return env_sub, env_above, bfs_nodes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_parent(self, node: Node, bfs_nodes: list[Node]) -> Node | None:
        if node not in bfs_nodes:
            return None
        idx = bfs_nodes.index(node)
        for cand in bfs_nodes[:idx]:
            if node in self.frame.near_nodes(cand):
                return cand
        return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _contract_except(
    tensor: torch.Tensor,
    dressed: torch.Tensor,
    keep_ax: int,
) -> torch.Tensor:
    """Contract all axes of ``tensor`` except ``keep_ax``.

    M[i, j] = Σ_{other axes} tensor*[..., i, ...] · dressed[..., j, ...].
    """
    shape = list(tensor.shape)
    d_keep = shape[keep_ax]
    other_shape = [shape[a] for a in range(tensor.ndim) if a != keep_ax]
    D_other = prod(other_shape) if other_shape else 1

    A = tensor.moveaxis(keep_ax, -1).reshape(D_other, d_keep)
    D = dressed.moveaxis(keep_ax, -1).reshape(D_other, d_keep)
    return A.conj().T @ D


def _kron_over_axes(
    shape: list[int],
    axis_ops: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Build the Kronecker product of per-axis operators.  Missing axes → identity."""
    result: torch.Tensor | None = None
    for ax, dim in enumerate(shape):
        M = axis_ops.get(ax, torch.eye(dim, dtype=DTYPE, device=DEVICE))
        if result is None:
            result = M
        else:
            result = kron_torch(result, M)
    if result is None:
        D = prod(shape)
        result = torch.eye(D, dtype=DTYPE, device=DEVICE)
    return result
