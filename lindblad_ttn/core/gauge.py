# coding: utf-8
"""SVD-based gauge operations for TTN states.

Functions in this module manipulate the orthogonality center of a
:class:`~lindblad_ttn.core.model.Model` by performing QR or SVD
decompositions along bonds.
"""

from __future__ import annotations

from math import prod

import torch

from lindblad_ttn.core.backend import DEVICE, DTYPE, svd_truncate
from lindblad_ttn.core.graph import Frame, Node
from lindblad_ttn.core.model import Model


def qr_decompose(
    tensor: torch.Tensor,
    legs_left: list[int],
    legs_right: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """QR decomposition of a tensor split into "left" and "right" groups of legs.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.
    legs_left : list[int]
        Axes that form the rows of the matrix.
    legs_right : list[int]
        Axes that form the columns of the matrix.

    Returns
    -------
    Q : torch.Tensor
        Isometric tensor; shape is ``left_shape + [bond_dim]``.
    R : torch.Tensor
        Upper triangular factor; shape is ``[bond_dim] + right_shape``.
    """
    left_shape = [tensor.shape[i] for i in legs_left]
    right_shape = [tensor.shape[i] for i in legs_right]
    all_legs = legs_left + legs_right
    perm = all_legs
    mat = tensor.permute(perm).reshape(prod(left_shape), prod(right_shape))
    Q_mat, R_mat = torch.linalg.qr(mat)
    bond_dim = Q_mat.shape[1]
    Q = Q_mat.reshape(left_shape + [bond_dim])
    R = R_mat.reshape([bond_dim] + right_shape)
    return Q, R


def svd_tensor(
    tensor: torch.Tensor,
    legs_left: list[int],
    legs_right: list[int],
    max_rank: int | None = None,
    atol: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """SVD of a tensor with optional rank truncation.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.
    legs_left : list[int]
        Axes that form the rows of the matrix.
    legs_right : list[int]
        Axes that form the columns of the matrix.
    max_rank : int, optional
        Maximum bond dimension.
    atol : float
        Absolute tolerance for singular-value truncation.

    Returns
    -------
    U : torch.Tensor
        Shape ``left_shape + [rank]``.
    S : torch.Tensor
        Shape ``[rank]``.
    Vh : torch.Tensor
        Shape ``[rank] + right_shape``.
    trunc_err : float
        Truncation error (Frobenius norm of discarded singular values).
    """
    left_shape = [tensor.shape[i] for i in legs_left]
    right_shape = [tensor.shape[i] for i in legs_right]
    all_legs = legs_left + legs_right
    mat = tensor.permute(all_legs).reshape(prod(left_shape), prod(right_shape))
    U_mat, S, Vh_mat, trunc_err = svd_truncate(mat, max_rank=max_rank, atol=atol)
    rank = S.shape[0]
    U = U_mat.reshape(left_shape + [rank])
    Vh = Vh_mat.reshape([rank] + right_shape)
    return U, S, Vh, trunc_err


def enforce_semiunitarity(tensor: torch.Tensor, bond_axis: int) -> torch.Tensor:
    """Project a tensor onto the semi-unitary manifold via QR.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor with ``bond_axis`` as the "outgoing" bond.
    bond_axis : int
        The axis along which to orthogonalize.

    Returns
    -------
    torch.Tensor
        Isometric tensor (Q factor), same shape as input.
    """
    shape = list(tensor.shape)
    dim = shape[bond_axis]
    left_shape = shape[:bond_axis] + shape[bond_axis + 1:]
    mat = tensor.moveaxis(bond_axis, -1).reshape(-1, dim)
    Q, _ = torch.linalg.qr(mat)
    new_dim = Q.shape[1]
    shape_new = left_shape + [new_dim]
    return Q.reshape(shape_new).moveaxis(-1, bond_axis)


def move_gauge_qr(
    model: Model,
    frame: Frame,
    src: Node,
    dst: Node,
) -> None:
    """Move the orthogonality center from ``src`` to ``dst`` using QR.

    Modifies ``model`` in-place.  At each step along the path from ``src``
    to ``dst``, the current node is QR-decomposed, the isometric ``Q`` stays
    at the current node, and the upper-triangular ``R`` is absorbed into the
    next node.

    Parameters
    ----------
    model : Model
        TTN state (modified in-place).
    frame : Frame
        Tree topology.
    src : Node
        Current gauge center.
    dst : Node
        Target gauge center.

    Notes
    -----
    After this call, ``model.gauge_center`` is updated to ``dst``.
    """
    if src is dst:
        return

    node_path = frame.path(src, dst)
    for i in range(len(node_path) - 1):
        current = node_path[i]
        nxt = node_path[i + 1]
        ax_current, ax_next = frame.axes(current, nxt)

        tensor = model[current]
        shape = list(tensor.shape)

        # Build left legs = all axes EXCEPT the bond toward dst
        left_legs = [j for j in range(tensor.ndim) if j != ax_current]
        right_legs = [ax_current]

        Q, R = qr_decompose(tensor, left_legs, right_legs)

        # Restore original axis order: Q's last axis goes back to ax_current
        n_left = len(left_legs)
        # Q shape: left_shape + [bond_dim]; put bond_dim back at ax_current
        Q_reordered = Q.moveaxis(-1, ax_current)
        model[current] = Q_reordered

        # Absorb R into next node along ax_next
        # R shape: [new_bond, old_bond] — contract old_bond with next_tensor's ax_next
        next_tensor = model[nxt]
        new_next = torch.tensordot(R, next_tensor, dims=([1], [ax_next]))
        # new_next shape: (new_bond, ...) with ax_next dim removed, then appended at front
        new_next = new_next.movedim(0, ax_next)
        model[nxt] = new_next

    model.gauge_center = dst
