# coding: utf-8
"""Per-site Liouville-space superoperators for arbitrary local dimension.

For a site with local Hilbert dimension ``d``, the local Liouville space has
dimension ``d²``.  The interleaved convention encodes a single-site density
matrix element ``rho_{ij}`` at flat index ``i * d + j``.

Three superoperator building blocks act on this d²-dimensional space:

    left_local(A)   : rho ↦ A · rho          ↔  A ⊗ I_d         (d²×d²)
    right_local(B)  : rho ↦ rho · B          ↔  I_d ⊗ B^T        (d²×d²)
    jump_local(A,B) : rho ↦ A · rho · B†     ↔  A ⊗ B*           (d²×d²)

These reduce to the existing spin-1/2 (d=2, d²=4) formulas when ``d=2``.
"""

from __future__ import annotations

import numpy as np


def left_local(A: np.ndarray) -> np.ndarray:
    """Local left-multiplication superoperator: ρ ↦ A·ρ.

    Parameters
    ----------
    A : (d, d) complex ndarray

    Returns
    -------
    (d², d²) complex ndarray  equal to ``A ⊗ I_d``.
    """
    d = A.shape[0]
    return np.kron(A, np.eye(d, dtype=complex))


def right_local(B: np.ndarray) -> np.ndarray:
    """Local right-multiplication superoperator: ρ ↦ ρ·B."""
    d = B.shape[0]
    return np.kron(np.eye(d, dtype=complex), B.T)


def jump_local(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Local two-sided multiplication: ρ ↦ A·ρ·B† (yields A ⊗ B*)."""
    return np.kron(A, B.conj())


def identity_super(d: int) -> np.ndarray:
    """Identity superoperator on the local Liouville space (d², d²)."""
    return np.eye(d * d, dtype=complex)


# ---------------------------------------------------------------------------
# Vectorisation helpers (interleaved convention)
# ---------------------------------------------------------------------------

def vec_local(rho: np.ndarray) -> np.ndarray:
    """Vectorise a single-site density matrix in row-major order.

    For ``rho`` of shape ``(d, d)``, returns a length-``d²`` vector with
    ``v[i*d + j] = rho[i, j]``.
    """
    return rho.flatten()


def unvec_local(v: np.ndarray, d: int) -> np.ndarray:
    """Inverse of :func:`vec_local`."""
    return v.reshape(d, d)


def permute_rowmajor_to_interleaved(
    rho_vec: np.ndarray,
    site_dims: list[int],
) -> np.ndarray:
    """Permute a row-major vec(rho) of size ``D²`` into interleaved per-site ordering.

    Given a full density matrix ρ of dimension ``D = ∏ d_s``, the row-major
    vectorisation has index ordering ``(i_0, i_1, ..., i_{N-1}, j_0, j_1, ...,
    j_{N-1})``.  The interleaved per-site ordering is ``(i_0, j_0, i_1, j_1, ...,
    i_{N-1}, j_{N-1})`` — each site ``s`` then carries a single local index of
    range ``d_s²``.

    Parameters
    ----------
    rho_vec : (D²,) complex ndarray
        Row-major vec(ρ).
    site_dims : list[int]
        Per-site dimensions in order.

    Returns
    -------
    (D²,) complex ndarray
        Interleaved permutation.
    """
    N = len(site_dims)
    if N == 1:
        return rho_vec
    shape = list(site_dims) + list(site_dims)  # 2N dimensions
    # perm: position s in output is (ket-of-site-s, bra-of-site-s)
    perm: list[int] = []
    for s in range(N):
        perm.extend([s, N + s])
    return rho_vec.reshape(shape).transpose(perm).reshape(-1)


def permute_interleaved_to_rowmajor(
    rho_vec: np.ndarray,
    site_dims: list[int],
) -> np.ndarray:
    """Inverse of :func:`permute_rowmajor_to_interleaved`."""
    N = len(site_dims)
    if N == 1:
        return rho_vec
    # Interleaved shape: (d_0, d_0, d_1, d_1, ..., d_{N-1}, d_{N-1})
    shape_inter = []
    for d in site_dims:
        shape_inter.extend([d, d])
    # Pull out kets (even positions: 0, 2, ..., 2N-2) then bras (odd: 1, 3, ..., 2N-1)
    perm_inv = list(range(0, 2 * N, 2)) + list(range(1, 2 * N, 2))
    return rho_vec.reshape(shape_inter).transpose(perm_inv).reshape(-1)
