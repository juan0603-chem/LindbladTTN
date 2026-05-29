# coding: utf-8
"""Steady-state and low-lying eigenstate solvers (M7).

Two routines:

* :func:`steady_state_dense` — exact dense eigenvalue of the Liouvillian L,
  returning the unique eigenvector with eigenvalue 0 (or smallest |λ|),
  normalised to ``Tr(ρ) = 1``.
* :func:`energy_levels_dense` — exact lowest-``k`` eigenstates of a
  Hamiltonian via dense ``scipy.linalg.eigh``.

These are dense routines intended for parameter sweeps and template
calibration.  TTN-based DMRG-style energy solvers are out of scope for
this milestone and are flagged as a future extension.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import eig, eigh

from lindblad_ttn.core.sop import SumOfProducts
from lindblad_ttn.physics.liouvillian_nd import build_lindblad_sop_nd
from lindblad_ttn.physics.liouville_nd import (
    permute_interleaved_to_rowmajor,
)


# ---------------------------------------------------------------------------
# Steady state from H_terms + L_terms (heterogeneous API)
# ---------------------------------------------------------------------------

def steady_state_dense(
    site_dims: list[int],
    dof_names: list[str],
    H_terms: list[tuple[complex, dict]],
    L_terms: list[tuple[float, dict]],
    eigenvalue_tol: float = 1e-9,
) -> np.ndarray:
    """Steady state of the Lindblad master equation by dense eigensolve.

    Build the full Liouvillian L as a ``(D², D²)`` matrix where ``D = ∏ d_s``,
    diagonalise, and return the eigenvector with eigenvalue closest to 0.
    Normalised so ``Tr(ρ) = 1``.

    Parameters
    ----------
    site_dims : list[int]
    dof_names : list[str]
    H_terms, L_terms : as accepted by :class:`LindbladTTN`.
    eigenvalue_tol : float
        Warning threshold for the smallest |λ|: a healthy steady state has
        ``|λ| < tol``.

    Returns
    -------
    rho_ss : ndarray
        Steady-state density matrix of shape ``(D, D)``.
    """
    D = int(np.prod(site_dims))
    sop = build_lindblad_sop_nd(H_terms, L_terms)
    L_inter = _sop_to_dense(sop, site_dims, dof_names)
    # Permute from interleaved back to row-major to use standard vec(ρ) form.
    L_row = _interleaved_to_rowmajor_matrix(L_inter, site_dims)

    eigs, vecs = eig(L_row)
    idx = int(np.argmin(np.abs(eigs)))
    lam = eigs[idx]
    if abs(lam) > eigenvalue_tol:
        import warnings
        warnings.warn(
            f"Steady-state eigenvalue {lam} > tol {eigenvalue_tol}; the system "
            "may not have a unique stationary state.",
            RuntimeWarning,
            stacklevel=2,
        )
    rho_vec = vecs[:, idx]
    rho = rho_vec.reshape(D, D)
    # Hermitise and normalise.
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) < 1e-12:
        raise RuntimeError("Steady-state has zero trace — pick another eigenvector.")
    rho = rho / tr
    return rho


def energy_levels_dense(
    H: np.ndarray, k: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest ``k`` eigenvalues and eigenvectors of a Hermitian ``H``."""
    eigs, vecs = eigh(H)
    return eigs[:k], vecs[:, :k]


# ---------------------------------------------------------------------------
# Helpers (SoP → dense Liouvillian)
# ---------------------------------------------------------------------------

def _sop_to_dense(
    sop: SumOfProducts,
    site_dims: list[int],
    dof_names: list[str],
) -> np.ndarray:
    """Materialise the SoP as a dense ``(D², D²)`` Liouvillian in interleaved
    per-site ordering."""
    liouv_dims = [d * d for d in site_dims]
    D2 = int(np.prod(liouv_dims))
    L_dense = np.zeros((D2, D2), dtype=complex)
    eye_by_dof = {
        name: np.eye(d2, dtype=complex)
        for name, d2 in zip(dof_names, liouv_dims)
    }
    for coeff, op_dict in sop.terms:
        mats = []
        for name in dof_names:
            M = op_dict.get(name)
            if M is None:
                mats.append(eye_by_dof[name])
            else:
                mats.append(M.cpu().numpy() if hasattr(M, "cpu") else np.asarray(M))
        prod_mat = mats[0]
        for M in mats[1:]:
            prod_mat = np.kron(prod_mat, M)
        L_dense = L_dense + complex(coeff) * prod_mat
    return L_dense


def _interleaved_to_rowmajor_matrix(
    L_inter: np.ndarray, site_dims: list[int],
) -> np.ndarray:
    """Permute a Liouvillian matrix from interleaved to row-major ordering.

    Uses ``P L_inter P^T`` where P is the permutation matrix that maps
    row-major vec(ρ) → interleaved vec(ρ).
    """
    D2 = L_inter.shape[0]
    # Build the permutation matrix by applying the permutation to basis vectors.
    P = np.zeros((D2, D2), dtype=complex)
    for i in range(D2):
        e = np.zeros(D2, dtype=complex)
        e[i] = 1.0
        P[:, i] = permute_interleaved_to_rowmajor(e, site_dims)
    return P @ L_inter @ P.conj().T
