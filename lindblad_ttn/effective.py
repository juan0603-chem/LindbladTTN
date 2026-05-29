# coding: utf-8
"""Effective-Hamiltonian tools (M5).

Three building blocks:

* :func:`schrieffer_wolff` — numerical SW transformation up to a given order,
  block-diagonalising a Hamiltonian ``H₀ + V`` by eliminating ``V`` between
  energy-distinct subspaces.
* :func:`dispersive_shift` — numerical χ extraction from a full
  Hamiltonian by exact diagonalisation.
* :func:`magnus_average` — time-averaged effective Hamiltonian for a
  periodic drive via the Magnus expansion (orders 1 and 2 implemented).

All routines operate on dense numpy matrices — they are intended for
small-to-medium Hamiltonians (≤ 1000 levels) used to PARAMETERISE the
full TTN simulation, not for the TTN dynamics themselves.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.linalg import eigh, expm


# ---------------------------------------------------------------------------
# Schrieffer–Wolff transformation
# ---------------------------------------------------------------------------

def schrieffer_wolff(
    H0: np.ndarray,
    V: np.ndarray,
    projector_A: np.ndarray | None = None,
    order: int = 2,
) -> np.ndarray:
    """Schrieffer–Wolff transformation of ``H = H0 + V`` to a chosen order.

    The transformation block-diagonalises ``H`` between the two subspaces
    defined by ``projector_A`` (subspace A) and ``I − projector_A`` (subspace B),
    eliminating the off-diagonal coupling perturbatively in ``V``.

    Parameters
    ----------
    H0 : ndarray
        Unperturbed Hamiltonian (Hermitian).  Must be diagonal in the chosen
        basis OR ``projector_A`` must commute with ``H0``.
    V : ndarray
        Perturbation (Hermitian).
    projector_A : ndarray, optional
        Projector onto subspace A.  If omitted, uses the lower half of the
        ``H0``-eigenstate spectrum as A.
    order : int, default 2
        Maximum order of the SW expansion (2 or 4).

    Returns
    -------
    H_eff : ndarray
        Effective Hamiltonian on the full Hilbert space, block-diagonal up to
        the requested order.  Restrict to subspace A by sandwiching with
        ``projector_A``.

    Notes
    -----
    The generator ``S`` is anti-Hermitian and satisfies ``[H0, S] = V_od``
    where ``V_od`` is the off-diagonal block of ``V``.  The transformed
    Hamiltonian is::

        H_eff = e^{-S} H e^{S}
              = H0 + V_d + (1/2)[V_od, S] + (1/24)[[[V_od, S], S], S] + ...

    where ``V_d`` is the block-diagonal part of ``V``.
    """
    H0 = np.asarray(H0, dtype=complex)
    V = np.asarray(V, dtype=complex)
    d = H0.shape[0]
    if projector_A is None:
        # Default: lower half of H0 spectrum is subspace A
        eigs, vecs = eigh(H0)
        idx = np.argsort(eigs)
        kA = d // 2
        PA = vecs[:, idx[:kA]] @ vecs[:, idx[:kA]].conj().T
    else:
        PA = np.asarray(projector_A, dtype=complex)
    PB = np.eye(d, dtype=complex) - PA

    # Off-diagonal and diagonal parts of V
    V_od = PA @ V @ PB + PB @ V @ PA
    V_d = PA @ V @ PA + PB @ V @ PB

    # Solve [H0, S^(1)] = V_od for S^(1) (in the eigenbasis of H0)
    eigs, U = eigh(H0)
    V_od_eig = U.conj().T @ V_od @ U
    S1_eig = np.zeros_like(V_od_eig)
    for i in range(d):
        for j in range(d):
            de = eigs[i] - eigs[j]
            if abs(de) > 1e-12:
                S1_eig[i, j] = V_od_eig[i, j] / de
    S1 = U @ S1_eig @ U.conj().T

    H_eff = H0 + V_d + 0.5 * (V_od @ S1 - S1 @ V_od)
    if order >= 4:
        # 4th-order correction (Bravyi-DiVincenzo-Loss 2011, Eq. 3.7)
        comm = _commutator
        triple = comm(comm(comm(V_od, S1), S1), S1)
        H_eff = H_eff + triple / 24.0
    return H_eff


def dispersive_shift(
    H_qc: np.ndarray,
    qubit_dim: int,
    cavity_dim: int,
    n_max: int = 2,
) -> float:
    """Extract the dispersive shift χ from a full qubit+cavity Hamiltonian.

    χ is defined via the eigenenergies of ``H_qc`` as::

        χ ≈ ½ [(E_{|e, n=1>} − E_{|e, n=0>}) − (E_{|g, n=1>} − E_{|g, n=0>})]

    where ``|g>``, ``|e>`` are the qubit ground/excited states and ``|n>``
    are the cavity Fock states.

    Parameters
    ----------
    H_qc : ndarray, shape (qubit_dim*cavity_dim, qubit_dim*cavity_dim)
        Full Hamiltonian in basis ``|i_q> ⊗ |n_c>``.  Index ordering:
        ``index = i_q * cavity_dim + n_c``.
    qubit_dim : int
    cavity_dim : int
    n_max : int
        Maximum photon number to track for the χ extraction.

    Returns
    -------
    chi : float
        Dispersive shift (same units as H_qc).
    """
    eigs, vecs = eigh(H_qc)
    # Build σ_z ⊗ I as an observable to label each eigenstate by qubit state.
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    sz_full = np.kron(sz, np.eye(cavity_dim))
    n_op = np.kron(np.eye(qubit_dim), np.diag(np.arange(cavity_dim)).astype(complex))
    sz_vals = np.array([
        float(np.real(vecs[:, k].conj() @ sz_full @ vecs[:, k]))
        for k in range(len(eigs))
    ])
    n_vals = np.array([
        float(np.real(vecs[:, k].conj() @ n_op @ vecs[:, k]))
        for k in range(len(eigs))
    ])
    # |g> eigenstates have ⟨σz⟩ ≈ −1, |e> have ⟨σz⟩ ≈ +1.
    g_mask = sz_vals < 0
    e_mask = sz_vals > 0
    E_g = sorted(zip(n_vals[g_mask], eigs[g_mask]))
    E_e = sorted(zip(n_vals[e_mask], eigs[e_mask]))
    n_max = min(n_max, len(E_g) - 1, len(E_e) - 1)
    omega_c_g = E_g[n_max][1] - E_g[0][1]
    omega_c_e = E_e[n_max][1] - E_e[0][1]
    return 0.5 * (omega_c_e - omega_c_g) / max(n_max, 1)


# ---------------------------------------------------------------------------
# Magnus expansion
# ---------------------------------------------------------------------------

def magnus_average(
    H_t: Callable[[float], np.ndarray],
    period: float,
    n_points: int = 64,
    order: int = 1,
) -> np.ndarray:
    """Time-averaged Hamiltonian over one period via the Magnus expansion.

    Magnus writes ``U(T) = exp[Σ_k Ω_k(T)]`` with::

        Ω₁(T) = −i ∫₀ᵀ H(t) dt
        Ω₂(T) = −½ ∫₀ᵀ ∫₀^{t₁} [H(t₁), H(t₂)] dt₂ dt₁

    ``H_avg = i · (Ω₁ + Ω₂ + ...) / T`` is the effective (Floquet) Hamiltonian.

    Parameters
    ----------
    H_t : callable
        ``H(t) → np.ndarray`` returning the Hamiltonian at time ``t``.
    period : float
        Period over which to average.
    n_points : int
        Number of trapezoidal-rule samples.
    order : int, default 1
        Magnus order to keep (1 or 2).

    Returns
    -------
    H_eff : ndarray
        Effective time-independent Hamiltonian.
    """
    times = np.linspace(0.0, period, n_points)
    dt = times[1] - times[0]
    H_samples = [H_t(t) for t in times]

    # Order 1: average H(t)
    H1 = np.zeros_like(H_samples[0], dtype=complex)
    for k, H in enumerate(H_samples):
        w = 0.5 if k in (0, n_points - 1) else 1.0
        H1 = H1 + w * H
    H1 = H1 * dt / period

    if order < 2:
        return H1

    # Order 2: Magnus correction
    Omega2 = np.zeros_like(H1, dtype=complex)
    for k1, H1_t in enumerate(H_samples):
        inner = np.zeros_like(H1, dtype=complex)
        for k2 in range(k1 + 1):
            w2 = 0.5 if k2 in (0, k1) else 1.0
            inner = inner + w2 * H_samples[k2]
        inner = inner * dt
        comm = _commutator(H1_t, inner)
        w1 = 0.5 if k1 in (0, n_points - 1) else 1.0
        Omega2 = Omega2 + w1 * comm
    Omega2 = -0.5j * Omega2 * dt
    # H_eff = (Ω₁ + Ω₂) · i / T, but Ω₁ = -i∫H ⇒ -i Ω₁ / T = H1 already.
    return H1 + 1j * Omega2 / period


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A
