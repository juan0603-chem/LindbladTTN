# coding: utf-8
"""Builds the Liouvillian as a Sum-of-Products operator.

Strategy
--------
1.  Pauli-decompose every operator (H0, each L_k, V) in the Hilbert space:
    ``M = Σ_α c_α P_α^(0) ⊗ … ⊗ P_α^(N-1)`` where P_α^(s) ∈ {I,X,Y,Z}.
2.  For each Pauli string, map to local 4×4 superoperators acting on the
    *interleaved* Liouville space of each qubit:
      left_local(P)  = P ⊗ I₂  — left multiplication on (ket_s, bra_s) site
      right_local(P) = I₂ ⊗ Pᵀ — right multiplication
      jump_local(P,Q)= P ⊗ Q*  — for L ρ L† at local site
3.  Each Pauli string becomes one or a few SoP terms with one 4×4 matrix per DOF.

DOF names: ``'q0'``, ``'q1'``, ..., ``'q{N-1}'``.
Local Liouville dimension per site: 4 (interleaved ket-bra convention).

IMPORTANT: The SoP operators here are in the *interleaved* Liouville convention
where site s has index (i_s, j_s) — ket bit and bra bit interleaved.  The
solver must encode the state vector using the same interleaved permutation.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
import torch

from lindblad_ttn.core.backend import DEVICE, DTYPE, to_torch
from lindblad_ttn.core.sop import SumOfProducts


# ---------------------------------------------------------------------------
# Pauli matrices (2×2, numpy)
# ---------------------------------------------------------------------------

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_NP: list[np.ndarray] = [_I2, _X, _Y, _Z]  # 0=I,1=X,2=Y,3=Z


# ---------------------------------------------------------------------------
# Local 4×4 superoperators in interleaved (ket_s, bra_s) convention
# ---------------------------------------------------------------------------

def _left_local(P: np.ndarray) -> np.ndarray:
    """P ⊗ I₂ — left multiplication by P on the local ket index."""
    return np.kron(P, _I2)


def _right_local(P: np.ndarray) -> np.ndarray:
    """I₂ ⊗ Pᵀ — right multiplication by P on the local bra index."""
    return np.kron(_I2, P.T)


def _jump_local(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """P ⊗ Q* — for L ρ L† with L~P (ket) and L†~Q† (bra)."""
    return np.kron(P, Q.conj())


# ---------------------------------------------------------------------------
# Pauli decomposition (Hilbert space)
# ---------------------------------------------------------------------------

def pauli_decompose(
    M: np.ndarray,
    n_sites: int,
    threshold: float = 1e-14,
) -> list[tuple[complex, list[int]]]:
    """Decompose an N-qubit operator into Pauli strings.

    ``M = Σ_α c_α P_α`` where ``c_α = Tr(P_α M) / 2^N``.

    Parameters
    ----------
    M : np.ndarray  shape (2^N, 2^N)
    n_sites : int
    threshold : float

    Returns
    -------
    list of (coeff, pauli_indices)
        ``pauli_indices`` is a list of length N with values in {0,1,2,3}.
    """
    d = 2 ** n_sites
    assert M.shape == (d, d), f"Expected ({d},{d}), got {M.shape}."
    norm = 1.0 / d

    result = []
    for indices in itertools.product(range(4), repeat=n_sites):
        P = PAULI_NP[indices[0]]
        for idx in indices[1:]:
            P = np.kron(P, PAULI_NP[idx])
        coeff = complex(np.trace(P @ M) * norm)
        if abs(coeff) > threshold:
            result.append((coeff, list(indices)))
    return result


# ---------------------------------------------------------------------------
# Main SoP builder
# ---------------------------------------------------------------------------

def build_lindblad_sop(
    n_sites: int,
    H: np.ndarray | None,
    L_ops: list[tuple[float, np.ndarray]],
    threshold: float = 1e-14,
) -> SumOfProducts:
    """Build the Lindblad Liouvillian as a SoP in the interleaved convention.

    Parameters
    ----------
    n_sites : int
    H : ndarray or None
        Hamiltonian, shape (2^N, 2^N).
    L_ops : list of (gamma, L)
    threshold : float

    Returns
    -------
    SumOfProducts
        Each term has one 4×4 matrix per DOF 'q0'..'q{N-1}'.
    """
    sop = SumOfProducts()

    # ------------------------------------------------------------------
    # Unitary part: -i [H, ρ] = -i H ρ + i ρ H
    # ------------------------------------------------------------------
    if H is not None:
        H_np = np.asarray(H, dtype=complex)
        for c_alpha, indices in pauli_decompose(H_np, n_sites, threshold):
            # -i c_α P_α ρ  (left multiplication at each site)
            sop.add_term(
                -1j * c_alpha,
                {f"q{s}": to_torch(_left_local(PAULI_NP[k]))
                 for s, k in enumerate(indices)},
            )
            # +i c_α ρ P_α  (right multiplication at each site)
            sop.add_term(
                1j * c_alpha,
                {f"q{s}": to_torch(_right_local(PAULI_NP[k]))
                 for s, k in enumerate(indices)},
            )

    # ------------------------------------------------------------------
    # Dissipator: γ (L ρ L† − ½ L†L ρ − ½ ρ L†L)
    # ------------------------------------------------------------------
    for gamma, L in L_ops:
        L_np = np.asarray(L, dtype=complex)
        L_terms = pauli_decompose(L_np, n_sites, threshold)

        LdL_np = L_np.conj().T @ L_np
        LdL_terms = pauli_decompose(LdL_np, n_sites, threshold)

        # Jump term: γ Σ_{α,β} d_α d_β* P_α ρ P_β
        for d_alpha, alpha in L_terms:
            for d_beta, beta in L_terms:
                coeff = gamma * d_alpha * np.conj(d_beta)
                if abs(coeff) < threshold:
                    continue
                sop.add_term(
                    coeff,
                    {f"q{s}": to_torch(_jump_local(PAULI_NP[alpha[s]], PAULI_NP[beta[s]]))
                     for s in range(n_sites)},
                )

        # Anti-commutator: −γ/2 (L†L ρ + ρ L†L)
        for e_gamma, gamma_indices in LdL_terms:
            coeff_lr = -gamma / 2.0 * e_gamma
            if abs(coeff_lr) < threshold:
                continue
            # Left: L†L ρ
            sop.add_term(
                coeff_lr,
                {f"q{s}": to_torch(_left_local(PAULI_NP[k]))
                 for s, k in enumerate(gamma_indices)},
            )
            # Right: ρ L†L
            sop.add_term(
                coeff_lr,
                {f"q{s}": to_torch(_right_local(PAULI_NP[k]))
                 for s, k in enumerate(gamma_indices)},
            )

    return sop


# ---------------------------------------------------------------------------
# LiouvillianSoP
# ---------------------------------------------------------------------------

class LiouvillianSoP:
    """Builds the Lindblad Liouvillian as a Sum-of-Products.

    Supports an arbitrary number of independent time-dependent drives,
    matching the pytenso ``f_list`` pattern: the propagator evaluates
    ``H₀ + Σᵢ fᵢ(t)·Vᵢ`` at every timestep rather than chaining solvers.

    Parameters
    ----------
    n_sites : int
    H0 : ndarray or None
    L_ops : list of (gamma, L)
    V : ndarray or None
        Single drive operator for ``H(t) = H₀ + f(t)·V``. Kept for backward
        compatibility — internally promoted to ``Vs=[V]``.
    Vs : list of ndarray or None
        Multiple drive operators, one per time-dependent channel. Mutually
        exclusive with ``V``.
    threshold : float
    """

    def __init__(
        self,
        n_sites: int,
        H0: np.ndarray | None,
        L_ops: list[tuple[float, np.ndarray]],
        V: np.ndarray | None = None,
        Vs: list[np.ndarray] | None = None,
        threshold: float = 1e-14,
    ) -> None:
        if V is not None and Vs is not None:
            raise ValueError("Pass either V or Vs, not both.")

        self.n_sites = n_sites
        self._sop_H0 = build_lindblad_sop(n_sites, H0, L_ops, threshold)

        # Normalize to a list internally; preserve backward-compat scalar V.
        if V is not None:
            Vs = [V]
        self._sop_Vs: list[SumOfProducts] = (
            [build_lindblad_sop(n_sites, V_i, [], threshold) for V_i in Vs]
            if Vs is not None else []
        )

    # ------------------------------------------------------------------
    # Backward-compatible scalar accessor
    # ------------------------------------------------------------------
    @property
    def _sop_V(self) -> SumOfProducts | None:
        """Legacy single-drive accessor: returns the first drive SoP or None.

        Existing call sites read ``liouv._sop_V`` expecting at most one
        drive; we keep that working for ``len(_sop_Vs) <= 1``.
        """
        if not self._sop_Vs:
            return None
        if len(self._sop_Vs) > 1:
            raise AttributeError(
                "_sop_V is undefined when multiple drives are present; "
                "use _sop_Vs (list) instead."
            )
        return self._sop_Vs[0]

    def build_sop_H0(self) -> SumOfProducts:
        return self._sop_H0

    def build_sop_V(self) -> SumOfProducts:
        if not self._sop_Vs:
            raise ValueError("No driving operator V was provided.")
        if len(self._sop_Vs) > 1:
            raise ValueError(
                "Multiple drives present; iterate over .build_sop_Vs() instead."
            )
        return self._sop_Vs[0]

    def build_sop_Vs(self) -> list[SumOfProducts]:
        """Return all drive SoPs in registration order."""
        return list(self._sop_Vs)

    def combine(self, f_val: float) -> SumOfProducts:
        """Return SoP for ``H₀ + f_val·V₀`` (single-drive helper, legacy)."""
        if not self._sop_Vs:
            return self._sop_H0
        if len(self._sop_Vs) > 1:
            raise ValueError(
                "combine() is single-drive only; use TimeDependentSoP for "
                "multi-drive evaluation."
            )
        return self._sop_H0 + (f_val * self._sop_Vs[0])

    def __repr__(self) -> str:
        n_v_terms = [sop.n_terms for sop in self._sop_Vs]
        return (
            f"LiouvillianSoP(n_sites={self.n_sites}, "
            f"n_terms_H0={self._sop_H0.n_terms}, "
            f"n_drives={len(self._sop_Vs)}, "
            f"n_terms_V={n_v_terms})"
        )
