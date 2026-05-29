# coding: utf-8
"""Liouville-space vectorization and superoperator maps.

Vectorization convention (row-major / "column stack"):
    vec(ρ)_{i·d + j} = ρ_{ij}

So a (d×d) density matrix is mapped to a length-d² vector by flattening
in row-major order.  The inverse is ``unvec(v, d) = v.reshape(d, d)``.

Superoperator conventions
--------------------------
left_sop(A)     : ρ ↦ A·ρ        ↔  A ⊗ I     (d²×d² matrix)
right_sop(B)    : ρ ↦ ρ·B        ↔  I ⊗ Bᵀ
both_sop(A, B)  : ρ ↦ A·ρ·B†     ↔  A ⊗ B*

Lindblad dissipator for jump operator L with rate γ:
    D[L]ρ = γ (L·ρ·L† − ½ L†L·ρ − ½ ρ·L†L)
          ↔  γ (L⊗L* − ½ L†L⊗I − ½ I⊗(L†L)*)

Unitary part (von Neumann):
    −i[H, ρ] = −i(H·ρ − ρ·H)
             ↔  −i(H⊗I − I⊗Hᵀ)
"""

from __future__ import annotations

import torch

from lindblad_ttn.core.backend import DEVICE, DTYPE, eye, kron_torch


# ---------------------------------------------------------------------------
# vec / unvec
# ---------------------------------------------------------------------------

def vec(rho: torch.Tensor) -> torch.Tensor:
    """Vectorize a density matrix in row-major order.

    Parameters
    ----------
    rho : torch.Tensor
        Shape ``(d, d)``.

    Returns
    -------
    torch.Tensor
        Shape ``(d²,)``.
    """
    return rho.flatten()


def unvec(v: torch.Tensor, d: int) -> torch.Tensor:
    """Unvectorize a Liouville-space vector back to a density matrix.

    Parameters
    ----------
    v : torch.Tensor
        Shape ``(d²,)``.
    d : int
        Physical Hilbert-space dimension.

    Returns
    -------
    torch.Tensor
        Shape ``(d, d)``.
    """
    return v.reshape(d, d)


# ---------------------------------------------------------------------------
# Superoperator building blocks
# ---------------------------------------------------------------------------

def left_sop(A: torch.Tensor) -> torch.Tensor:
    """Left-multiplication superoperator: ρ ↦ A·ρ.

    Parameters
    ----------
    A : torch.Tensor
        Shape ``(d, d)``.

    Returns
    -------
    torch.Tensor
        Shape ``(d², d²)`` — equals ``A ⊗ I``.
    """
    d = A.shape[0]
    return kron_torch(A, eye(d))


def right_sop(B: torch.Tensor) -> torch.Tensor:
    """Right-multiplication superoperator: ρ ↦ ρ·B.

    Parameters
    ----------
    B : torch.Tensor
        Shape ``(d, d)``.

    Returns
    -------
    torch.Tensor
        Shape ``(d², d²)`` — equals ``I ⊗ Bᵀ``.
    """
    d = B.shape[0]
    return kron_torch(eye(d), B.T)


def both_sop(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Two-sided multiplication superoperator: ρ ↦ A·ρ·B†.

    Parameters
    ----------
    A : torch.Tensor
        Shape ``(d, d)``.
    B : torch.Tensor
        Shape ``(d, d)``.

    Returns
    -------
    torch.Tensor
        Shape ``(d², d²)`` — equals ``A ⊗ B*``.
    """
    return kron_torch(A, B.conj())


def dissipator_sop(L: torch.Tensor, gamma: float) -> torch.Tensor:
    """Lindblad dissipator superoperator.

    Returns the matrix representation of:
        D[L] = γ (L·ρ·L† − ½ L†L·ρ − ½ ρ·L†L)

    Parameters
    ----------
    L : torch.Tensor
        Jump operator, shape ``(d, d)``.
    gamma : float
        Decay rate (must be ≥ 0).

    Returns
    -------
    torch.Tensor
        Shape ``(d², d²)``.
    """
    d = L.shape[0]
    I = eye(d)
    LdL = L.conj().T @ L
    jump = both_sop(L, L)                     # L ⊗ L*
    left_decay = left_sop(LdL)               # L†L ⊗ I
    right_decay = right_sop(LdL)             # I ⊗ (L†L)ᵀ
    return gamma * (jump - 0.5 * left_decay - 0.5 * right_decay)


def unitary_sop(H: torch.Tensor) -> torch.Tensor:
    """Unitary (von Neumann) superoperator: −i[H, ρ].

    Returns the matrix representation of:
        −i(H·ρ − ρ·H) = −i(H⊗I − I⊗Hᵀ)

    Parameters
    ----------
    H : torch.Tensor
        Hamiltonian, shape ``(d, d)``.  Should be Hermitian.

    Returns
    -------
    torch.Tensor
        Shape ``(d², d²)``.
    """
    d = H.shape[0]
    I = eye(d)
    return -1j * (kron_torch(H, I) - kron_torch(I, H.T))


# ---------------------------------------------------------------------------
# Observables from vectorized state
# ---------------------------------------------------------------------------

def trace_from_vec(v: torch.Tensor, d: int) -> complex:
    """Compute Tr(ρ) from the vectorized state.

    Parameters
    ----------
    v : torch.Tensor
        Vectorized density matrix, shape ``(d²,)``.
    d : int
        Physical Hilbert-space dimension.

    Returns
    -------
    complex
        The trace, which should be 1 for a normalized state.
    """
    rho = unvec(v, d)
    return rho.diagonal().sum().item()


def expect_from_vec(v: torch.Tensor, O: torch.Tensor, d: int) -> complex:
    """Compute ⟨O⟩ = Tr(O·ρ) from the vectorized state.

    Parameters
    ----------
    v : torch.Tensor
        Vectorized density matrix, shape ``(d²,)``.
    O : torch.Tensor
        Observable, shape ``(d, d)``.
    d : int
        Physical Hilbert-space dimension.

    Returns
    -------
    complex
        Expectation value.
    """
    rho = unvec(v, d)
    return (O @ rho).diagonal().sum().item()


# ---------------------------------------------------------------------------
# Local-site Liouville operators (for the SoP decomposition)
# ---------------------------------------------------------------------------

def local_unitary_sop(H_local: torch.Tensor) -> torch.Tensor:
    """Unitary superoperator for a single-site operator.

    Equivalent to :func:`unitary_sop` but emphasised for single-site use
    in the Pauli decomposition.

    Parameters
    ----------
    H_local : torch.Tensor
        Single-site operator, shape ``(d, d)``.

    Returns
    -------
    torch.Tensor
        Shape ``(d², d²)``.
    """
    return unitary_sop(H_local)


def local_dissipator_sop(L_local: torch.Tensor, gamma: float) -> torch.Tensor:
    """Dissipator superoperator for a single-site jump operator.

    Parameters
    ----------
    L_local : torch.Tensor
        Single-site jump operator, shape ``(d, d)``.
    gamma : float
        Decay rate.

    Returns
    -------
    torch.Tensor
        Shape ``(d², d²)``.
    """
    return dissipator_sop(L_local, gamma)
