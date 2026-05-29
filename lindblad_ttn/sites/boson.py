# coding: utf-8
"""Bosonic site (M2) — Fock-truncated harmonic oscillator.

The Fock cutoff ``N_cut`` is the dimension of the local Hilbert space; the
ladder operators ``a``, ``a_dag`` are truncated accordingly.

Conventions
-----------
* Basis: ``|0>, |1>, ..., |N_cut - 1>`` with ``n|n> = n|n>``.
* ``a|n> = sqrt(n) |n-1>`` (with ``a|0> = 0``).
* ``a_dag|n> = sqrt(n+1) |n+1>`` (with ``a_dag|N_cut-1> = 0`` because of truncation).
* ``x = (a + a_dag)/sqrt(2)``, ``p = (a - a_dag)/(i sqrt(2))``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lindblad_ttn.sites.base import Site


def _boson_operators(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (a, a_dag, n, x, p) truncated to dimension ``N``."""
    a = np.zeros((N, N), dtype=complex)
    for n in range(1, N):
        a[n - 1, n] = np.sqrt(n)
    a_dag = a.conj().T
    n_op = a_dag @ a
    x = (a + a_dag) / np.sqrt(2)
    p = (a - a_dag) / (1j * np.sqrt(2))
    return a, a_dag, n_op, x, p


@dataclass
class BosonSite(Site):
    """A truncated harmonic-oscillator site.

    Parameters
    ----------
    N_cut : int
        Fock-space cutoff.  Local dimension equals ``N_cut``.
    name : str
        DOF name (e.g. ``'c0'``, ``'cavity'``).

    Attributes
    ----------
    a, adag : (N_cut, N_cut) complex arrays
        Ladder operators.
    n : (N_cut, N_cut) complex array
        Number operator.
    x, p : (N_cut, N_cut) complex arrays
        Quadrature operators in dimensionless units.
    """

    N_cut: int = 8
    dim: int = field(default=8, init=False)
    a: np.ndarray = field(default_factory=lambda: np.zeros((8, 8), dtype=complex), repr=False)
    adag: np.ndarray = field(default_factory=lambda: np.zeros((8, 8), dtype=complex), repr=False)
    n: np.ndarray = field(default_factory=lambda: np.zeros((8, 8), dtype=complex), repr=False)
    x: np.ndarray = field(default_factory=lambda: np.zeros((8, 8), dtype=complex), repr=False)
    p: np.ndarray = field(default_factory=lambda: np.zeros((8, 8), dtype=complex), repr=False)

    def __post_init__(self) -> None:
        if self.N_cut < 2:
            raise ValueError(f"BosonSite needs N_cut >= 2, got {self.N_cut}.")
        self.dim = int(self.N_cut)
        super().__post_init__()
        self.a, self.adag, self.n, self.x, self.p = _boson_operators(self.N_cut)

    def kerr(self, K: float) -> np.ndarray:
        """Single-photon Kerr operator: ``-K/2 * a_dag a_dag a a``."""
        return -0.5 * K * (self.adag @ self.adag @ self.a @ self.a)

    def coherent_state(self, alpha: complex) -> np.ndarray:
        """Return the coherent-state vector |alpha> truncated to N_cut."""
        N = self.N_cut
        vec = np.zeros(N, dtype=complex)
        # |alpha> = e^(-|alpha|^2 / 2) sum_n alpha^n / sqrt(n!) |n>
        norm = np.exp(-0.5 * abs(alpha) ** 2)
        coeff = 1.0
        vec[0] = norm
        for n in range(1, N):
            coeff *= alpha / np.sqrt(n)
            vec[n] = norm * coeff
        return vec

    def fock_state(self, n: int) -> np.ndarray:
        """Return the Fock state ``|n>`` as a vector of length N_cut."""
        if not (0 <= n < self.N_cut):
            raise ValueError(f"Fock index out of range: {n} not in [0, {self.N_cut}).")
        v = np.zeros(self.N_cut, dtype=complex)
        v[n] = 1.0
        return v


def boson_site(N_cut: int, name: str) -> BosonSite:
    """Convenience constructor: ``boson_site(8, 'c0')`` returns a Fock-8 cavity."""
    return BosonSite(name=name, N_cut=N_cut)
