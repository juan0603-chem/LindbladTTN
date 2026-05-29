# coding: utf-8
"""Spin-S site (M3) — higher spin with S_x, S_y, S_z and Stevens operators.

Conventions
-----------
We use the standard angular-momentum basis ``|S, m>`` with ``m`` running
from ``S`` (top) down to ``-S`` (bottom).  The matrix elements are::

    <m'|S_z|m>       = m * delta_{m',m}
    <m'|S_+|m>       = sqrt(S(S+1) - m(m+1)) * delta_{m', m+1}
    <m'|S_-|m>       = sqrt(S(S+1) - m(m-1)) * delta_{m', m-1}
    S_x = (S_+ + S_-)/2
    S_y = (S_+ - S_-)/(2i)

Stevens operators ``O_k^q(S)`` are the standard polynomials used in
molecular-magnetism literature for crystal-field Hamiltonians (Abragam &
Bleaney, "Electron Paramagnetic Resonance of Transition Ions", 1970).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np

from lindblad_ttn.sites.base import Site


def _spin_matrices(S: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (S_x, S_y, S_z, S_+, S_-) for spin S.

    Parameters
    ----------
    S : float
        Total spin (0.5, 1, 1.5, 2, ...).  Must be a non-negative half-integer.

    Returns
    -------
    Sx, Sy, Sz, Sp, Sm : (d, d) complex arrays where d = 2S + 1.
    """
    if S < 0 or abs(2 * S - round(2 * S)) > 1e-9:
        raise ValueError(f"S must be a non-negative half-integer, got {S}.")

    d = int(round(2 * S + 1))
    # Basis ordering: |S>, |S-1>, ..., |-S>  → index i has m = S - i.
    m_vals = np.array([S - i for i in range(d)], dtype=float)

    Sz = np.diag(m_vals).astype(complex)

    Sp = np.zeros((d, d), dtype=complex)
    Sm = np.zeros((d, d), dtype=complex)
    for i in range(d):
        m = m_vals[i]
        # S_+ |m> = sqrt(S(S+1) - m(m+1)) |m+1>  → row index of |m+1> is i-1
        if i > 0:
            Sp[i - 1, i] = sqrt(S * (S + 1) - m * (m + 1))
        # S_- |m> = sqrt(S(S+1) - m(m-1)) |m-1>  → row index of |m-1> is i+1
        if i < d - 1:
            Sm[i + 1, i] = sqrt(S * (S + 1) - m * (m - 1))

    Sx = 0.5 * (Sp + Sm)
    Sy = -0.5j * (Sp - Sm)
    return Sx, Sy, Sz, Sp, Sm


@dataclass
class SpinSite(Site):
    """A spin-S site with angular-momentum operators in the standard basis.

    Parameters
    ----------
    S : float
        Total spin quantum number.
    name : str
        DOF name (e.g. ``'q0'``, ``'I_V51'``).

    Attributes
    ----------
    S : float
    Sx, Sy, Sz, Sp, Sm : (2S+1, 2S+1) complex arrays
    """

    S: float = 0.5
    dim: int = field(default=2, init=False)
    Sx: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=complex), repr=False)
    Sy: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=complex), repr=False)
    Sz: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=complex), repr=False)
    Sp: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=complex), repr=False)
    Sm: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=complex), repr=False)

    def __post_init__(self) -> None:
        self.dim = int(round(2 * self.S + 1))
        super().__post_init__()
        Sx, Sy, Sz, Sp, Sm = _spin_matrices(self.S)
        self.Sx, self.Sy, self.Sz, self.Sp, self.Sm = Sx, Sy, Sz, Sp, Sm

    def Sz_squared(self) -> np.ndarray:
        return self.Sz @ self.Sz

    def S_squared(self) -> np.ndarray:
        return self.Sx @ self.Sx + self.Sy @ self.Sy + self.Sz @ self.Sz

    def stevens(self, k: int, q: int) -> np.ndarray:
        """Return the Stevens operator ``O_k^q`` for this spin."""
        return stevens_operator(k, q, self.S)


def spin_site(S: float, name: str) -> SpinSite:
    """Convenience constructor: ``spin_site(0.5, 'q0')`` returns S=1/2 site."""
    return SpinSite(name=name, S=S)


# ---------------------------------------------------------------------------
# Stevens operators
# ---------------------------------------------------------------------------

def stevens_operator(k: int, q: int, S: float) -> np.ndarray:
    """Build the Stevens operator ``O_k^q`` for spin ``S``.

    Implements the standard set used in molecular magnetism:

    * O_2^0  = 3 S_z^2 - X
    * O_2^2  = (S_+^2 + S_-^2) / 2
    * O_4^0  = 35 S_z^4 - (30 X - 25) S_z^2 + 3 X^2 - 6 X
    * O_4^2  = ((7 S_z^2 - X - 5)(S_+^2 + S_-^2) + (S_+^2 + S_-^2)(7 S_z^2 - X - 5)) / 4
    * O_4^4  = (S_+^4 + S_-^4) / 2
    * O_6^0  = 231 S_z^6 - (315 X - 735) S_z^4 + (105 X^2 - 525 X + 294) S_z^2
                - 5 X^3 + 40 X^2 - 60 X
    * O_6^6  = (S_+^6 + S_-^6) / 2

    where ``X = S(S+1)``.  These match Abragam & Bleaney (1970) and the
    convention used in the EasySpin software.

    Parameters
    ----------
    k : int
        Rank (2, 4, or 6 implemented).
    q : int
        Order ``-k ≤ q ≤ k``.  Only ``q in {0, 2, 4, 6}`` implemented for now
        (covers the dominant terms in lanthanide / vanadyl crystal fields).
    S : float
        Spin value.

    Returns
    -------
    np.ndarray
        Shape ``(2S+1, 2S+1)``, complex.
    """
    Sx, Sy, Sz, Sp, Sm = _spin_matrices(S)
    d = Sz.shape[0]
    I = np.eye(d, dtype=complex)
    X = S * (S + 1)

    if k == 2 and q == 0:
        return 3 * (Sz @ Sz) - X * I
    if k == 2 and q == 2:
        return 0.5 * (Sp @ Sp + Sm @ Sm)
    if k == 2 and q == -2:
        return -0.5j * (Sp @ Sp - Sm @ Sm)
    if k == 4 and q == 0:
        Sz2 = Sz @ Sz
        Sz4 = Sz2 @ Sz2
        return 35 * Sz4 - (30 * X - 25) * Sz2 + (3 * X**2 - 6 * X) * I
    if k == 4 and q == 2:
        Sz2 = Sz @ Sz
        A = 7 * Sz2 - X * I - 5 * I
        Sppm = Sp @ Sp + Sm @ Sm
        return 0.25 * (A @ Sppm + Sppm @ A)
    if k == 4 and q == 4:
        return 0.5 * (np.linalg.matrix_power(Sp, 4) + np.linalg.matrix_power(Sm, 4))
    if k == 6 and q == 0:
        Sz2 = Sz @ Sz
        Sz4 = Sz2 @ Sz2
        Sz6 = Sz2 @ Sz4
        return (
            231 * Sz6
            - (315 * X - 735) * Sz4
            + (105 * X**2 - 525 * X + 294) * Sz2
            + (-5 * X**3 + 40 * X**2 - 60 * X) * I
        )
    if k == 6 and q == 6:
        return 0.5 * (np.linalg.matrix_power(Sp, 6) + np.linalg.matrix_power(Sm, 6))

    raise NotImplementedError(
        f"Stevens operator O_{k}^{q} not implemented. "
        f"Currently supported: (k, q) in [(2, 0), (2, ±2), (4, 0), (4, 2), (4, 4), (6, 0), (6, 6)]."
    )
