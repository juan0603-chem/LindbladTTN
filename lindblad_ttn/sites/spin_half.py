# coding: utf-8
"""Spin-1/2 site — the original qubit, exposed as a structured Site."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lindblad_ttn.sites.base import Site


_I2 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_SP = np.array([[0, 1], [0, 0]], dtype=complex)  # |0><1|: sz|0>=+|0>, raises |1>->|0>
_SM = np.array([[0, 0], [1, 0]], dtype=complex)  # |1><0|: lowers |0> -> |1>
_PROJ0 = np.array([[1, 0], [0, 0]], dtype=complex)
_PROJ1 = np.array([[0, 0], [0, 1]], dtype=complex)


@dataclass
class SpinHalfSite(Site):
    """A spin-1/2 (qubit) site with the standard Pauli/ladder operators.

    Conventions
    -----------
    * ``sz`` has eigenvalues ``(+1, -1)`` for ``(|0>, |1>)``.
    * ``sm = |1><0|`` lowers the excited state ``|0>`` to the ground ``|1>``
      — matches the existing LindbladTTN convention.
    * ``sp = sm†``.

    Attributes
    ----------
    sx, sy, sz : (2, 2) complex arrays
        Pauli matrices.
    sm, sp     : (2, 2) complex arrays
        Lowering / raising operators.
    proj0, proj1 : (2, 2) complex arrays
        Projectors onto ``|0>`` and ``|1>``.
    """

    dim: int = field(default=2, init=False)
    sx: np.ndarray = field(default_factory=lambda: _SX.copy(), repr=False)
    sy: np.ndarray = field(default_factory=lambda: _SY.copy(), repr=False)
    sz: np.ndarray = field(default_factory=lambda: _SZ.copy(), repr=False)
    sp: np.ndarray = field(default_factory=lambda: _SP.copy(), repr=False)
    sm: np.ndarray = field(default_factory=lambda: _SM.copy(), repr=False)
    proj0: np.ndarray = field(default_factory=lambda: _PROJ0.copy(), repr=False)
    proj1: np.ndarray = field(default_factory=lambda: _PROJ1.copy(), repr=False)


def spin_half_site(name: str) -> SpinHalfSite:
    """Convenience constructor for a spin-1/2 site.

    Parameters
    ----------
    name : str
        DOF name (e.g. ``'q0'``).

    Returns
    -------
    SpinHalfSite
    """
    return SpinHalfSite(name=name)
