# coding: utf-8
"""Abstract Site base class — a (dim, name, operators) bundle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Site:
    """A physical site with a fixed local Hilbert-space dimension.

    Parameters
    ----------
    dim : int
        Local Hilbert-space dimension.
    name : str
        DOF name used in the Sum-of-Products dictionaries (e.g. ``'q0'``).

    Attributes
    ----------
    dim : int
    name : str
    """

    dim: int
    name: str

    def identity(self) -> np.ndarray:
        """Return the local identity operator as a (dim, dim) complex array."""
        return np.eye(self.dim, dtype=complex)

    def zero(self) -> np.ndarray:
        """Return a (dim, dim) zero matrix."""
        return np.zeros((self.dim, self.dim), dtype=complex)

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError(f"Site dim must be ≥ 1, got {self.dim}.")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"Site name must be a non-empty string, got {self.name!r}.")
