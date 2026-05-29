# coding: utf-8
"""Sum-of-Products (SoP) operator representation.

A Hamiltonian or Liouvillian is stored as a sum of tensor-product operators:

    L = Σ_k  c_k  ⊗_{s}  O_s^(k)

Each term maps a DOF name (string) to a local operator matrix.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import torch


class SumOfProducts:
    """Sum-of-products operator.

    Parameters
    ----------
    terms : list of (coeff, op_dict), optional
        Initial terms.  ``op_dict`` maps DOF name (str) to a local operator
        matrix (torch.Tensor).

    Examples
    --------
    >>> sop = SumOfProducts()
    >>> sop.add_term(-1j * 0.5, {'q0': H_local})
    >>> sop2 = sop + sop   # doubles all coefficients
    """

    def __init__(
        self,
        terms: list[tuple[complex, dict[str, torch.Tensor]]] | None = None,
    ) -> None:
        self.terms: list[tuple[complex, dict[str, torch.Tensor]]] = (
            list(terms) if terms is not None else []
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_term(self, coeff: complex, op_dict: dict[str, torch.Tensor]) -> None:
        """Append one product term.

        Parameters
        ----------
        coeff : complex
            Scalar coefficient.
        op_dict : dict[str, torch.Tensor]
            Mapping from DOF name to local operator matrix.
        """
        self.terms.append((complex(coeff), op_dict))

    # ------------------------------------------------------------------
    # Algebraic operations
    # ------------------------------------------------------------------

    def scale(self, scalar: complex) -> "SumOfProducts":
        """Return a new SumOfProducts with all coefficients multiplied by ``scalar``.

        Parameters
        ----------
        scalar : complex

        Returns
        -------
        SumOfProducts
        """
        new_terms = [(c * scalar, ops) for c, ops in self.terms]
        return SumOfProducts(new_terms)

    def __mul__(self, scalar: complex) -> "SumOfProducts":
        return self.scale(scalar)

    def __rmul__(self, scalar: complex) -> "SumOfProducts":
        return self.scale(scalar)

    def __add__(self, other: "SumOfProducts") -> "SumOfProducts":
        """Concatenate two SoP operators.

        Parameters
        ----------
        other : SumOfProducts

        Returns
        -------
        SumOfProducts
        """
        return SumOfProducts(self.terms + other.terms)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def n_terms(self) -> int:
        """Number of product terms."""
        return len(self.terms)

    def dof_names(self) -> set[str]:
        """Return the set of all DOF names appearing in any term."""
        names: set[str] = set()
        for _, op_dict in self.terms:
            names.update(op_dict.keys())
        return names

    def get_ops(self, dof_name: str) -> list[tuple[complex, torch.Tensor]]:
        """Return all ``(coeff, op)`` pairs for a given DOF name.

        Parameters
        ----------
        dof_name : str

        Returns
        -------
        list of (complex, torch.Tensor)
            Each entry corresponds to one product term that has an explicit
            operator on this DOF.  Terms without an entry for ``dof_name``
            act as identity on that DOF.
        """
        return [(c, ops[dof_name]) for c, ops in self.terms if dof_name in ops]

    def __repr__(self) -> str:
        return f"SumOfProducts(n_terms={self.n_terms}, dofs={sorted(self.dof_names())})"
