# coding: utf-8
"""Build the Lindblad Liouvillian as a SumOfProducts directly from
single-site operators — no Pauli decomposition, no ``4^N`` scan.

The user provides product-form operators::

    H_terms = [
        (omega_q, {"q0": sz}),                       # single-site
        (g,       {"q0": sm, "c0": adag}),           # two-site
        (g,       {"q0": sp, "c0": a   }),
    ]
    L_terms = [
        (gamma, {"q0": sm}),
        (kappa, {"c0": a }),
    ]

Each ``op_dict`` lists ONLY the sites where the operator is non-identity.
Missing sites are implicitly identity.  This sparse form is ``O(n_terms)``
in storage and avoids the exponential cost of full Pauli decomposition.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from lindblad_ttn.core.backend import to_torch
from lindblad_ttn.core.sop import SumOfProducts
from lindblad_ttn.physics.liouville_nd import (
    jump_local,
    left_local,
    right_local,
)


SiteOpDict = dict[str, np.ndarray]
HTerm = tuple[complex, SiteOpDict]
LTerm = tuple[float, SiteOpDict]


# ---------------------------------------------------------------------------
# Term-to-SoP conversion
# ---------------------------------------------------------------------------

def _vn_terms_from_h_term(
    coeff: complex,
    op_dict: SiteOpDict,
) -> list[tuple[complex, dict[str, torch.Tensor]]]:
    """Convert a product-form Hamiltonian term to two Liouville SoP terms.

    For ``H = coeff * (⊗_s A_s)`` (with identity on missing sites), the von
    Neumann commutator ``-i [H, rho]`` contributes::

        -i * coeff * (⊗_s left_local(A_s)) · vec(rho)
        +i * coeff * (⊗_s right_local(A_s)) · vec(rho)

    Returns
    -------
    list of (complex, dict[str, torch.Tensor])
        Exactly two SoP terms (left + right halves of -i[H,rho]).
    """
    left_part = {dof: to_torch(left_local(A)) for dof, A in op_dict.items()}
    right_part = {dof: to_torch(right_local(A)) for dof, A in op_dict.items()}
    return [
        (-1j * coeff, left_part),
        (+1j * coeff, right_part),
    ]


def _dissipator_terms_from_l_term(
    gamma: float,
    op_dict: SiteOpDict,
) -> list[tuple[complex, dict[str, torch.Tensor]]]:
    """Convert a product-form Lindblad term to three Liouville SoP terms.

    For ``L = ⊗_s A_s`` with rate ``gamma``, the dissipator contributes::

        +gamma * (⊗_s jump_local(A_s, A_s))     ← L rho L†
        -gamma/2 * (⊗_s left_local(A_s†A_s))    ← −½ L†L rho
        -gamma/2 * (⊗_s right_local(A_s†A_s))   ← −½ rho L†L

    Returns
    -------
    list of (complex, dict[str, torch.Tensor])
        Three SoP terms.
    """
    jump_part = {dof: to_torch(jump_local(A, A)) for dof, A in op_dict.items()}
    LdL_per_site = {dof: A.conj().T @ A for dof, A in op_dict.items()}
    left_part = {dof: to_torch(left_local(M)) for dof, M in LdL_per_site.items()}
    right_part = {dof: to_torch(right_local(M)) for dof, M in LdL_per_site.items()}
    return [
        (+complex(gamma), jump_part),
        (-0.5 * complex(gamma), left_part),
        (-0.5 * complex(gamma), right_part),
    ]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_lindblad_sop_nd(
    H_terms: Iterable[HTerm] | None,
    L_terms: Iterable[LTerm] | None,
) -> SumOfProducts:
    """Build a Liouvillian SoP from product-form Hamiltonian and Lindblad terms.

    Parameters
    ----------
    H_terms : iterable of (coeff, op_dict) or None
        Each entry contributes ``coeff * (⊗_s op_s)`` to the Hamiltonian.
    L_terms : iterable of (gamma, op_dict) or None
        Each entry contributes a Lindblad dissipator with rate ``gamma`` and
        jump operator ``L = ⊗_s op_s``.

    Returns
    -------
    SumOfProducts
        Liouvillian SoP in the interleaved per-site Liouville convention.
    """
    sop = SumOfProducts()

    if H_terms is not None:
        for coeff, op_dict in H_terms:
            for c, d in _vn_terms_from_h_term(complex(coeff), op_dict):
                if abs(c) > 0:
                    sop.add_term(c, d)

    if L_terms is not None:
        for gamma, op_dict in L_terms:
            if gamma == 0:
                continue
            for c, d in _dissipator_terms_from_l_term(float(gamma), op_dict):
                if abs(c) > 0:
                    sop.add_term(c, d)

    return sop


class LiouvillianSoPND:
    """Time-dependent Liouvillian built from product-form terms.

    Mirrors :class:`~lindblad_ttn.physics.liouvillian.LiouvillianSoP` but for
    arbitrary per-site dimensions.

    Parameters
    ----------
    H_terms : iterable of (coeff, op_dict)
        Static Hamiltonian terms.
    L_terms : iterable of (gamma, op_dict)
        Lindblad jump terms.
    V_terms_list : list of list of (coeff, op_dict), optional
        One list of H-terms per independent drive channel.  Each channel ``i``
        contributes ``f_i(t)·V_i`` to ``H(t)``.
    """

    def __init__(
        self,
        H_terms: Iterable[HTerm] | None,
        L_terms: Iterable[LTerm] | None,
        V_terms_list: list[Iterable[HTerm]] | None = None,
    ) -> None:
        self._sop_H0 = build_lindblad_sop_nd(H_terms, L_terms)
        self._sop_Vs: list[SumOfProducts] = (
            [build_lindblad_sop_nd(V_terms, None) for V_terms in V_terms_list]
            if V_terms_list else []
        )

    def build_sop_H0(self) -> SumOfProducts:
        return self._sop_H0

    def build_sop_Vs(self) -> list[SumOfProducts]:
        return list(self._sop_Vs)

    def __repr__(self) -> str:
        n_v = [sop.n_terms for sop in self._sop_Vs]
        return (
            f"LiouvillianSoPND(n_terms_H0={self._sop_H0.n_terms}, "
            f"n_drives={len(self._sop_Vs)}, n_terms_V={n_v})"
        )
