# coding: utf-8
"""Site catalogue for heterogeneous TTN simulations.

A *site* carries (1) a local Hilbert-space dimension, (2) a DOF name, and
(3) a small library of named operators.  The :class:`Site` instances feed
into :class:`~lindblad_ttn.LindbladTTN` via the ``sites=`` parameter, which
unlocks bosonic modes, higher spins, and mixed-dimension trees.

Typical usage::

    from lindblad_ttn.sites import spin_half_site, boson_site

    q = spin_half_site("q0")
    c = boson_site(8, "c0")

    H_terms = [
        (omega_q,        {q.name: q.sz}),
        (omega_c,        {c.name: c.n}),
        (g,              {q.name: q.sp, c.name: c.a}),
        (g,              {q.name: q.sm, c.name: c.adag}),
    ]
    L_terms = [(gamma, {q.name: q.sm}), (kappa, {c.name: c.a})]
"""

from lindblad_ttn.sites.base import Site
from lindblad_ttn.sites.spin_half import SpinHalfSite, spin_half_site
from lindblad_ttn.sites.spin import SpinSite, spin_site, stevens_operator
from lindblad_ttn.sites.boson import BosonSite, boson_site

__all__ = [
    "Site",
    "SpinHalfSite", "spin_half_site",
    "SpinSite", "spin_site", "stevens_operator",
    "BosonSite", "boson_site",
]
