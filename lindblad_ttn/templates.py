# coding: utf-8
"""Hamiltonian templates for hybrid molecular-spin / cQED platforms (M4).

Each template returns a tuple ``(site_objects, H_terms, L_terms)`` that
can be passed directly to :class:`~lindblad_ttn.LindbladTTN`::

    from lindblad_ttn.templates import jaynes_cummings
    sites, H_terms, L_terms = jaynes_cummings(
        omega_q=5e9, omega_c=4.8e9, g=50e6, N_cut=8,
    )
    solver = LindbladTTN(
        sites=sites, H_terms=H_terms, L_terms=L_terms,
        bond_dim=16, topology="train",
    )

Conventions
-----------
* Frequencies are angular: a quoted "5 GHz qubit" should be passed as
  ``omega_q = 2π × 5e9``.  The templates do NOT multiply by 2π for you.
* Spin operators follow :mod:`lindblad_ttn.sites.spin_half` and friends:
  ``sz`` has eigenvalues ``(+1, -1)`` for ``(|0>, |1>)``; ``sm = |1><0|``
  lowers the excited state.
* The Hamiltonian template carries dissipators (``L_terms``); pass an empty
  list to opt out.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from lindblad_ttn.sites import (
    BosonSite, Site, SpinHalfSite, SpinSite,
    boson_site, spin_half_site, spin_site,
)


# ---------------------------------------------------------------------------
# Single-system templates
# ---------------------------------------------------------------------------

def transmon(
    omega_q: float,
    alpha: float,
    N_cut: int = 4,
    T1: float | None = None,
    Tphi: float | None = None,
    name: str = "t0",
) -> tuple[list[BosonSite], list[tuple[complex, dict]], list[tuple[float, dict]]]:
    """Transmon as a Duffing oscillator on a truncated Fock space.

    Hamiltonian: ``H = omega_q a†a − (alpha/2) a†a†aa``.

    Parameters
    ----------
    omega_q : float
        Qubit (0↔1) angular frequency (rad/s if SI, dimensionless if natural).
    alpha : float
        Anharmonicity (typically negative, e.g. ``-2π × 300 MHz``).
    N_cut : int
        Fock cutoff (4 is typical for transmons; 6 for stronger drives).
    T1 : float, optional
        Energy-relaxation time.  Adds ``L = sqrt(1/T1)·a`` if present.
    Tphi : float, optional
        Pure dephasing time.  Adds ``L = sqrt(2/Tphi)·a†a`` if present.

    Returns
    -------
    sites, H_terms, L_terms
    """
    t = boson_site(N_cut, name)
    H_terms = [
        (complex(omega_q), {t.name: t.adag @ t.a}),
        (complex(-0.5 * alpha), {t.name: t.adag @ t.adag @ t.a @ t.a}),
    ]
    L_terms: list[tuple[float, dict]] = []
    if T1 is not None and T1 > 0:
        L_terms.append((1.0 / T1, {t.name: t.a}))
    if Tphi is not None and Tphi > 0:
        L_terms.append((2.0 / Tphi, {t.name: t.adag @ t.a}))
    return [t], H_terms, L_terms


def fluxonium(
    E_C: float,
    E_J: float,
    E_L: float,
    phi_ext: float = np.pi,
    N_cut: int = 30,
    T1: float | None = None,
    name: str = "f0",
) -> tuple[list[BosonSite], list[tuple[complex, dict]], list[tuple[float, dict]]]:
    """Fluxonium as a charge + cos(phi) potential on a charge-basis Fock space.

    H = 4 E_C n² − E_J cos(phi − phi_ext) + (1/2) E_L phi².

    We diagonalise the LC oscillator in a truncated Fock basis with
    n = i (a† − a) / sqrt(2 r),  phi = (a + a†) sqrt(r/2),
    where ``r = sqrt(8 E_C / E_L)`` is the impedance ratio.

    Parameters
    ----------
    E_C, E_J, E_L : float
        Capacitive, Josephson, and inductive energies (same units).
    phi_ext : float
        External flux in radians (π = sweet spot for the symmetric flux qubit).
    N_cut : int
        Fock cutoff; fluxoniums typically need 30–60 levels for convergence.
    T1 : float, optional
        Energy-relaxation time.

    Returns
    -------
    sites, H_terms, L_terms
    """
    fl = boson_site(N_cut, name)
    r = float(np.sqrt(8.0 * E_C / E_L))
    n_op = 1j * (fl.adag - fl.a) / np.sqrt(2.0 * r)
    phi_op = (fl.a + fl.adag) * np.sqrt(r / 2.0)

    # cos(phi - phi_ext) via matrix exponential of i(phi - phi_ext)
    arg = phi_op - phi_ext * np.eye(N_cut, dtype=complex)
    expi = _matrix_cos_sin(arg)
    cos_arg = expi[0]
    # H matrix
    H = (
        4.0 * E_C * (n_op @ n_op)
        - E_J * cos_arg
        + 0.5 * E_L * (phi_op @ phi_op)
    )
    H_terms = [(complex(1.0), {fl.name: H})]
    L_terms: list[tuple[float, dict]] = []
    if T1 is not None and T1 > 0:
        L_terms.append((1.0 / T1, {fl.name: fl.a}))
    return [fl], H_terms, L_terms


def bare_lc(
    omega_c: float,
    N_cut: int = 8,
    kappa: float | None = None,
    name: str = "c0",
) -> tuple[list[BosonSite], list[tuple[complex, dict]], list[tuple[float, dict]]]:
    """Bare LC resonator: ``H = omega_c a†a``.

    Parameters
    ----------
    omega_c : float
        Resonator angular frequency.
    N_cut : int
        Fock cutoff.
    kappa : float, optional
        Photon-loss rate (adds ``L = sqrt(kappa)·a``).
    """
    c = boson_site(N_cut, name)
    H_terms = [(complex(omega_c), {c.name: c.adag @ c.a})]
    L_terms: list[tuple[float, dict]] = []
    if kappa is not None and kappa > 0:
        L_terms.append((kappa, {c.name: c.a}))
    return [c], H_terms, L_terms


# ---------------------------------------------------------------------------
# Spin-qubit templates
# ---------------------------------------------------------------------------

def vanadyl_spin(
    omega_q: float,
    A_iso: float = 0.0,
    A_dip: float = 0.0,
    T1: float | None = None,
    T2: float | None = None,
    include_nucleus: bool = False,
    name_e: str = "q0",
    name_n: str = "I0",
) -> tuple[list[Site], list[tuple[complex, dict]], list[tuple[float, dict]]]:
    """Vanadyl-style spin qubit: S=1/2 electron + optional 51V nucleus.

    Hamiltonian (with nucleus): ``H = (omega_q/2) Sz + A_iso S·I + nuclear Zeeman``.

    Parameters
    ----------
    omega_q : float
        Electron Larmor frequency (rad/s).
    A_iso : float
        Isotropic hyperfine coupling between electron and nucleus (rad/s).
        Ignored when ``include_nucleus=False``.
    A_dip : float
        Anisotropic (dipolar) component; modifies ``Sz Iz`` term.
    T1, T2 : float, optional
        Electron-spin relaxation and coherence times.
    include_nucleus : bool
        If True, add a 51V nuclear-spin site (I = 7/2).
    """
    elec = spin_half_site(name_e)
    sites: list[Site] = [elec]
    H_terms: list[tuple[complex, dict]] = [
        (complex(0.5 * omega_q), {elec.name: elec.sz}),
    ]
    L_terms: list[tuple[float, dict]] = []
    if T1 is not None and T1 > 0:
        L_terms.append((1.0 / T1, {elec.name: elec.sm}))
    if T2 is not None and T2 > 0:
        gamma_phi = max(0.0, 1.0 / (2.0 * T2) - (1.0 / T1 / 2.0 if T1 else 0.0))
        if gamma_phi > 0:
            L_terms.append((gamma_phi, {elec.name: elec.sz}))

    if include_nucleus:
        nuc = spin_site(3.5, name_n)
        sites.append(nuc)
        if A_iso != 0.0:
            for axis_e, axis_n in (
                (elec.sx, nuc.Sx), (elec.sy, nuc.Sy), (elec.sz, nuc.Sz),
            ):
                H_terms.append(
                    (complex(A_iso), {elec.name: axis_e, nuc.name: axis_n})
                )
        if A_dip != 0.0:
            H_terms.append(
                (complex(A_dip), {elec.name: elec.sz, nuc.name: nuc.Sz})
            )
    return sites, H_terms, L_terms


def lanthanide_smm(
    g_J: float,
    omega: float,
    J: float,
    B20: float = 0.0,
    B40: float = 0.0,
    B44: float = 0.0,
    name: str = "Ln0",
) -> tuple[list[SpinSite], list[tuple[complex, dict]], list[tuple[float, dict]]]:
    """Lanthanide SMM template (TbPc2-style or Dy-SIM-style).

    Hamiltonian: ``H = g_J μ_B B J_z + B_2^0 O_2^0(J) + B_4^0 O_4^0(J) + B_4^4 O_4^4(J)``,
    written in natural units (B field absorbed into ``omega = g_J μ_B B``).

    Parameters
    ----------
    g_J : float
        Landé g-factor of the J multiplet (decorative — only used for naming).
    omega : float
        Zeeman frequency ``g_J μ_B B`` (rad/s).
    J : float
        Total angular momentum (e.g. 6 for Tb³⁺, 7.5 for Dy³⁺).
    B20, B40, B44 : float
        Stevens-operator coefficients.
    """
    ln = spin_site(J, name)
    H_terms: list[tuple[complex, dict]] = [
        (complex(omega), {ln.name: ln.Sz}),
    ]
    if B20 != 0.0:
        H_terms.append((complex(B20), {ln.name: ln.stevens(2, 0)}))
    if B40 != 0.0:
        H_terms.append((complex(B40), {ln.name: ln.stevens(4, 0)}))
    if B44 != 0.0:
        H_terms.append((complex(B44), {ln.name: ln.stevens(4, 4)}))
    return [ln], H_terms, []


# ---------------------------------------------------------------------------
# Hybrid templates: spin + cavity
# ---------------------------------------------------------------------------

def jaynes_cummings(
    omega_q: float,
    omega_c: float,
    g: float,
    N_cut: int = 8,
    gamma_q: float | None = None,
    kappa: float | None = None,
    name_q: str = "q0",
    name_c: str = "c0",
) -> tuple[list[Site], list[tuple[complex, dict]], list[tuple[float, dict]]]:
    """Spin-1/2 + bosonic cavity in the Jaynes–Cummings (RWA) limit.

    ``H = (omega_q/2) sz + omega_c a†a + g (sp a + sm a†)``.

    Parameters
    ----------
    omega_q, omega_c : float
        Qubit and cavity angular frequencies.
    g : float
        Single-photon Rabi coupling (rad/s).
    N_cut : int
        Cavity Fock cutoff.
    gamma_q, kappa : float, optional
        Spin relaxation rate and cavity photon-loss rate.
    """
    q = spin_half_site(name_q)
    c = boson_site(N_cut, name_c)
    H_terms = [
        (complex(0.5 * omega_q), {q.name: q.sz}),
        (complex(omega_c),        {c.name: c.adag @ c.a}),
        (complex(g),              {q.name: q.sp, c.name: c.a}),
        (complex(g),              {q.name: q.sm, c.name: c.adag}),
    ]
    L_terms: list[tuple[float, dict]] = []
    if gamma_q is not None and gamma_q > 0:
        L_terms.append((gamma_q, {q.name: q.sm}))
    if kappa is not None and kappa > 0:
        L_terms.append((kappa, {c.name: c.a}))
    return [q, c], H_terms, L_terms


def tavis_cummings(
    omega_qs: list[float],
    omega_c: float,
    gs: list[float],
    N_cut: int = 8,
    gamma_qs: list[float] | None = None,
    kappa: float | None = None,
    name_c: str = "c0",
    name_prefix: str = "q",
) -> tuple[list[Site], list[tuple[complex, dict]], list[tuple[float, dict]]]:
    """N spins + 1 cavity in the Tavis–Cummings limit.

    ``H = Σ_i (omega_i/2) sz_i + omega_c a†a + Σ_i g_i (sp_i a + sm_i a†)``.
    """
    if len(gs) != len(omega_qs):
        raise ValueError("len(gs) must equal len(omega_qs).")
    if gamma_qs is not None and len(gamma_qs) != len(omega_qs):
        raise ValueError("len(gamma_qs) must equal len(omega_qs).")

    spins = [spin_half_site(f"{name_prefix}{i}") for i in range(len(omega_qs))]
    c = boson_site(N_cut, name_c)
    sites: list[Site] = spins + [c]
    H_terms: list[tuple[complex, dict]] = [
        (complex(omega_c), {c.name: c.adag @ c.a}),
    ]
    L_terms: list[tuple[float, dict]] = []
    for i, q in enumerate(spins):
        H_terms.append((complex(0.5 * omega_qs[i]), {q.name: q.sz}))
        H_terms.append((complex(gs[i]), {q.name: q.sp, c.name: c.a}))
        H_terms.append((complex(gs[i]), {q.name: q.sm, c.name: c.adag}))
        if gamma_qs is not None and gamma_qs[i] > 0:
            L_terms.append((gamma_qs[i], {q.name: q.sm}))
    if kappa is not None and kappa > 0:
        L_terms.append((kappa, {c.name: c.a}))
    return sites, H_terms, L_terms


def dispersive_readout(
    omega_q: float,
    omega_c: float,
    chi: float,
    N_cut: int = 6,
    gamma_q: float | None = None,
    kappa: float | None = None,
    name_q: str = "q0",
    name_c: str = "c0",
) -> tuple[list[Site], list[tuple[complex, dict]], list[tuple[float, dict]]]:
    """Dispersive Hamiltonian: ``H = (ω_q/2) sz + (ω_c + χ sz) a†a``.

    The χ shift produces the qubit-state-dependent cavity frequency used for
    standard cQED readout.  Equivalent (up to Lamb-shift) to the JC template
    in the dispersive regime ``|g/Δ| ≪ 1``.
    """
    q = spin_half_site(name_q)
    c = boson_site(N_cut, name_c)
    H_terms = [
        (complex(0.5 * omega_q), {q.name: q.sz}),
        (complex(omega_c), {c.name: c.adag @ c.a}),
        (complex(chi), {q.name: q.sz, c.name: c.adag @ c.a}),
    ]
    L_terms: list[tuple[float, dict]] = []
    if gamma_q is not None and gamma_q > 0:
        L_terms.append((gamma_q, {q.name: q.sm}))
    if kappa is not None and kappa > 0:
        L_terms.append((kappa, {c.name: c.a}))
    return [q, c], H_terms, L_terms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def merge(
    *bundles: tuple[list[Site], list[tuple[complex, dict]], list[tuple[float, dict]]],
) -> tuple[list[Site], list[tuple[complex, dict]], list[tuple[float, dict]]]:
    """Merge several ``(sites, H_terms, L_terms)`` bundles into one.

    Site names must be unique across all bundles.
    """
    all_sites: list[Site] = []
    seen: set[str] = set()
    H_all: list[tuple[complex, dict]] = []
    L_all: list[tuple[float, dict]] = []
    for sites, H_terms, L_terms in bundles:
        for s in sites:
            if s.name in seen:
                raise ValueError(
                    f"Duplicate site name {s.name!r} when merging templates."
                )
            seen.add(s.name)
            all_sites.append(s)
        H_all.extend(H_terms)
        L_all.extend(L_terms)
    return all_sites, H_all, L_all


def _matrix_cos_sin(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (cos(M), sin(M)) via matrix exponential.

    cos(M) = (exp(iM) + exp(-iM)) / 2
    sin(M) = (exp(iM) − exp(-iM)) / (2i)
    """
    from scipy.linalg import expm
    e_pos = expm(1j * M)
    e_neg = expm(-1j * M)
    return 0.5 * (e_pos + e_neg), -0.5j * (e_pos - e_neg)
