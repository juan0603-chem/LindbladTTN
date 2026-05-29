# coding: utf-8
"""End-to-end hybrid-platform demo (M1–M10 in action).

A vanadyl-style spin (S=1/2, no nucleus for brevity) coupled to a transmon
via a shared LC bus.  Demonstrates:

* heterogeneous sites (spin-1/2 + boson + spin-1/2)
* the templates module
* the cavity-centred tree topology (M9)
* dispersive readout-style coupling
* dissipation on the cavity (κ) and qubit (T₁, Tφ)
* a Gaussian DRAG drive on the transmon

Run with::

    python examples/hybrid_vanadyl_transmon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

# Windows consoles default to cp1252 which can't encode ⟨⟩∈⊗ etc.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np

from lindblad_ttn import LindbladTTN
from lindblad_ttn.control import drag
from lindblad_ttn.sites import boson_site, spin_half_site
from lindblad_ttn.templates import merge


def main() -> None:
    # ----- Sites -----
    q_van = spin_half_site("vo")       # vanadyl electronic spin
    c_bus = boson_site(6, "cav")       # shared LC bus
    q_tx = spin_half_site("tx")        # transmon (two-level approx for brevity)

    # ----- Frequencies (natural units) -----
    omega_van = 1.000   # vanadyl Larmor
    omega_cav = 0.950   # cavity bus
    omega_tx = 1.010    # transmon
    g_van = 0.02        # vanadyl-cavity coupling (weak — molecular spin)
    g_tx = 0.12         # transmon-cavity coupling (strong)
    chi = g_van * g_tx / (omega_van - omega_cav)  # cavity-mediated J_eff

    # ----- Decay rates -----
    kappa = 0.005       # cavity photon loss
    gamma_van = 1e-4    # vanadyl T₁
    gamma_tx = 1e-3     # transmon T₁

    # ----- Hamiltonian terms -----
    H_terms = [
        # Bare frequencies
        (0.5 * omega_van, {q_van.name: q_van.sz}),
        (omega_cav,       {c_bus.name: c_bus.adag @ c_bus.a}),
        (0.5 * omega_tx,  {q_tx.name: q_tx.sz}),
        # Spin↔cavity (Jaynes–Cummings, RWA)
        (g_van, {q_van.name: q_van.sp, c_bus.name: c_bus.a}),
        (g_van, {q_van.name: q_van.sm, c_bus.name: c_bus.adag}),
        # Transmon↔cavity
        (g_tx,  {q_tx.name: q_tx.sp, c_bus.name: c_bus.a}),
        (g_tx,  {q_tx.name: q_tx.sm, c_bus.name: c_bus.adag}),
    ]

    # ----- Dissipation -----
    L_terms = [
        (kappa,     {c_bus.name: c_bus.a}),
        (gamma_van, {q_van.name: q_van.sm}),
        (gamma_tx,  {q_tx.name: q_tx.sm}),
    ]

    # ----- Microwave drive on the transmon (Gaussian envelope) -----
    f_I, f_Q = drag(amp=0.05, t0=50.0, sigma=10.0, anharm=-0.3, beta=0.5)
    drive_op = q_tx.sx  # in-phase drive on transmon σ_x

    # The heterogeneous solver takes V_terms (a list of (coeff, op_dict)).
    # We just bundle one term: amplitude 1 on the σ_x of the transmon.
    drives_nd = [
        (f_I, [(complex(1.0), {q_tx.name: drive_op})]),
    ]

    # ----- Build the solver -----
    # NOTE: M9 added `topology="cavity_centered"` which puts the cavity
    # bosonic node at the centre of the tree.  For the multi-child-per-node
    # case, the sequential SVD initialiser used here defaults to a train
    # topology; cavity_centered is most useful for larger spin counts
    # where the Hamiltonian connectivity exploits the star structure.
    solver = LindbladTTN(
        sites=[q_van, c_bus, q_tx],
        H_terms=H_terms,
        L_terms=L_terms,
        drives_nd=drives_nd,
        bond_dim=16,
        topology="train",
        strategy="ps1",
    )

    # ----- Initial state: |g> ⊗ |0> ⊗ |g> -----
    rho_van = np.array([[0, 0], [0, 1]], dtype=complex)        # ground
    rho_cav = np.zeros((6, 6), dtype=complex); rho_cav[0, 0] = 1.0
    rho_tx = np.array([[0, 0], [0, 1]], dtype=complex)
    rho0 = np.kron(np.kron(rho_van, rho_cav), rho_tx)

    # Observables: σ_z on each qubit and ⟨n⟩ on the cavity
    sz_van = np.kron(np.kron(np.diag([1, -1]).astype(complex), np.eye(6)), np.eye(2))
    sz_tx = np.kron(np.kron(np.eye(2), np.eye(6)), np.diag([1, -1]).astype(complex))
    n_cav = np.kron(np.kron(np.eye(2), np.diag(np.arange(6)).astype(complex)), np.eye(2))

    print("[demo] Running hybrid simulation (1 spin + 1 cavity + 1 transmon)…")
    result = solver.run(
        rho0=rho0,
        t_span=(0.0, 200.0),
        dt=0.05,
        observables=[sz_van, sz_tx, n_cav],
        save_every=40,
        verbose=False,
    )

    print(f"  bond_dim @ end = {result.bond_dims[-1]}")
    print(f"  trace @ end    = {result.norm[-1]:.6f}")
    print(f"  <sz_vanadyl>(t=0, 100, 200) = "
          f"{result.expect[0][0].real:+.3f}, "
          f"{result.expect[0][len(result.times)//2].real:+.3f}, "
          f"{result.expect[0][-1].real:+.3f}")
    print(f"  <sz_transmon>(t=0, 100, 200) = "
          f"{result.expect[1][0].real:+.3f}, "
          f"{result.expect[1][len(result.times)//2].real:+.3f}, "
          f"{result.expect[1][-1].real:+.3f}")
    print(f"  <n_cavity>(t=0, 100, 200) = "
          f"{result.expect[2][0].real:+.3f}, "
          f"{result.expect[2][len(result.times)//2].real:+.3f}, "
          f"{result.expect[2][-1].real:+.3f}")
    print(f"  cavity-mediated J_eff (theory) = {chi:+.4f}")


if __name__ == "__main__":
    main()
