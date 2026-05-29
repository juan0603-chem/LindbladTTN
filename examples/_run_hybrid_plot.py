# coding: utf-8
"""Driver: run hybrid_vanadyl_transmon.py and save a plot of the trajectories.

Usage:
    py examples/_run_hybrid_plot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import matplotlib.pyplot as plt
import numpy as np

from lindblad_ttn import LindbladTTN
from lindblad_ttn.control import drag
from lindblad_ttn.sites import boson_site, spin_half_site


def main() -> None:
    # ----- Sites -----
    q_van = spin_half_site("vo")        # vanadyl electronic spin
    c_bus = boson_site(6, "cav")        # shared LC bus
    q_tx = spin_half_site("tx")         # transmon (two-level approx)

    # ----- Parameters (natural units) -----
    omega_van, omega_cav, omega_tx = 1.000, 0.950, 1.010
    g_van, g_tx = 0.02, 0.12
    chi = g_van * g_tx / (omega_van - omega_cav)  # cavity-mediated J_eff (theory)
    kappa, gamma_van, gamma_tx = 0.005, 1e-4, 1e-3

    # ----- Hamiltonian terms -----
    H_terms = [
        (0.5 * omega_van, {q_van.name: q_van.sz}),
        (omega_cav,       {c_bus.name: c_bus.adag @ c_bus.a}),
        (0.5 * omega_tx,  {q_tx.name: q_tx.sz}),
        (g_van, {q_van.name: q_van.sp, c_bus.name: c_bus.a}),
        (g_van, {q_van.name: q_van.sm, c_bus.name: c_bus.adag}),
        (g_tx,  {q_tx.name: q_tx.sp,  c_bus.name: c_bus.a}),
        (g_tx,  {q_tx.name: q_tx.sm,  c_bus.name: c_bus.adag}),
    ]
    L_terms = [
        (kappa,     {c_bus.name: c_bus.a}),
        (gamma_van, {q_van.name: q_van.sm}),
        (gamma_tx,  {q_tx.name: q_tx.sm}),
    ]

    # ----- DRAG drive on the transmon -----
    f_I, f_Q = drag(amp=0.05, t0=50.0, sigma=10.0, anharm=-0.3, beta=0.5)
    drives_nd = [
        (f_I, [(complex(1.0), {q_tx.name: q_tx.sx})]),
    ]

    solver = LindbladTTN(
        sites=[q_van, c_bus, q_tx],
        H_terms=H_terms,
        L_terms=L_terms,
        drives_nd=drives_nd,
        bond_dim=16,
        topology="train",
        strategy="ps1",
    )

    # |g> ⊗ |0> ⊗ |g>
    rho_van = np.array([[0, 0], [0, 1]], dtype=complex)
    rho_cav = np.zeros((6, 6), dtype=complex); rho_cav[0, 0] = 1.0
    rho_tx = np.array([[0, 0], [0, 1]], dtype=complex)
    rho0 = np.kron(np.kron(rho_van, rho_cav), rho_tx)

    # Observables
    sz_van = np.kron(np.kron(np.diag([1, -1]).astype(complex), np.eye(6)), np.eye(2))
    sz_tx  = np.kron(np.kron(np.eye(2), np.eye(6)), np.diag([1, -1]).astype(complex))
    n_cav  = np.kron(np.kron(np.eye(2), np.diag(np.arange(6)).astype(complex)), np.eye(2))

    # σx for the spins to see coherence; ⟨adag a⟩ already covered by n_cav
    sx_van = np.kron(np.kron(np.array([[0, 1], [1, 0]], dtype=complex),
                              np.eye(6)), np.eye(2))
    sx_tx  = np.kron(np.kron(np.eye(2), np.eye(6)),
                      np.array([[0, 1], [1, 0]], dtype=complex))

    print("[demo] Running hybrid simulation (vanadyl + cavity[N=6] + transmon)...")
    t_final = 200.0
    result = solver.run(
        rho0=rho0,
        t_span=(0.0, t_final),
        dt=0.05,
        observables=[sz_van, sz_tx, n_cav, sx_van, sx_tx],
        save_every=40,
        verbose=False,
    )

    times = np.asarray(result.times)
    sz_van_t = result.expect[0].real
    sz_tx_t  = result.expect[1].real
    n_cav_t  = result.expect[2].real
    sx_van_t = result.expect[3].real
    sx_tx_t  = result.expect[4].real

    print(f"  bond_dim @ end = {result.bond_dims[-1]}")
    print(f"  trace @ end    = {result.norm[-1]:.6f}")
    print(f"  cavity-mediated J_eff (theory chi) = {chi:+.5f}")
    print(f"  <sz_van>(0, t/2, T)   = "
          f"{sz_van_t[0]:+.3f}, {sz_van_t[len(times)//2]:+.3f}, {sz_van_t[-1]:+.3f}")
    print(f"  <sz_tx >(0, t/2, T)   = "
          f"{sz_tx_t[0]:+.3f}, {sz_tx_t[len(times)//2]:+.3f}, {sz_tx_t[-1]:+.3f}")
    print(f"  <n_cav >(0, t/2, T)   = "
          f"{n_cav_t[0]:+.3f}, {n_cav_t[len(times)//2]:+.3f}, {n_cav_t[-1]:+.3f}")

    # ---------------- Plot ----------------
    pulse_t = np.linspace(0.0, t_final, 800)
    f_I_t = np.asarray([f_I(t) for t in pulse_t])
    f_Q_t = np.asarray([f_Q(t) for t in pulse_t])

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 2, 1]})

    ax0, ax1, ax2, ax3 = axes
    ax0.plot(times, sz_van_t, color="C0", lw=1.6, label=r"$\langle\sigma_z\rangle_{\rm vanadyl}$")
    ax0.plot(times, sz_tx_t,  color="C3", lw=1.6, label=r"$\langle\sigma_z\rangle_{\rm transmon}$")
    ax0.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax0.set_ylabel(r"$\langle\sigma_z\rangle$")
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(alpha=0.3)
    ax0.set_title("Hybrid demo: vanadyl spin + 6-level cavity + transmon "
                  "(JC couplings, DRAG drive on transmon, T1 + cavity loss)")

    ax1.plot(times, sx_van_t, color="C0", lw=1.4, ls="--",
             label=r"$\langle\sigma_x\rangle_{\rm vanadyl}$")
    ax1.plot(times, sx_tx_t,  color="C3", lw=1.4, ls="--",
             label=r"$\langle\sigma_x\rangle_{\rm transmon}$")
    ax1.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax1.set_ylabel(r"$\langle\sigma_x\rangle$")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(times, n_cav_t, color="C2", lw=1.6, label=r"$\langle a^\dagger a\rangle_{\rm cavity}$")
    ax2.set_ylabel(r"cavity $\langle n\rangle$")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.3)

    ax3.plot(pulse_t, f_I_t, color="C5", lw=1.4, label="DRAG I (on $\\sigma_x$ of transmon)")
    ax3.plot(pulse_t, f_Q_t, color="C6", lw=1.0, ls=":", label="DRAG Q (unused here)")
    ax3.set_xlabel("time")
    ax3.set_ylabel("drive amp")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(alpha=0.3)

    # Highlight the pulse region
    for ax in axes:
        ax.axvspan(50.0 - 30.0, 50.0 + 30.0, color="C5", alpha=0.06)

    fig.tight_layout()
    out_png = _HERE / "hybrid_vanadyl_transmon.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"\nSaved plot to {out_png}")


if __name__ == "__main__":
    main()
