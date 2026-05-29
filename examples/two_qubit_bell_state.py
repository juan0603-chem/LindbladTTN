# coding: utf-8
"""
Two-qubit Bell-state generation via Rx(pi/2) + cross-resonance gate (ns scale).

Protocol
--------
  1. Rx(pi/2) on qubit 0 (2 ns rectangular X pulse, Omega/(2pi) = 62.5 MHz)
     H_drive = Omega(t) * X x I  with Omega(t) = Omega_H for t < T_H, else 0.
     U = exp(-i Omega_H T_H X) = Rx(pi/2): |0> -> (|0> - i|1>)/sqrt(2)

  2. Cross-resonance (ZX) entangling gate (~25 ns)
     Static coupling H0 = J/2 * Z x X  with J = 2*pi * 10 MHz.
     After time T_CX = pi/(2J) the joint state becomes a Bell state.

Ideal noiseless final state starting from |00>:
  |Phi> = (|00> + |11> - i|01> - i|10>) / 2   =>  concurrence C = 1

Physical parameters (superconducting transmon qubits, typical values):
  J/(2pi) = 10 MHz,  T1 = 100 us,  T2 = 50 us,  total gate time ~27 ns.

This example uses a direct (exact) Lindblad master-equation solver —
integrating the 16-dimensional Liouville-space ODE with 4th-order Runge-Kutta.
This is exact for a 2-qubit system (no TTN truncation) and serves as a
reference/validation for the TTN solver.

Run
---
  python examples/two_qubit_bell_state.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

_HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Pauli matrices
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X  = np.array([[0,  1 ], [1,  0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1,  0 ], [0, -1]], dtype=complex)
sm = np.array([[0,  0 ], [1,  0]], dtype=complex)   # lowering |1><0|


def kron(*ops: np.ndarray) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


# ---------------------------------------------------------------------------
# Physical parameters  (times in ns, angular frequencies in GHz * 2*pi)
# ---------------------------------------------------------------------------
J         = 2 * np.pi * 0.010   # ZX coupling 10 MHz  => T_CX ~25 ns
T1        = 100_000.0            # T1 = 100 us in ns
T2        =  50_000.0            # T2 =  50 us in ns
gamma1    = 1.0 / T1
gamma_phi = 1.0 / (2 * T2) - gamma1 / 2

T_H   = 2.0
T_CX  = np.pi / (2 * J)          # ~25.13 ns
T_tot = T_H + T_CX                # ~27.13 ns

# Drive: U = exp(-i Omega_H T_H X) = Rx(pi/2)  =>  Omega_H * T_H = pi/4
Omega_H = (np.pi / 4.0) / T_H    # 62.5 MHz (2*pi)

# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
H0 = (J / 2.0) * kron(Z, X)
V  = kron(X, I2)

def drive(t: float) -> float:
    return float(Omega_H if t < T_H else 0.0)

L_ops = [
    (gamma1,    kron(sm, I2)),
    (gamma1,    kron(I2, sm)),
    (gamma_phi, kron(Z,  I2)),
    (gamma_phi, kron(I2, Z )),
]

# ---------------------------------------------------------------------------
# Liouville-space superoperator helpers
# ---------------------------------------------------------------------------
d = 4    # Hilbert-space dimension (2 qubits)
D = d*d  # Liouville-space dimension


def vec(rho: np.ndarray) -> np.ndarray:
    """Row-major vectorisation: vec(rho)_{i*d+j} = rho_{ij}."""
    return rho.reshape(-1)


def unvec(v: np.ndarray) -> np.ndarray:
    return v.reshape(d, d)


def left_sop(A: np.ndarray) -> np.ndarray:
    """A x I: left multiplication superoperator."""
    return np.kron(A, np.eye(d, dtype=complex))


def right_sop(B: np.ndarray) -> np.ndarray:
    """I x B^T: right multiplication superoperator."""
    return np.kron(np.eye(d, dtype=complex), B.T)


def commutator_sop(H: np.ndarray) -> np.ndarray:
    """Unitary Liouvillian: -i[H, .] = -i(H x I - I x H^T)."""
    return -1j * (left_sop(H) - right_sop(H))


def dissipator_sop(gamma: float, L: np.ndarray) -> np.ndarray:
    """gamma ( L rho L^dag - 1/2 L^dag L rho - 1/2 rho L^dag L )."""
    LdL = L.conj().T @ L
    return gamma * (
        np.kron(L, L.conj())
        - 0.5 * left_sop(LdL)
        - 0.5 * right_sop(LdL)
    )


def build_liouvillian(t: float) -> np.ndarray:
    """Build the full 16x16 time-dependent Liouvillian at time t."""
    H_t = H0 + drive(t) * V
    L_mat = commutator_sop(H_t)
    for gamma, Lj in L_ops:
        L_mat = L_mat + dissipator_sop(gamma, Lj)
    return L_mat


# ---------------------------------------------------------------------------
# RK4 time integration
# ---------------------------------------------------------------------------
dt     = 0.25          # time step in ns
n_steps = int(round(T_tot / dt))

rho0 = np.zeros((d, d), dtype=complex)
rho0[0, 0] = 1.0

v = vec(rho0)

# Observables: 4 populations + 6 coherences (upper triangle)
basis_labels = ["|00>", "|01>", "|10>", "|11>"]
coh_pairs  = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
coh_labels = ["rho_01", "rho_02", "rho_03", "rho_12", "rho_13", "rho_23"]

# Storage
t_arr  = np.zeros(n_steps)
pop    = np.zeros((4, n_steps))       # populations
coh    = np.zeros((6, n_steps), dtype=complex)  # coherences (complex)
norm_t = np.zeros(n_steps)           # Tr(rho)

t = 0.0
for step in range(n_steps):
    # Midpoint SoP evaluation
    t_mid = t + 0.5 * dt
    L_mid = build_liouvillian(t_mid)

    # RK4
    k1 = L_mid @ v
    k2 = L_mid @ (v + 0.5 * dt * k1)
    k3 = L_mid @ (v + 0.5 * dt * k2)
    k4 = L_mid @ (v + dt * k3)
    v  = v + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    t += dt

    rho = unvec(v)
    t_arr[step]  = t
    norm_t[step] = np.trace(rho).real
    for i in range(4):
        pop[i, step] = rho[i, i].real
    for k, (i, j) in enumerate(coh_pairs):
        coh[k, step] = rho[i, j]

coh_abs = np.abs(coh)

# ---------------------------------------------------------------------------
# Concurrence & final-state metrics
# ---------------------------------------------------------------------------
def concurrence(rho: np.ndarray) -> float:
    sysy = kron(Y, Y)
    rho_tilde = sysy @ rho.conj() @ sysy
    ev  = np.linalg.eigvals(rho @ rho_tilde)
    lam = np.sort(np.sqrt(np.maximum(ev.real, 0.0)))[::-1]
    return float(max(0.0, lam[0] - lam[1] - lam[2] - lam[3]))


def purity(rho: np.ndarray) -> float:
    return float(np.trace(rho @ rho).real)


C_t = np.array([concurrence(unvec(v_s))
                for v_s in [vec(unvec(v))]])   # just final; compute below

# Compute time-resolved concurrence (rerun fast since we already have pop/coh)
C_t = np.zeros(n_steps)
rho_run = vec(rho0)
t_run = 0.0
for step in range(n_steps):
    t_mid = t_run + 0.5 * dt
    Lm = build_liouvillian(t_mid)
    k1 = Lm @ rho_run
    k2 = Lm @ (rho_run + 0.5*dt*k1)
    k3 = Lm @ (rho_run + 0.5*dt*k2)
    k4 = Lm @ (rho_run + dt*k3)
    rho_run = rho_run + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    t_run += dt
    C_t[step] = concurrence(unvec(rho_run))

rho_f = unvec(v)
C_f   = concurrence(rho_f)
P_f   = purity(rho_f)

phi_ideal = np.array([1, -1j, -1j, 1], dtype=complex) / 2.0
rho_ideal = np.outer(phi_ideal, phi_ideal.conj())
fidelity  = float(np.trace(rho_ideal @ rho_f).real)
C_ideal   = concurrence(rho_ideal)

# Also compute Sz and correlator observables from final step
sz0_t  = np.array([rho_s @ kron(Z, I2) for rho_s in [rho_f]])   # just fill from rho loop
xx_t   = []
sz0_arr = []
sz1_arr = []
zz_arr  = []
xx_arr  = []

# One more pass to get correlation observables
rho_run2 = vec(rho0)
for step in range(n_steps):
    t_mid2 = step * dt + 0.5 * dt
    Lm = build_liouvillian(t_mid2)
    k1 = Lm @ rho_run2
    k2 = Lm @ (rho_run2 + 0.5*dt*k1)
    k3 = Lm @ (rho_run2 + 0.5*dt*k2)
    k4 = Lm @ (rho_run2 + dt*k3)
    rho_run2 = rho_run2 + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    rho_s = unvec(rho_run2)
    sz0_arr.append(np.trace(kron(Z, I2) @ rho_s).real)
    sz1_arr.append(np.trace(kron(I2, Z ) @ rho_s).real)
    zz_arr.append (np.trace(kron(Z,  Z ) @ rho_s).real)
    xx_arr.append (np.trace(kron(X,  X ) @ rho_s).real)

sz0_arr = np.array(sz0_arr)
sz1_arr = np.array(sz1_arr)
zz_arr  = np.array(zz_arr)
xx_arr  = np.array(xx_arr)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
sep = "-" * 60
print(f"\n{sep}")
print(f"  Two-qubit Rx(pi/2) + cross-resonance gate  [exact solver]")
print(sep)
print(f"  System   : 2 superconducting transmon qubits")
print(f"  J/(2pi)  = {J/(2*np.pi)*1e3:.1f} MHz  (ZX cross-resonance)")
print(f"  T1 / T2  = {T1/1000:.0f} us / {T2/1000:.0f} us")
print(f"  Drive    : Omega/(2pi) = {Omega_H/(2*np.pi)*1e3:.1f} MHz for {T_H:.1f} ns")
print(sep)
print(f"  Rx(pi/2) pulse : {T_H:.2f} ns")
print(f"  ZX gate        : {T_CX:.2f} ns  (pi/(2J))")
print(f"  Total          : {T_tot:.2f} ns")
print(sep)
print(f"  Results at t = {T_tot:.2f} ns")
print(f"    Tr(rho)         = {norm_t[-1]:.8f}  (trace preservation)")
print(f"    Purity          = {P_f:.6f}  (1 = pure state)")
print(f"    Concurrence C   = {C_f:.6f}  (target: {C_ideal:.4f})")
print(f"    Fidelity F      = {fidelity:.6f}  (overlap with ideal Bell state)")
print(f"    <Sz> qubit 0    = {sz0_arr[-1]:+.6f}")
print(f"    <Sz> qubit 1    = {sz1_arr[-1]:+.6f}")
print(f"    <ZZ>            = {zz_arr[-1]:+.6f}")
print(f"    <XX>            = {xx_arr[-1]:+.6f}")
print(sep)
print(f"\n  Final density matrix |rho| (absolute values):")
for row in np.abs(rho_f):
    print("    " + "  ".join(f"{v:.4f}" for v in row))
print(f"\n  Ideal Bell state |(|00>+|11>-i|01>-i|10>)/2|:")
for row in np.abs(rho_ideal):
    print("    " + "  ".join(f"{v:.4f}" for v in row))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    pop_colors = ["#2166ac", "#d6604d", "#4dac26", "#8856a7"]
    coh_colors = ["#1b7837", "#762a83", "#e08214", "#4393c3", "#d73027", "#808080"]
    pop_ls     = ["-", "--", "-.", ":"]

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    fig.suptitle(
        "Two-qubit Bell-state generation\n"
        "Rx(pi/2) on qubit 0  +  ZX cross-resonance gate  (exact Lindblad, ns scale)",
        fontsize=11, y=0.99,
    )

    def shade(ax):
        ax.axvspan(0,    T_H,   alpha=0.08, color="gold",    zorder=0)
        ax.axvspan(T_H,  T_tot, alpha=0.08, color="skyblue", zorder=0)
        ax.axvline(T_H, color="k", ls="--", lw=1, alpha=0.5, zorder=1)

    # ------------------------------------------------------------------
    # Panel 1 — Populations
    # ------------------------------------------------------------------
    ax = axes[0]
    shade(ax)
    for i in range(4):
        ax.plot(t_arr, pop[i], color=pop_colors[i], lw=2,
                ls=pop_ls[i], label=basis_labels[i])
    ax.set_ylabel("Population", fontsize=10)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=9, ncol=4, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_title("Populations  rho_ii(t)", fontsize=10)

    # ------------------------------------------------------------------
    # Panel 2 — Coherence magnitudes
    # ------------------------------------------------------------------
    ax = axes[1]
    shade(ax)
    for k in range(6):
        ax.plot(t_arr, coh_abs[k], color=coh_colors[k], lw=2,
                label=f"|{coh_labels[k]}|")
    ax.set_ylabel("|rho_ij|", fontsize=10)
    ax.set_ylim(-0.02, 0.55)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_title("Coherence magnitudes  |rho_ij(t)|", fontsize=10)

    # ------------------------------------------------------------------
    # Panel 3 — Wootters concurrence
    # ------------------------------------------------------------------
    ax = axes[2]
    shade(ax)
    ax.plot(t_arr, C_t, color="#c0392b", lw=2.5, label="C(t)")
    ax.axhline(1.0, color="orange", ls=":", lw=1.5, label="C = 1 (Bell state)")
    ax.set_ylabel("Concurrence C", fontsize=10)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25)
    ax.set_title("Entanglement: Wootters concurrence  C(t)", fontsize=10)

    # ------------------------------------------------------------------
    # Panel 4 — Trace
    # ------------------------------------------------------------------
    ax = axes[3]
    shade(ax)
    ax.plot(t_arr, norm_t, color="#27ae60", lw=2, label="Tr(rho)")
    ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Time (ns)", fontsize=10)
    ax.set_ylabel("Tr(rho)", fontsize=10)
    ax.set_ylim(0.9990, 1.0010)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)
    ax.set_title("Trace preservation  Tr[rho(t)]", fontsize=10)

    # Legend for shaded regions on last panel
    drive_patch = mpatches.Patch(color="gold",    alpha=0.4,
                                  label=f"Drive on  [0, {T_H} ns]")
    cx_patch    = mpatches.Patch(color="skyblue", alpha=0.4,
                                  label=f"ZX only   [{T_H:.0f}, {T_tot:.1f} ns]")
    vline_leg   = Line2D([0], [0], color="k", ls="--", lw=1,
                          label=f"End of drive  t = {T_H} ns")
    axes[3].legend(handles=[drive_patch, cx_patch, vline_leg],
                   fontsize=8, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02), ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = _HERE / "bell_state_dynamics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  [Plot saved to {out_path}]")
    plt.show()

except ImportError:
    print("\n  [matplotlib not found -- skipping plot]")
