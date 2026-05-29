# coding: utf-8
"""Pulse-level 2-qubit circuit: Hadamard on q0, then CNOT(q0 → q1).

Both gates are driven by **Gaussian pulses** in the rotating frame, with T1
decay on both qubits throughout:

  • Hadamard ≡ Ry(π/2)   — Gaussian on V₁ = Y ⊗ I   (single-site drive)
  • CNOT     ≡ ZX(π/2)   — Gaussian on V₂ = Z ⊗ X   (two-site drive)
                            (locally equivalent to CNOT; this is the bare
                             cross-resonance entangler used on transmon HW)

Starting from |00⟩, the ideal noiseless output of CNOT·H₀ is a Bell state.
The locally-equivalent ZX(π/2) variant used here produces a maximally-
entangled state that differs from |Φ⁺⟩ only by single-qubit basis rotations.

pytenso-style multi-drive API
-----------------------------
This script uses ``LindbladTTN(... drives=[(f₁, V₁), (f₂, V₂)])`` — one
solver instance, both pulses live in the same SoP evaluated as
``H₀ + Σᵢ fᵢ(t)·Vᵢ`` at each timestep. That mirrors pytenso's
``f_list(time)`` mechanism in ``tenso/prototypes/heom.py``, where every
time-dependent channel is its own SoP term added on the fly. qutip handles
the same shape natively via a list of ``[op, callable]`` terms in ``mesolve``,
which we use as the reference.

Run
---
    py examples/qutip_circuit.py
"""

from __future__ import annotations

import sys
from functools import reduce
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

# Windows consoles default to cp1252 which can't encode ⟨⟩∈⊗ etc.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from lindblad_ttn import LindbladTTN

# Operators
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1],   [1, 0]],  dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0],   [0, -1]], dtype=complex)
sm = np.array([[0, 0],   [1, 0]],  dtype=complex)


def kron(*ops: np.ndarray) -> np.ndarray:
    return reduce(np.kron, ops)


YI  = kron(Y,  I2)         # Hadamard generator (Ry on q0)
ZX  = kron(Z,  X)          # CR / ZX entangler  (CNOT-equivalent generator)
ZI  = kron(Z,  I2)
IZ  = kron(I2, Z)
ZZ  = kron(Z,  Z)
XX  = kron(X,  X)
sm0 = kron(sm, I2)
sm1 = kron(I2, sm)

# -----------------------------------------------------------------------------
# Pulse calibration
# -----------------------------------------------------------------------------
# Propagator for H(t) = f(t)·V with Hermitian V is U = exp(-i V · ∫f dt).
# We want U = exp(-i(θ/2) V), so   ∫f(t) dt = θ/2.
# For a Gaussian f(t) = A·exp(-(t-t_c)²/(2σ²)):  ∫f dt = A·σ·√(2π)
# →  A = (θ/2) / (σ·√(2π)).
SIGMA_H        = 0.30      # Hadamard pulse width
SIGMA_CX       = 0.50      # CNOT     pulse width
N_SIGMAS       = 5         # window: ±5σ → envelope < 4e-6 outside
GATE_BUFFER    = 0.5       # idle between pulses (no rephasing in rotating frame)

T_H_CENTER   = N_SIGMAS * SIGMA_H                            # 1.5
T_H_END      = T_H_CENTER + N_SIGMAS * SIGMA_H               # 3.0
T_CX_CENTER  = T_H_END + GATE_BUFFER + N_SIGMAS * SIGMA_CX   # 6.0
T_CX_END     = T_CX_CENTER + N_SIGMAS * SIGMA_CX             # 8.5

THETA_H  = np.pi / 2       # Hadamard ≡ Ry(π/2)
THETA_CX = np.pi / 2       # CNOT     ≡ ZX(π/2)
A_H      = (THETA_H  / 2) / (SIGMA_H  * np.sqrt(2 * np.pi))
A_CX     = (THETA_CX / 2) / (SIGMA_CX * np.sqrt(2 * np.pi))

GAMMA = 0.0002               # T1 rate (both qubits)


def gauss(t: float, A: float, tc: float, sigma: float) -> float:
    if abs(t - tc) > N_SIGMAS * sigma:
        return 0.0
    return A * float(np.exp(-(t - tc) ** 2 / (2 * sigma ** 2)))


def f_h(t: float)  -> float: return gauss(t, A_H,  T_H_CENTER,  SIGMA_H)
def f_cx(t: float) -> float: return gauss(t, A_CX, T_CX_CENTER, SIGMA_CX)


# -----------------------------------------------------------------------------
# LindbladTTN — single solver, two drives (pytenso f_list style)
# -----------------------------------------------------------------------------
print("\nLindbladTTN: 2 drives in one run (H pulse on Y⊗I, CR pulse on Z⊗X)")
solver = LindbladTTN(
    H0=None,
    drives=[(f_h, YI), (f_cx, ZX)],
    L_ops=[(GAMMA, sm0), (GAMMA, sm1)],
    n_sites=2, bond_dim=16, topology="train", strategy="ps1",
    vmf_atol=1e-10,
)

rho0 = np.zeros((4, 4), dtype=complex);  rho0[0, 0] = 1.0       # |00⟩⟨00|
res = solver.run(
    rho0=rho0, t_span=(0.0, T_CX_END), dt=0.01,
    observables=[ZI, IZ, ZZ, XX], save_every=4, verbose=False,
)

ttn_times = np.concatenate([[0.0], res.times])


# -----------------------------------------------------------------------------
# qutip.mesolve — reference
# -----------------------------------------------------------------------------
dims2 = [[2, 2], [2, 2]]
H_q = [
    qt.Qobj(np.zeros((4, 4)), dims=dims2),
    [qt.Qobj(YI, dims=dims2), lambda t: f_h(t)],
    [qt.Qobj(ZX, dims=dims2), lambda t: f_cx(t)],
]
c_ops = [np.sqrt(GAMMA) * qt.Qobj(sm0, dims=dims2),
         np.sqrt(GAMMA) * qt.Qobj(sm1, dims=dims2)]
e_ops = [qt.Qobj(O, dims=dims2) for O in (ZI, IZ, ZZ, XX)]
qres = qt.mesolve(H_q, qt.Qobj(rho0, dims=dims2), ttn_times,
                  c_ops=c_ops, e_ops=e_ops)


# -----------------------------------------------------------------------------
# Report + plot
# -----------------------------------------------------------------------------
names     = ("ZI", "IZ", "ZZ", "XX")
operators = (ZI, IZ, ZZ, XX)
colors    = ("C0", "C1", "C2", "C3")

# Index where the Hadamard phase ends in the saved arrays
n_h = int(np.searchsorted(ttn_times, T_H_END, side="right"))

print("\nMax errors per phase:")
print(f"  {'observable':<12} {'phase 1 (H)':>14} {'phase 2 (CR)':>14} {'overall':>14}")

fig, (axE, axP) = plt.subplots(
    2, 1, figsize=(9, 6), sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)
for i, (name, O, c) in enumerate(zip(names, operators, colors)):
    ttn_init = float(np.trace(O @ rho0).real)
    ttn = np.concatenate([[ttn_init], res.expect[i].real])
    qu  = np.asarray(qres.expect[i]).real
    diff = np.abs(ttn - qu)
    err_h   = float(diff[:n_h].max())
    err_cx  = float(diff[n_h:].max()) if n_h < len(diff) else 0.0
    err_all = float(diff.max())
    print(f"  ⟨{name}⟩       {err_h:>14.2e} {err_cx:>14.2e} {err_all:>14.2e}")
    axE.plot(ttn_times, qu,  color=c, lw=2.0, label=f"qutip ⟨{name}⟩")
    axE.plot(ttn_times, ttn, color=c, lw=1.0, ls="--", label=f"TTN  ⟨{name}⟩")

# Pulse envelopes (visual context)
pulse_t = np.linspace(0.0, T_CX_END, 600)
axP.plot(pulse_t, [f_h(t)  for t in pulse_t], color="C4", lw=1.5, label="f_H(t)  (Y⊗I)")
axP.plot(pulse_t, [f_cx(t) for t in pulse_t], color="C5", lw=1.5, label="f_CX(t) (Z⊗X)")
axP.set_ylabel("envelope")
axP.set_xlabel("t")
axP.legend(loc="upper right", fontsize=9);  axP.grid(alpha=0.3)

# Shade pulse regions for orientation
for ax_ in (axE, axP):
    ax_.axvspan(0.0,       T_H_END,  color="C4", alpha=0.06)
    ax_.axvspan(T_H_END,   T_CX_END, color="C5", alpha=0.06)

axE.set_ylabel("expectation")
axE.set_title("Pulse-level Hadamard + CNOT (2 qubits, Gaussian) — LindbladTTN vs qutip.mesolve")
axE.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
axE.grid(alpha=0.3)

out = _HERE / "qutip_circuit.png"
fig.tight_layout();  fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved figure to {out}")

# Final-state diagnostic
print("\nFinal-state diagnostics:")
rho_ttn_final = res.rho_final
rho_qt_final  = np.asarray(
    qt.mesolve(H_q, qt.Qobj(rho0, dims=dims2), [0.0, T_CX_END],
               c_ops=c_ops).states[-1].full()
)
print(f"  max|ρ_TTN − ρ_qutip| = {np.max(np.abs(rho_ttn_final - rho_qt_final)):.2e}")
print(f"  Tr(ρ_TTN)  = {np.trace(rho_ttn_final).real:.6f}")
print(f"  Tr(ρ_qutip) = {np.trace(rho_qt_final).real:.6f}")
