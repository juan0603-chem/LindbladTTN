# coding: utf-8
"""Minimal LindbladTTN vs qutip — driven qubit (Rabi + decay).

  H(t) = ω/2 σz + Ω cos(ω_d t) σx
  L    = √γ σ−,   ρ₀ = |0⟩⟨0|

On-resonance drive (ω_d = ω): the population executes Rabi oscillations at
frequency Ω while T1 dissipation damps the envelope toward the steady state.

LindbladTTN evaluates H(t) on the fly via ``f(t)·V``; qutip uses the
``[H0, [V, f]]`` time-dependent format. Both should agree to ~1e-6.

Run
---
    py examples/qutip_driven.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from lindblad_ttn import LindbladTTN

# Operators
X  = np.array([[0, 1],   [1, 0]],  dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0],   [0, -1]], dtype=complex)
sm = np.array([[0, 0],   [1, 0]],  dtype=complex)

OMEGA   = 1.0    # qubit frequency
OMEGA_D = 1.0    # drive frequency (on resonance)
OMEGA_R = 0.4    # drive amplitude → Rabi frequency Ω_R (RWA)
GAMMA   = 0.05   # T1 decay rate
T_FINAL = 60.0


def drive(t: float) -> float:
    """Coherent drive envelope: Ω_R · cos(ω_d t)."""
    return OMEGA_R * float(np.cos(OMEGA_D * t))


# ---- LindbladTTN ----------------------------------------------------------
# LindbladTTN signature for H(t) = H0 + f(t)·V :
#   H0 = ω/2 σz,  V = σx,  f(t) = Ω_R cos(ω_d t).
solver = LindbladTTN(
    H0=0.5 * OMEGA * Z, f=drive, V=X,
    L_ops=[(GAMMA, sm)],
    n_sites=1, bond_dim=4, topology="train", strategy="ps1",
)
rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
res = solver.run(
    rho0=rho0, t_span=(0.0, T_FINAL), dt=0.01,
    observables=[Z, X, Y], save_every=4, verbose=False,
)

# ---- qutip.mesolve --------------------------------------------------------
# qutip 5.x: time-dependent piece is ``[op, callable(t)]``.
times = np.concatenate([[0.0], res.times])
H_q   = [qt.Qobj(0.5 * OMEGA * Z), [qt.Qobj(X), lambda t: drive(t)]]
qres  = qt.mesolve(
    H_q, qt.Qobj(rho0), times,
    c_ops=[np.sqrt(GAMMA) * qt.Qobj(sm)],
    e_ops=[qt.Qobj(Z), qt.Qobj(X), qt.Qobj(Y)],
)

# ---- Report + plot --------------------------------------------------------
names = ("σz", "σx", "σy")
operators = (Z, X, Y)
fig, ax = plt.subplots(figsize=(8, 4.5))
for i, (name, op, color) in enumerate(zip(names, operators, ("C0", "C1", "C2"))):
    ttn_initial = float(np.trace(op @ rho0).real)
    ttn = np.concatenate([[ttn_initial], res.expect[i].real])
    qu  = np.asarray(qres.expect[i]).real
    print(f"  ⟨{name}⟩:  max|TTN − qutip| = {np.max(np.abs(ttn - qu)):.2e}")
    ax.plot(times, qu,  color=color, lw=2.0, label=f"qutip ⟨{name}⟩")
    ax.plot(times, ttn, color=color, lw=1.0, ls="--", label=f"TTN ⟨{name}⟩")

ax.set_xlabel("t");  ax.set_ylabel("expectation")
ax.set_title(f"Driven qubit (Ω_R={OMEGA_R}, γ={GAMMA}) — LindbladTTN vs qutip.mesolve")
ax.legend(loc="lower right", fontsize=9);  ax.grid(alpha=0.3)
out = _HERE / "qutip_driven.png"
fig.tight_layout();  fig.savefig(out, dpi=130)
print(f"\nSaved figure to {out}")
