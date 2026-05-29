# coding: utf-8
"""Minimal LindbladTTN vs qutip comparison — single qubit decay.

  H = ω/2 σz,   L = √γ σ−,   ρ₀ = |0⟩⟨0|

Both solvers integrate the same master equation; their ⟨σz, σx, σy⟩
trajectories should agree to ~1e-7.

Run
---
    py examples/qutip_simple.py
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

# Operators (numpy)
X  = np.array([[0, 1],   [1, 0]],  dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0],   [0, -1]], dtype=complex)
sm = np.array([[0, 0],   [1, 0]],  dtype=complex)

OMEGA = 1.0
GAMMA = 0.3
T_FINAL = 15.0

# ---- LindbladTTN ----------------------------------------------------------
# Note: LindbladTTN takes ``L_ops=[(γ, L), ...]``. qutip takes
# ``c_ops=[√γ · L, ...]``. Identical physics, different API surface.
solver = LindbladTTN(
    H0=0.5 * OMEGA * Z, f=None, V=None,
    L_ops=[(GAMMA, sm)],
    n_sites=1, bond_dim=4, topology="train", strategy="ps1",
)
rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
res = solver.run(
    rho0=rho0, t_span=(0.0, T_FINAL), dt=0.01,
    observables=[Z, X, Y], save_every=1, verbose=False,
)

# ---- qutip.mesolve --------------------------------------------------------
# Build the shared time grid by prepending t=0 to LindbladTTN's saved times.
# qutip treats tlist[0] as the time of ρ₀, so this also serves as its
# initial-condition anchor. We use the same anchor for the TTN side too,
# computing ⟨O⟩(t=0) directly from ρ₀ so both curves start at the true
# initial values (without it, plotting starts at the first SAVED step,
# t = dt·save_every, where the state has already evolved).
times = np.concatenate([[0.0], res.times])
qres = qt.mesolve(
    qt.Qobj(0.5 * OMEGA * Z), qt.Qobj(rho0), times,
    c_ops=[np.sqrt(GAMMA) * qt.Qobj(sm)],
    e_ops=[qt.Qobj(Z), qt.Qobj(X), qt.Qobj(Y)],
)

# ---- Report + plot --------------------------------------------------------
names = ("σz", "σx", "σy")
operators = (Z, X, Y)
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, (name, op, color) in enumerate(zip(names, operators, ("C0", "C1", "C2"))):
    ttn_initial = float(np.trace(op @ rho0).real)
    ttn = np.concatenate([[ttn_initial], res.expect[i].real])
    qu  = np.asarray(qres.expect[i]).real
    print(f"  ⟨{name}⟩:  max|TTN − qutip| = {np.max(np.abs(ttn - qu)):.2e}")
    ax.plot(times, qu,  color=color, lw=2.0, label=f"qutip ⟨{name}⟩")
    ax.plot(times, ttn, color=color, lw=1.0, ls="--", label=f"TTN ⟨{name}⟩")

ax.set_xlabel("t");  ax.set_ylabel("expectation")
ax.set_title("Single-qubit decay — LindbladTTN vs qutip.mesolve")
ax.legend(loc="best", fontsize=9);  ax.grid(alpha=0.3)
out = _HERE / "qutip_simple.png"
fig.tight_layout();  fig.savefig(out, dpi=130)
print(f"\nSaved figure to {out}")
