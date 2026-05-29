# coding: utf-8
"""Reproduce qutip examples with LindbladTTN.

Two side-by-side comparisons of ``LindbladTTN.run`` against ``qutip.mesolve``:

  Part A — single qubit decay on σz: H = ω/2 σz, L = σ−. PS1 strategy.
           Matches qutip to ~1e-7 on every ⟨σ_z,x,y⟩ trajectory point.

  Part B — 2-qubit independent decay at two different rates, ρ₀ = |00⟩⟨00|.
           VMF strategy. Matches qutip to ~1e-6 on ⟨ZI⟩, ⟨IZ⟩, ⟨ZZ⟩ and on
           the final density matrix. (See the comment block above part_b
           for why we don't reproduce the driven Bell-gate example here.)

Convention note
---------------
qutip expects collapse operators in the form ``c = sqrt(γ) L`` (rate folded
into the operator), while LindbladTTN takes ``L_ops=[(γ, L), ...]`` (rate
kept separate). The physics is identical — only the API surface differs.

qutip.mesolve takes ``tlist[0]`` as the time of ρ₀, so we prepend t=0 to
the qutip time grid and drop the first sample. LindbladTTN's saved times
start at the first SAVED step, not at t=0.

Run
---
    py examples/qutip_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))   # allow running without `pip install -e .`

import numpy as np

try:
    import qutip as qt
except ImportError:
    print("qutip is not installed — `pip install qutip` to run this demo.")
    sys.exit(0)

import matplotlib.pyplot as plt

from lindblad_ttn import LindbladTTN

# Pauli matrices, numpy form (LindbladTTN's API)
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1],  [1, 0]],  dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0],  [0, -1]], dtype=complex)
sm = np.array([[0, 0],  [1, 0]],  dtype=complex)   # |1⟩⟨0| (lowering)


def kron(*ops: np.ndarray) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def max_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# =============================================================================
# Part A — single qubit decay
# =============================================================================

def part_a_single_qubit() -> tuple[np.ndarray, dict, dict]:
    print("\n" + "=" * 68)
    print("Part A — single-qubit decay: H = ω/2 σz,  L = σ−")
    print("=" * 68)

    omega = 1.0
    gamma = 0.3
    t_final = 15.0
    dt = 0.01
    save_every = 10

    # --- LindbladTTN -----------------------------------------------------
    solver = LindbladTTN(
        H0=0.5 * omega * Z,
        f=None, V=None,
        L_ops=[(gamma, sm)],
        n_sites=1, bond_dim=4,
        topology="train", device="cpu", strategy="ps1",
    )
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)  # |0⟩⟨0|
    res = solver.run(
        rho0=rho0, t_span=(0.0, t_final), dt=dt,
        observables=[Z, X, Y], save_every=save_every, verbose=False,
    )
    times = res.times
    ttn = {"sz": res.expect[0].real, "sx": res.expect[1].real, "sy": res.expect[2].real}

    # --- qutip.mesolve ---------------------------------------------------
    # NOTE on conventions: this package's ``sm = [[0,0],[1,0]]`` corresponds
    # to qutip's ``sigmap()`` (it acts as |0⟩⟨1|.dag()). To compare apples to
    # apples we wrap the same numpy matrices as Qobjs rather than using
    # ``qt.sigmam()`` etc., guaranteeing identical operators on both sides.
    # qutip.mesolve takes tlist[0] as the time of rho0, so we prepend t=0
    # and drop the first sample below to align with res.times.
    tlist_q = np.concatenate([[0.0], times])
    H_q   = qt.Qobj(0.5 * omega * Z)
    c_ops = [np.sqrt(gamma) * qt.Qobj(sm)]    # rate folded into operator
    rho0_q = qt.Qobj(rho0)
    e_ops = [qt.Qobj(Z), qt.Qobj(X), qt.Qobj(Y)]
    qres  = qt.mesolve(H_q, rho0_q, tlist_q, c_ops=c_ops, e_ops=e_ops)
    qu = {"sz": np.asarray(qres.expect[0][1:]),
          "sx": np.asarray(qres.expect[1][1:]),
          "sy": np.asarray(qres.expect[2][1:])}

    for k in ("sz", "sx", "sy"):
        print(f"  <{k}>(t):  max|TTN − qutip| = {max_err(ttn[k], qu[k]):.2e}")
    return times, ttn, qu


# =============================================================================
# Part B — two-qubit independent decay (VMF strategy)
# =============================================================================
#
# Why this configuration?
#
# We chose this case after probing the package's multi-qubit propagators
# against qutip during this work. Two pre-existing defects surfaced:
#
#   • PS1 has a 2× rate bug for ≥2 qubits — the backward bond-evolution
#     step of the projector-splitting Strang sweep is missing, so the
#     effective Hamiltonian is applied twice and the system evolves at
#     twice the physical rate. ⇒ use VMF for multi-qubit.
#   • VMF holds the bond dimension at the rank produced by
#     ``_initialize_model`` and cannot grow it. Non-driven observables of
#     non-trivial Hamiltonian dynamics drift from qutip when ρ₀ is rank-
#     deficient *and* the Hamiltonian couples sites. For pure-dissipation
#     dynamics on independent qubits (this case), VMF stays exact: each
#     site's reduced state evolves in its own subspace, so a rank-1 product
#     initial state is sufficient.
#
# Result: two independent decay rates (0.3 on qubit 0, 0.5 on qubit 1)
# starting from |00⟩⟨00|. ⟨ZI⟩ and ⟨IZ⟩ decay independently; ⟨ZZ⟩ tracks
# their product. Matches qutip to ~1e-6 across the run.
# =============================================================================

def part_b_two_qubit_decay() -> tuple[np.ndarray, dict, dict]:
    print("\n" + "=" * 68)
    print("Part B — 2-qubit independent decay (VMF, two rates, |00⟩⟨00|)")
    print("=" * 68)

    gamma0 = 0.3
    gamma1 = 0.5
    t_final = 6.0
    dt = 0.05
    save_every = 4

    L0 = kron(sm, I2)    # decay on qubit 0 at rate gamma0
    L1 = kron(I2, sm)    # decay on qubit 1 at rate gamma1
    ZI = kron(Z, I2);  IZ = kron(I2, Z)
    ZZ = kron(Z, Z)

    # --- LindbladTTN -----------------------------------------------------
    # VMF integrates all node EOMs simultaneously via torchdiffeq dopri5.
    # See comment block above for why we don't use PS1 or the driven Bell
    # example from the package's existing ``two_qubit_bell_state.py``.
    solver = LindbladTTN(
        H0=None, f=None, V=None,
        L_ops=[(gamma0, L0), (gamma1, L1)],
        n_sites=2, bond_dim=8,
        topology="train", device="cpu", strategy="vmf", vmf_atol=1e-10,
    )
    rho0 = np.zeros((4, 4), dtype=complex);  rho0[0, 0] = 1.0  # |00⟩⟨00|
    res = solver.run(
        rho0=rho0, t_span=(0.0, t_final), dt=dt,
        observables=[ZI, IZ, ZZ], save_every=save_every, verbose=False,
    )
    times = res.times
    ttn = {"ZI": res.expect[0].real, "IZ": res.expect[1].real,
           "ZZ": res.expect[2].real}
    rho_ttn = res.rho_final

    # --- qutip.mesolve ---------------------------------------------------
    dims2 = [[2, 2], [2, 2]]
    c_ops = [np.sqrt(gamma0) * qt.Qobj(L0, dims=dims2),
             np.sqrt(gamma1) * qt.Qobj(L1, dims=dims2)]
    rho0_q = qt.Qobj(rho0, dims=dims2)
    e_ops = [qt.Qobj(O, dims=dims2) for O in (ZI, IZ, ZZ)]
    tlist_q = np.concatenate([[0.0], times])
    qres = qt.mesolve(qt.Qobj(np.zeros((4, 4)), dims=dims2), rho0_q, tlist_q,
                      c_ops=c_ops, e_ops=e_ops)
    qu = {"ZI": np.asarray(qres.expect[0][1:]).real,
          "IZ": np.asarray(qres.expect[1][1:]).real,
          "ZZ": np.asarray(qres.expect[2][1:]).real}

    # qutip final density matrix
    qres_final = qt.mesolve(qt.Qobj(np.zeros((4, 4)), dims=dims2),
                            rho0_q, [0.0, t_final], c_ops=c_ops)
    rho_qt = np.asarray(qres_final.states[-1].full())

    for k in ("ZI", "IZ", "ZZ"):
        print(f"  <{k}>(t):  max|TTN − qutip| = {max_err(ttn[k], qu[k]):.2e}")
    print(f"  final ρ:   max|ρ_TTN − ρ_qutip| = {max_err(rho_ttn, rho_qt):.2e}")
    return times, ttn, qu


# =============================================================================
# Plot
# =============================================================================

def plot(times_a, ttn_a, qu_a, times_b, ttn_b, qu_b) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for key, color in (("sz", "C0"), ("sx", "C1"), ("sy", "C2")):
        ax.plot(times_a, qu_a[key],  color=color, lw=2.0,
                label=f"qutip ⟨{key}⟩")
        ax.plot(times_a, ttn_a[key], color=color, lw=1.0, ls="--",
                label=f"TTN  ⟨{key}⟩")
    ax.set_xlabel("t")
    ax.set_ylabel("expectation")
    ax.set_title("Part A — single qubit decay")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for key, color in (("ZI", "C0"), ("IZ", "C1"), ("ZZ", "C2")):
        ax.plot(times_b, qu_b[key],  color=color, lw=2.0,
                label=f"qutip ⟨{key}⟩")
        ax.plot(times_b, ttn_b[key], color=color, lw=1.0, ls="--",
                label=f"TTN  ⟨{key}⟩")
    ax.set_xlabel("t")
    ax.set_ylabel("expectation")
    ax.set_title("Part B — 2-qubit independent decay")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = _HERE / "qutip_parity.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved figure to {out}")


def main() -> None:
    times_a, ttn_a, qu_a = part_a_single_qubit()
    times_b, ttn_b, qu_b = part_b_two_qubit_decay()
    plot(times_a, ttn_a, qu_a, times_b, ttn_b, qu_b)


if __name__ == "__main__":
    main()
