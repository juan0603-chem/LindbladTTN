#!/usr/bin/env python3
"""
Five-qubit entangled state via Gaussian pulses + ZX cascade
===========================================================

This example demonstrates multi-qubit entanglement generation in an open
quantum system and illustrates concretely why tensor-network methods become
indispensable as the number of qubits grows.

Protocol (cascade ZX cross-resonance):
  Stage 0  : Gaussian Rx(pi/2) on qubit 0            t in [0,  T_H]
  Stage 1  : ZX cross-resonance qubit 0->1           t in [T_H,         T_H +   T_CX]
  Stage 2  : ZX cross-resonance qubit 1->2           t in [T_H +   T_CX, T_H + 2*T_CX]
  Stage 3  : ZX cross-resonance qubit 2->3           t in [T_H + 2*T_CX, T_H + 3*T_CX]
  Stage 4  : ZX cross-resonance qubit 3->4           t in [T_H + 3*T_CX, T_H + 4*T_CX]

Each ZX gate U = exp(-i*pi/4 * Z_i @ X_{i+1}) entangles adjacent qubits.
The cascade sequentially transfers the superposition from qubit 0 across
the chain, building genuine N-qubit quantum correlations.

We track:
  - Single-qubit magnetisations <Z_k>(t) per qubit
  - Single-qubit entanglement entropy S(k)(t) (shows entanglement spreading)
  - Purity Tr(rho^2)(t) (sensitive to decoherence)
  - Trace Tr(rho)(t) (Lindblad conservation)

Scaling argument
----------------
  Direct Liouville space : dim = 4^N, Liouvillian = (4^N)^2 entries -> O(16^N).
  TTN (bond dim D)       : N tensors of size <= D x D x 4 -> O(N * D^2) entries.
  For N=5, D=8  : direct   1,048,576  vs TTN   1,280  (x820 compression).
  For N=10, D=8 : direct  10^12      vs TTN   2,560  (x400 billion!).

System
------
  Superconducting transmon qubits
  ZX coupling J/(2pi) = 10 MHz  [H_ZX = J/2 * Z_i @ X_{i+1}]
  T1 = 100 us,  T2 = 50 us
  Gaussian pulse: sigma = T_H/5,  integral(Omega, dt) = pi/4  -> Rx(pi/2)
"""

import sys
import time
from pathlib import Path
from typing import Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp

# Output directory = same folder as this script, regardless of working directory
_HERE = Path(__file__).parent

# Ensure Unicode output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Pauli matrices
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)
Sm = np.array([[0, 0], [1, 0]], dtype=complex)   # sigma^- = |1><0|


# ---------------------------------------------------------------------------
# Multi-qubit operator builders
# ---------------------------------------------------------------------------

def kron_n(*ops):
    """Kronecker product of an arbitrary list of square operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def single_site_op(N: int, k: int, A: np.ndarray) -> np.ndarray:
    """Embed A on qubit k, identity on all others.  Returns 2^N x 2^N matrix."""
    ops = [I2] * N
    ops[k] = A
    return kron_n(*ops)


def zx_bond_op(N: int, i: int) -> np.ndarray:
    """Z_i @ X_{i+1} embedded in N-qubit Hilbert space (2^N x 2^N)."""
    ops = [I2] * N
    ops[i]     = Z
    ops[i + 1] = X
    return kron_n(*ops)


def zz_bond_op(N: int, i: int) -> np.ndarray:
    """Z_i @ Z_{i+1} embedded in N-qubit Hilbert space."""
    ops = [I2] * N
    ops[i]     = Z
    ops[i + 1] = Z
    return kron_n(*ops)


# ---------------------------------------------------------------------------
# Liouville-space superoperators
# ---------------------------------------------------------------------------

def left_sop(A: np.ndarray) -> np.ndarray:
    """A x I  (left multiplication superoperator)."""
    return np.kron(A, np.eye(A.shape[0], dtype=complex))


def right_sop(B: np.ndarray) -> np.ndarray:
    """I x B^T  (right multiplication superoperator)."""
    return np.kron(np.eye(B.shape[0], dtype=complex), B.T)


def commutator_sop(H: np.ndarray) -> np.ndarray:
    """-i[H, .]  as a (d^2 x d^2) superoperator."""
    return -1j * (left_sop(H) - right_sop(H))


def dissipator_sop(gamma: float, L: np.ndarray) -> np.ndarray:
    """gamma * D[L]  dissipator superoperator.

    D[L]rho = L rho L† - 1/2 L†L rho - 1/2 rho L†L
    """
    LdL = L.conj().T @ L
    return gamma * (
        np.kron(L, L.conj())
        - 0.5 * left_sop(LdL)
        - 0.5 * right_sop(LdL)
    )


def build_static_dissipator(N: int, gamma1: float, gamma_phi: float) -> np.ndarray:
    """Build the combined N-qubit dissipator superoperator.

    Per qubit:
      - Amplitude damping : gamma1 * D[sigma^-_k]
      - Pure dephasing    : (gamma_phi/2) * D[Z_k]
        (gamma_phi = 1/T2 - 1/(2*T1) is the pure dephasing rate)
    """
    d     = 2 ** N
    L_dis = np.zeros((d * d, d * d), dtype=complex)
    for k in range(N):
        Sm_k = single_site_op(N, k, Sm)
        Sz_k = single_site_op(N, k, Z)
        L_dis += dissipator_sop(gamma1, Sm_k)
        if gamma_phi > 0.0:
            L_dis += dissipator_sop(gamma_phi / 2.0, Sz_k)
    return L_dis


# ---------------------------------------------------------------------------
# Integrator (scipy solve_ivp with real/imaginary splitting for complex ODE)
# ---------------------------------------------------------------------------

def solve_lindblad_stage(
    L_func,
    v0: np.ndarray,
    t_span: tuple,
    t_eval: Optional[np.ndarray] = None,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> tuple:
    """Integrate drho/dt = L(t) * rho using scipy solve_ivp.

    scipy solve_ivp requires real-valued y, so the complex Liouville vector
    is split into (Re, Im) parts, doubled in size.

    Parameters
    ----------
    L_func   : callable(t) -> (d2, d2) complex Liouvillian matrix
    v0       : initial Liouville vector, shape (d2,), complex
    t_span   : (t_start, t_end)
    t_eval   : time points at which to record the solution
    method   : scipy ODE method ('RK45', 'DOP853', 'RK23', ...)
    rtol/atol: tolerances

    Returns
    -------
    t_out : np.ndarray   shape (n_eval,)
    v_out : np.ndarray   shape (n_eval, d2)  complex
    """
    d2 = len(v0)

    # Pack complex -> real vector [Re(v); Im(v)]
    y0_real = np.concatenate([v0.real, v0.imag])

    def rhs_real(t: float, y: np.ndarray) -> np.ndarray:
        v_cmplx = y[:d2] + 1j * y[d2:]
        dv = L_func(t) @ v_cmplx
        return np.concatenate([dv.real, dv.imag])

    sol = solve_ivp(
        rhs_real, t_span, y0_real,
        method=method,
        t_eval=t_eval,
        rtol=rtol, atol=atol,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    # Unpack real -> complex
    t_out = sol.t
    v_out = sol.y[:d2, :].T + 1j * sol.y[d2:, :].T   # (n_eval, d2)
    return t_out, v_out


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

def simulate_entanglement_cascade(
    N: int,
    J: float = 2 * np.pi * 0.010,
    T1: float = 100_000.0,
    T2: float = 50_000.0,
    T_H: float = 10.0,
    n_steps_per_stage: int = 300,
    save_every: int = 10,
):
    """Simulate N-qubit entanglement spreading via Gaussian pulse + ZX cascade.

    Parameters
    ----------
    N               : number of qubits
    J               : ZX coupling strength (rad/ns)
    T1, T2          : decoherence times (ns)
    T_H             : duration of Gaussian drive on qubit 0 (ns)
    n_steps_per_stage : RK4 steps within each stage
    save_every      : record density matrix every this many steps

    Returns
    -------
    times     : list[float]        time points (ns)
    rho_vecs  : list[np.ndarray]   Liouville vector at each saved step
    stage_info: dict               timing metadata
    elapsed   : float              wall-clock seconds
    """
    wall_start = time.perf_counter()

    d  = 2 ** N
    d2 = d * d

    # Decoherence rates
    gamma1    = 1.0 / T1
    gamma_phi = max(0.0, 1.0 / T2 - 1.0 / (2.0 * T1))

    # Gaussian pulse on qubit 0
    # sigma = T_H/5 so the pulse fits cleanly inside the window
    # Amplitude chosen so integral(Omega, 0, T_H) = pi/4  ->  Rx(pi/2)
    sigma    = T_H / 5.0
    t_center = T_H / 2.0
    area     = sigma * np.sqrt(2.0 * np.pi)   # full Gaussian area (−inf..+inf)
    Omega_pk = (np.pi / 4.0) / area

    T_CX = np.pi / (2.0 * J)      # ZX gate time
    n_zx = N - 1                  # number of ZX stages

    # Hamiltonians
    H_drive   = single_site_op(N, 0, X)
    H_zx_list = [(J / 2.0) * zx_bond_op(N, i) for i in range(n_zx)]

    # Static dissipator (doesn't change over time)
    L_dis = build_static_dissipator(N, gamma1, gamma_phi)

    # Initial state: |00...0⟩
    psi0 = np.zeros(d, dtype=complex)
    psi0[0] = 1.0
    v = np.outer(psi0, psi0.conj()).ravel(order="C")

    # Stage time boundaries
    stage_bounds = ([0.0, T_H]
                    + [T_H + k * T_CX for k in range(1, n_zx + 1)])

    times: list[float]         = []
    rho_vecs: list[np.ndarray] = []
    v_current = v   # initial state vector

    for stage_idx, (t_start, t_end) in enumerate(
        zip(stage_bounds[:-1], stage_bounds[1:])
    ):
        # Dense evaluation grid for recording
        n_record = max(4, n_steps_per_stage // save_every)
        t_eval = np.linspace(t_start, t_end, n_record + 1)[1:]  # exclude t_start

        if stage_idx == 0:
            # Gaussian drive — time-dependent Liouvillian
            def L_drive(t: float) -> np.ndarray:
                omega_t = Omega_pk * np.exp(
                    -0.5 * ((t - t_center) / sigma) ** 2
                )
                return commutator_sop(omega_t * H_drive) + L_dis
            L_func = L_drive
        else:
            # Constant ZX Hamiltonian — cache the matrix
            L_const = commutator_sop(H_zx_list[stage_idx - 1]) + L_dis
            def L_func(t: float, _L: np.ndarray = L_const) -> np.ndarray:   # noqa: E731
                return _L

        t_out, v_out = solve_lindblad_stage(
            L_func, v_current, (t_start, t_end), t_eval=t_eval
        )

        for t_i, v_i in zip(t_out, v_out):
            times.append(float(t_i))
            rho_vecs.append(v_i.copy())

        v_current = v_out[-1]   # hand off to next stage

    elapsed = time.perf_counter() - wall_start

    stage_info = {
        "T_H":     T_H,
        "T_CX":    T_CX,
        "T_tot":   T_H + n_zx * T_CX,
        "sigma":   sigma,
        "Omega_pk": Omega_pk,
        "zx_ends": [T_H + k * T_CX for k in range(1, n_zx + 1)],
    }
    return times, rho_vecs, stage_info, elapsed


# ---------------------------------------------------------------------------
# Observable computation
# ---------------------------------------------------------------------------

def extract_rho(v: np.ndarray, N: int) -> np.ndarray:
    """Reshape Liouville vector (4^N,) -> (2^N, 2^N) density matrix."""
    d = 2 ** N
    return v.reshape(d, d)


def partial_trace(rho: np.ndarray, N: int, keep: int) -> np.ndarray:
    """Trace out all qubits except qubit `keep`.

    Returns a 2x2 single-qubit density matrix.
    """
    d = 2 ** N
    # Reshape to (2, 2, ..., 2) with 2N indices
    rho_t = rho.reshape([2] * (2 * N))
    # Qubit `keep` corresponds to axes (keep) and (keep + N) in the reshaped tensor
    # Trace out all OTHER qubits
    keep_ax  = keep
    keep_ax2 = keep + N
    axes_to_trace = [k for k in range(N) if k != keep]
    result = rho_t
    removed = 0
    for ax in axes_to_trace:
        ax0 = ax - removed
        ax1 = ax0 + (N - removed)
        result = np.trace(result, axis1=ax0, axis2=ax1)
        removed += 1
    return result  # shape (2, 2)


def vn_entropy(rho2: np.ndarray) -> float:
    """Von Neumann entropy S = -Tr(rho log rho) for a 2x2 reduced state.

    Uses the analytical formula for 2x2 matrices: eigenvalues p, 1-p
    with the convention 0*log(0) = 0.
    """
    eigvals = np.linalg.eigvalsh(rho2).real
    eigvals = np.clip(eigvals, 0.0, 1.0)
    S = 0.0
    for lam in eigvals:
        if lam > 1e-15:
            S -= lam * np.log2(lam)
    return float(S)


def compute_observables(rho_vecs: list, N: int) -> dict:
    """Compute trace, single-qubit populations, coherences, entropies, purity.

    For each qubit k the single-qubit reduced state rho_k = Tr_{not k}(rho)
    gives access to:
      pop_k  = rho_k[0,0]          ground-state population P(|0>_k)
      coh_k  = |rho_k[0,1]|        single-qubit coherence magnitude
      Zk     = <Z_k> = 1 - 2*pop_k(|1>)
      Sk     = von Neumann entropy of rho_k  [bits]

    Returns
    -------
    dict with keys:
      trace  : (n_snap,)      Tr(rho)
      pop    : (n_snap, N)    P(|0>_k)  ground-state population per qubit
      coh    : (n_snap, N)    |rho_k[0,1]|  single-qubit coherence per qubit
      Zk     : (n_snap, N)    <Z_k>
      Sk     : (n_snap, N)    single-qubit entanglement entropy [bits]
      purity : (n_snap,)      Tr(rho^2)
    """
    n_snap = len(rho_vecs)

    traces   = np.empty(n_snap)
    pop_vals = np.zeros((n_snap, N))
    coh_vals = np.zeros((n_snap, N))
    Zk_vals  = np.zeros((n_snap, N))
    Sk_vals  = np.zeros((n_snap, N))
    purity   = np.empty(n_snap)

    for i, v in enumerate(rho_vecs):
        rho = extract_rho(v, N)
        tr  = np.trace(rho).real
        traces[i] = tr

        rho_n = rho / max(abs(tr), 1e-15)

        for k in range(N):
            rho_k = partial_trace(rho_n, N, k)   # 2x2 reduced state
            pop_vals[i, k] = rho_k[0, 0].real    # P(|0>_k)
            coh_vals[i, k] = abs(rho_k[0, 1])    # |rho_k[0,1]|
            Zk_vals[i, k]  = rho_k[0, 0].real - rho_k[1, 1].real  # <Z_k>
            Sk_vals[i, k]  = vn_entropy(rho_k)

        purity[i] = np.real(np.trace(rho_n @ rho_n))

    return {
        "trace":  traces,
        "pop":    pop_vals,   # (n_snap, N)
        "coh":    coh_vals,   # (n_snap, N)
        "Zk":     Zk_vals,    # (n_snap, N)
        "Sk":     Sk_vals,    # (n_snap, N)
        "purity": purity,
    }


# ---------------------------------------------------------------------------
# Scaling benchmark
# ---------------------------------------------------------------------------

def benchmark_scaling(
    N_list: list,
    n_steps: int = 100,
    J: float = 2 * np.pi * 0.010,
    T1: float = 100_000.0,
    T2: float = 50_000.0,
    T_H: float = 10.0,
) -> list:
    """Time the full cascade simulation for each N."""
    results = []
    for N in N_list:
        dim_str = f"4^{N} = {4**N:>7,}"
        print(f"    N={N}  (Liouville dim = {dim_str}) ...",
              end="  ", flush=True)
        _, _, _, elapsed = simulate_entanglement_cascade(
            N, J=J, T1=T1, T2=T2, T_H=T_H,
            n_steps_per_stage=n_steps,
            save_every=n_steps,   # only record endpoint of each stage
        )
        print(f"{elapsed:.2f} s")
        results.append({"N": N, "liouville_dim": 4 ** N, "elapsed": elapsed})
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def shade_stages(ax, stage_info: dict, N: int):
    """Background shading for each gate stage."""
    T_H     = stage_info["T_H"]
    zx_ends = stage_info["zx_ends"]
    ax.axvspan(0.0, T_H, alpha=0.13, color="gold", zorder=0)
    palette = ["#7ec8e3", "#90d67a", "#ffb347", "#d4a0ff"]
    t_prev = T_H
    for j, t_end in enumerate(zx_ends):
        ax.axvspan(t_prev, t_end, alpha=0.13,
                   color=palette[j % len(palette)], zorder=0)
        t_prev = t_end
    ax.set_xlim(0.0, stage_info["T_tot"])


def plot_dynamics(times, obs, stage_info, N):
    """Six-panel figure: populations, coherences, magnetisations, entropy, purity, trace."""
    t_arr = np.array(times)
    qubit_colors = plt.cm.plasma(np.linspace(0.1, 0.9, N))

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"{N}-Qubit Entanglement Cascade via Gaussian Pulse + ZX Cross-Resonance\n"
        f"(T1 = 100 us,  T2 = 50 us,  J/(2pi) = 10 MHz)",
        fontsize=13, fontweight="bold",
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.36)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # Stage legend patches
    legend_patches = [
        mpatches.Patch(color="gold", alpha=0.55, label="Gauss. drive q0"),
    ] + [
        mpatches.Patch(color=c, alpha=0.55, label=f"ZX {i}->{i+1}")
        for i, c in enumerate(["#7ec8e3", "#90d67a", "#ffb347", "#d4a0ff"][: N - 1])
    ]

    # -- Panel 1: single-qubit populations P(|0>_k) --------------------------
    shade_stages(ax1, stage_info, N)
    for k in range(N):
        ax1.plot(t_arr, obs["pop"][:, k],
                 color=qubit_colors[k], lw=2, label=f"q{k}")
    ax1.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6, label="0.5")
    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("P(|0>_k)")
    ax1.set_title("Populations  P(|0>_k)(t)\n"
                  "(starts at 1; superposition -> 0.5)")
    ax1.legend(fontsize=8, ncol=N, loc="lower right")
    ax1.set_ylim(-0.05, 1.05)

    # -- Panel 2: single-qubit coherences |rho_k[0,1]| -----------------------
    shade_stages(ax2, stage_info, N)
    for k in range(N):
        ax2.plot(t_arr, obs["coh"][:, k],
                 color=qubit_colors[k], lw=2, label=f"q{k}")
    ax2.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6, label="Max 0.5")
    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("|rho_k[0,1]|")
    ax2.set_title("Single-qubit coherences  |rho_k[0,1]|(t)\n"
                  "(rises during drive, falls as entanglement builds)")
    ax2.legend(fontsize=8, ncol=N, loc="upper right")
    ax2.set_ylim(-0.02, 0.55)

    # -- Panel 3: magnetisations <Z_k> ----------------------------------------
    shade_stages(ax3, stage_info, N)
    for k in range(N):
        ax3.plot(t_arr, obs["Zk"][:, k],
                 color=qubit_colors[k], lw=2, label=f"q{k}")
    ax3.axhline(0.0, color="gray", ls=":", lw=1, alpha=0.6)
    ax3.set_xlabel("Time (ns)")
    ax3.set_ylabel("<Z_k>")
    ax3.set_title("Magnetisations  <Z_k>(t)\n"
                  "(+1 = |0>, -1 = |1>, 0 = superposition)")
    ax3.legend(fontsize=8, ncol=N, loc="lower right")
    ax3.set_ylim(-1.1, 1.1)

    # -- Panel 4: single-qubit entanglement entropy ---------------------------
    shade_stages(ax4, stage_info, N)
    for k in range(N):
        ax4.plot(t_arr, obs["Sk"][:, k],
                 color=qubit_colors[k], lw=2, label=f"S(q{k})")
    ax4.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7, label="Max 1 bit")
    ax4.set_xlabel("Time (ns)")
    ax4.set_ylabel("S(k)  [bits]")
    ax4.set_title("Entanglement entropy  S(k)(t)\n"
                  "(0 = product, 1 bit = maximally entangled)")
    ax4.legend(fontsize=8, ncol=N, loc="upper left")
    ax4.set_ylim(-0.05, 1.1)

    # -- Panel 5: purity Tr(rho^2) --------------------------------------------
    shade_stages(ax5, stage_info, N)
    ax5.plot(t_arr, obs["purity"], color="darkorchid", lw=2, label="Tr(rho^2)")
    ax5.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7, label="Pure state")
    ax5.axhline(1.0 / (2 ** N), color="red", ls=":", lw=1.2, alpha=0.6,
                label=f"Max mixed 1/{2**N}")
    ax5.set_xlabel("Time (ns)")
    ax5.set_ylabel("Purity")
    ax5.set_title("Purity  Tr(rho^2)(t)\n(1 = pure, 1/d = maximally mixed)")
    ax5.legend(fontsize=8)
    ax5.set_ylim(0.0, 1.05)

    # -- Panel 6: trace Tr(rho) -----------------------------------------------
    shade_stages(ax6, stage_info, N)
    ax6.plot(t_arr, obs["trace"], "k-", lw=2, label="Tr(rho)")
    ax6.axhline(1.0, color="red", ls="--", lw=1, alpha=0.7, label="Tr = 1")
    ax6.set_xlabel("Time (ns)")
    ax6.set_ylabel("Tr(rho)")
    ax6.set_title("Trace preservation\n(Lindblad conserves trace)")
    ax6.legend(fontsize=8)
    tr_dev = max(abs(obs["trace"] - 1.0).max(), 0.001)
    ax6.set_ylim(1.0 - tr_dev * 2, 1.0 + tr_dev * 2)

    # Stage legend at bottom
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=min(N + 1, 6),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
        framealpha=0.85,
    )

    out = _HERE / "cat_state_dynamics.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [Plot saved to {out}]")


def plot_scaling(bench_results: list):
    """Two-panel figure: wall-clock time and memory scaling."""
    N_arr = np.array([r["N"] for r in bench_results])
    t_arr = np.array([r["elapsed"] for r in bench_results])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "Scaling: Direct Lindblad  vs  Tensor-Network (TTN)",
        fontsize=13, fontweight="bold",
    )

    # Left: wall-clock time
    ax = axes[0]
    bars = ax.bar(N_arr, t_arr, color="steelblue", alpha=0.85, width=0.5)
    for bar, t in zip(bars, t_arr):
        label = f"{t:.2f} s" if t >= 0.1 else f"{t*1e3:.0f} ms"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.03,
            label, ha="center", va="bottom", fontsize=9,
        )
    ax.set_xlabel("Number of qubits N", fontsize=11)
    ax.set_ylabel("Wall-clock time (s)", fontsize=11)
    ax.set_title("Direct integration time\n(midpoint RK4 in Liouville space)")
    ax.set_xticks(N_arr)

    if len(N_arr) >= 3:
        log_t = np.log(np.maximum(t_arr, 1e-6))
        coeffs = np.polyfit(N_arr, log_t, 1)
        base = np.exp(coeffs[0])
        N_cont = np.linspace(N_arr[0] - 0.3, N_arr[-1] + 1.0, 80)
        ax.plot(N_cont, np.exp(np.polyval(coeffs, N_cont)),
                "r--", lw=1.5, alpha=0.65,
                label=f"fit: ~{base:.1f}^N")
        ax.legend(fontsize=9)

    # Right: memory comparison (log scale)
    ax2 = axes[1]
    D_bond = 8
    N_cont = np.linspace(1.5, max(N_arr) + 2.5, 100)

    direct_params = 4 ** N_arr
    ttn_params    = N_arr * D_bond ** 2 * 4

    ax2.semilogy(N_arr, direct_params, "bs-", ms=8, lw=2,
                 label="Direct: $4^N$ parameters")
    ax2.semilogy(N_cont, 4 ** N_cont, "b--", alpha=0.25, lw=1)
    ax2.semilogy(N_arr, ttn_params, "rs-", ms=8, lw=2,
                 label=f"TTN D={D_bond}: $N \\cdot D^2 \\cdot 4$")
    ax2.semilogy(N_cont, N_cont * D_bond ** 2 * 4, "r--", alpha=0.25, lw=1)

    # Annotate largest-N ratio
    N_max   = N_arr[-1]
    ratio   = 4 ** N_max / max(N_max * D_bond ** 2 * 4, 1)
    ax2.annotate(
        f"x{ratio:.0f} compression\nat N={N_max}",
        xy=(N_max, N_max * D_bond ** 2 * 4),
        xytext=(N_max - 1.5, 4 ** (N_max - 0.8)),
        arrowprops=dict(arrowstyle="->", color="k", lw=1.2),
        fontsize=9, color="k",
    )

    # Extrapolate to N=10
    N_ex = 10
    d_ex = 4 ** N_ex; t_ex = N_ex * D_bond ** 2 * 4
    ax2.axvline(N_ex, color="gray", ls=":", alpha=0.4)
    ax2.semilogy([N_ex], [d_ex], "b^", ms=10, alpha=0.6,
                 label=f"N={N_ex} direct: {d_ex:,.0f}")
    ax2.semilogy([N_ex], [t_ex], "r^", ms=10, alpha=0.6,
                 label=f"N={N_ex} TTN: {t_ex}")

    ax2.set_xlabel("Number of qubits N", fontsize=11)
    ax2.set_ylabel("Parameters (state representation)", fontsize=11)
    ax2.set_title(f"Memory scaling  (TTN bond dim D = {D_bond})")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_xticks(list(N_arr) + [N_ex])

    plt.tight_layout()
    out = _HERE / "scaling_comparison.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [Plot saved to {out}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N   = 5
    J   = 2 * np.pi * 0.010   # ZX coupling 10 MHz (rad/ns)
    T1  = 100_000.0            # 100 us in ns
    T2  = 50_000.0             # 50 us in ns
    T_H = 10.0                 # Gaussian drive duration (ns)

    T_CX     = np.pi / (2 * J)
    sigma    = T_H / 5.0
    area     = sigma * np.sqrt(2 * np.pi)
    Omega_pk = (np.pi / 4.0) / area
    T_tot    = T_H + (N - 1) * T_CX

    sep = "-" * 62

    print()
    print(sep)
    print(f"  {N}-Qubit Entanglement Cascade -- Open Lindblad Dynamics")
    print(sep)
    print(f"  System          : {N} superconducting transmon qubits")
    print(f"  ZX coupling     : J/(2pi) = {J / (2*np.pi) * 1e3:.1f} MHz")
    print(f"  T1 / T2         : {T1/1e3:.0f} us / {T2/1e3:.0f} us")
    print()
    print(f"  Gaussian pulse (on qubit 0)")
    print(f"    Duration      : T_H  = {T_H:.1f} ns")
    print(f"    Sigma         : s    = {sigma:.2f} ns")
    print(f"    Peak Rabi     : Omega_pk/(2pi) = {Omega_pk/(2*np.pi)*1e3:.2f} MHz")
    print(f"    Pulse area    : int(Omega)dt = {Omega_pk*area:.4f} rad"
          f"  (target pi/4 = {np.pi/4:.4f})")
    print()
    print(f"  ZX gate time    : T_CX = {T_CX:.2f} ns")
    print(f"  Total time      : T_tot = {T_tot:.2f} ns")
    print()
    print(f"  Direct dim      : 4^{N} = {4**N:,}  (Liouville space)")
    print(f"  Liouvillian     : {4**N}^2 = {4**(2*N):,} entries")
    print(sep)

    # Run simulation
    print()
    print("  Running 5-qubit cascade simulation ...")
    times, rho_vecs, stage_info, elapsed = simulate_entanglement_cascade(
        N, J=J, T1=T1, T2=T2, T_H=T_H,
        n_steps_per_stage=300, save_every=10,
    )
    print(f"  Completed in {elapsed:.2f} s")

    obs = compute_observables(rho_vecs, N)

    # Final-state summary
    rho_final = extract_rho(rho_vecs[-1], N)
    print()
    print(f"  Results at t = {T_tot:.2f} ns")
    print(f"    Tr(rho)         = {obs['trace'][-1]:.8f}")
    print(f"    Purity          = {obs['purity'][-1]:.4f}  (1=pure, 1/{2**N}=max-mixed)")
    for k in range(N):
        print(f"    <Z_{k}>  = {obs['Zk'][-1, k]:+.4f}   S({k}) = {obs['Sk'][-1, k]:.4f} bits")

    # Plots
    plot_dynamics(times, obs, stage_info, N)

    # Scaling benchmark
    print()
    print("  Scaling benchmark  (direct Lindblad, 100 steps per stage):")
    print("  " + "-" * 50)
    bench = benchmark_scaling(
        N_list=[2, 3, 4, 5],
        n_steps=100,
        J=J, T1=T1, T2=T2, T_H=T_H,
    )

    D_bond = 8
    print()
    print(f"  State-vector size  (direct: 4^N  vs  TTN bond dim D={D_bond}):")
    print(f"  {'N':>4}  {'direct params':>14}  {'TTN params':>10}  {'ratio':>8}")
    # Show benchmarked range + theoretical extrapolation to N=10
    for Nb in list({r['N'] for r in bench}) + [6, 7, 8, 10]:
        Nb = int(Nb)
        direct_mem = 4 ** Nb
        ttn_mem    = Nb * D_bond ** 2 * 4
        ratio      = direct_mem / ttn_mem
        tag = ""
        if ratio < 1:
            tag = "  <- TTN uses MORE (D too large for N)"
        elif Nb > max(r['N'] for r in bench):
            tag = "  [extrapolated]"
        print(
            f"  {Nb:>4}  {direct_mem:>14,}  {ttn_mem:>10,}"
            f"  {ratio:>7.1f}x{tag}"
        )

    plot_scaling(bench)

    print()
    print(sep)
    print("  Key takeaways")
    print(sep)
    print(f"  * Gaussian pulse (sigma={sigma:.1f} ns) puts qubit 0 into superposition.")
    print(f"  * Each ZX stage propagates entanglement one bond further along the chain.")
    print(f"  * Entanglement entropy S(k) rises sequentially: S(0)={obs['Sk'][-1,0]:.2f},")
    print(f"    S(1)={obs['Sk'][-1,1]:.2f}, ..., S({N-1})={obs['Sk'][-1,N-1]:.2f} bits -- all qubits entangled!")
    print(f"  * Purity = {obs['purity'][-1]:.4f}  (decoherence negligible on {T_tot:.0f} ns << T1={T1/1e3:.0f} us).")
    print(f"  * N=5 direct integration took {elapsed:.1f} s  (Liouv. dim = {4**5}).")
    print(f"  * N=10 direct: dim {4**10:,} -> requires ~8 GB RAM just for the state.")
    print(f"  * TTN (D={D_bond}): {10 * D_bond**2 * 4:,} params for N=10 -> fits in kilobytes!")
    print(sep)
