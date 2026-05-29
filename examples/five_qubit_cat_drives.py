#!/usr/bin/env python3
"""
Five-qubit GHZ cat state via global Gaussian drives
====================================================

This example generates a maximally entangled N-qubit cat state using a
purely drive-based protocol — all qubits are driven simultaneously, with no
sequential gate cascade.  It contrasts with ``five_qubit_cat_state.py``
where entanglement is spread qubit-by-qubit.

Protocol
--------
Stage 0  [0,  T_H]:
    Global Gaussian Rx(pi/2) on ALL qubits simultaneously.
    H_drive(t) = Omega(t) * sum_k X_k
    Result: every qubit enters (|0> - i|1>)/sqrt(2)  independently.

Stage 1  [T_H,  T_H + T_CX]:
    Parallel ZX entangling on ALL bonds simultaneously.
    H_ZX = (J/2) * sum_i  Z_i @ X_{i+1}
    All adjacent pairs entangle at the same time.  This creates a
    cluster-like state distinct from the sequential cascade.

Stage 2  [T_H + T_CX,  T_H + T_CX + T_H]:
    Local Gaussian Rx correction drives on qubits 1..N-1.
    H_corr(t) = Omega_c(t) * sum_{k>=1} X_k
    This rotates each target qubit back into the computational basis,
    mapping the cluster state toward (|0..0> + |1..1>)/sqrt(2).

The dynamics are tracked via:
    - Single-qubit populations P(|0>_k)(t)   per qubit
    - Single-qubit coherences  |rho_k[0,1]|(t)  per qubit
    - Entanglement entropy     S(k)(t)         per qubit
    - Global purity Tr(rho^2)(t) and trace Tr(rho)(t)

System
------
    N = 5 superconducting transmon qubits
    ZX coupling J/(2pi) = 10 MHz
    T1 = 100 us,  T2 = 50 us
    Gaussian pulse sigma = T_H / 5
"""

import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp

_HERE = Path(__file__).parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Pauli matrices
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)
Sm = np.array([[0, 0], [1, 0]], dtype=complex)


# ---------------------------------------------------------------------------
# Multi-qubit operator builders  (shared with five_qubit_cat_state.py)
# ---------------------------------------------------------------------------

def kron_n(*ops):
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def single_site_op(N: int, k: int, A: np.ndarray) -> np.ndarray:
    ops = [I2] * N; ops[k] = A
    return kron_n(*ops)


def zx_bond_op(N: int, i: int) -> np.ndarray:
    ops = [I2] * N; ops[i] = Z; ops[i + 1] = X
    return kron_n(*ops)


# ---------------------------------------------------------------------------
# Liouville-space superoperators
# ---------------------------------------------------------------------------

def left_sop(A):
    return np.kron(A, np.eye(A.shape[0], dtype=complex))

def right_sop(B):
    return np.kron(np.eye(B.shape[0], dtype=complex), B.T)

def commutator_sop(H):
    return -1j * (left_sop(H) - right_sop(H))

def dissipator_sop(gamma: float, L: np.ndarray) -> np.ndarray:
    LdL = L.conj().T @ L
    return gamma * (np.kron(L, L.conj())
                    - 0.5 * left_sop(LdL) - 0.5 * right_sop(LdL))

def build_static_dissipator(N: int, gamma1: float, gamma_phi: float) -> np.ndarray:
    d = 2 ** N
    L_dis = np.zeros((d * d, d * d), dtype=complex)
    for k in range(N):
        L_dis += dissipator_sop(gamma1, single_site_op(N, k, Sm))
        if gamma_phi > 0:
            L_dis += dissipator_sop(gamma_phi / 2.0, single_site_op(N, k, Z))
    return L_dis


# ---------------------------------------------------------------------------
# scipy solve_ivp wrapper for complex Liouville ODE
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
    """Integrate drho/dt = L(t)*rho using scipy solve_ivp.

    Complex state vector is split into (Re, Im) for scipy compatibility.
    """
    d2 = len(v0)
    y0 = np.concatenate([v0.real, v0.imag])

    def rhs(t, y):
        v = y[:d2] + 1j * y[d2:]
        dv = L_func(t) @ v
        return np.concatenate([dv.real, dv.imag])

    sol = solve_ivp(rhs, t_span, y0, method=method,
                    t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    t_out = sol.t
    v_out = sol.y[:d2, :].T + 1j * sol.y[d2:, :].T
    return t_out, v_out


# ---------------------------------------------------------------------------
# Gaussian pulse helper
# ---------------------------------------------------------------------------

def gaussian_pulse(t: float, t_center: float, sigma: float, area: float) -> float:
    """Gaussian envelope with specified time-integral area."""
    norm = area / (sigma * np.sqrt(2 * np.pi))
    return norm * np.exp(-0.5 * ((t - t_center) / sigma) ** 2)


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

def simulate_global_drive_cat(
    N: int,
    J: float = 2 * np.pi * 0.010,
    T1: float = 100_000.0,
    T2: float = 50_000.0,
    T_H: float = 10.0,
    n_pts_per_stage: int = 60,
):
    """Simulate N-qubit cat state via global drives + parallel ZX coupling.

    Protocol
    --------
    Stage 0 [0,  T_H]             : Global Gaussian Rx(pi/2) on all qubits
    Stage 1 [T_H, T_H+T_CX]      : Parallel ZX on all bonds (all at once)
    Stage 2 [T_H+T_CX, T_H+T_CX+T_H] : Local Rx correction drives on q1..qN-1

    Returns
    -------
    times, rho_vecs, stage_info, elapsed
    """
    wall_start = time.perf_counter()
    d  = 2 ** N
    d2 = d * d

    gamma1    = 1.0 / T1
    gamma_phi = max(0.0, 1.0 / T2 - 1.0 / (2.0 * T1))

    sigma    = T_H / 5.0
    T_CX     = np.pi / (2.0 * J)
    T_corr   = T_H                        # correction drive duration = same as initial

    # Hamiltonians
    # Global X drive: sum_k X_k  (acts on all qubits simultaneously)
    H_global_X = sum(single_site_op(N, k, X) for k in range(N))

    # Parallel ZX: (J/2) * sum_i  Z_i @ X_{i+1}
    H_zx_parallel = sum((J / 2.0) * zx_bond_op(N, i) for i in range(N - 1))

    # Correction drive: X on qubits 1..N-1 only (not qubit 0)
    H_corr_X = sum(single_site_op(N, k, X) for k in range(1, N))

    # Static dissipator
    L_dis = build_static_dissipator(N, gamma1, gamma_phi)

    # Initial state |00...0>
    psi0 = np.zeros(d, dtype=complex); psi0[0] = 1.0
    v_current = np.outer(psi0, psi0.conj()).ravel(order="C")

    # Stage boundaries
    stage_info = {
        "T_H":     T_H,
        "T_CX":    T_CX,
        "T_corr":  T_corr,
        "T_tot":   T_H + T_CX + T_corr,
        "sigma":   sigma,
        "stage_ends": [T_H, T_H + T_CX, T_H + T_CX + T_corr],
        "stage_labels": [
            "Global Rx(pi/2)",
            "Parallel ZX",
            "Local correction",
        ],
        "stage_colors": ["gold", "#7ec8e3", "#90d67a"],
    }

    times: list = []
    rho_vecs: list = []

    for stage_idx, (t_start, t_end) in enumerate([
        (0.0,             T_H),
        (T_H,             T_H + T_CX),
        (T_H + T_CX,      T_H + T_CX + T_corr),
    ]):
        t_eval = np.linspace(t_start, t_end, n_pts_per_stage + 1)[1:]

        if stage_idx == 0:
            # Stage 0: global Gaussian Rx(pi/2) on all qubits
            # Area pi/4 per qubit — but summing N qubits means each individual
            # qubit sees area pi/4 from the single-site component of H_global_X.
            # Normalize per-qubit: int(Omega) dt = pi/4
            area_per_q = np.pi / 4.0
            t_c = (t_start + t_end) / 2.0
            def L_s0(t: float) -> np.ndarray:
                omega = gaussian_pulse(t, t_c, sigma, area_per_q)
                return commutator_sop(omega * H_global_X) + L_dis
            L_func = L_s0

        elif stage_idx == 1:
            # Stage 1: parallel ZX (time-independent)
            L_zx = commutator_sop(H_zx_parallel) + L_dis
            def L_s1(t: float, _L=L_zx) -> np.ndarray:
                return _L
            L_func = L_s1

        else:
            # Stage 2: correction Rx(pi/2) on qubits 1..N-1
            # These qubits need Rx(-pi/2) to rotate back from Y to Z basis
            # Area = pi/4 per qubit (same magnitude, sign via phase)
            area_corr = np.pi / 4.0
            t_c2 = (t_start + t_end) / 2.0
            def L_s2(t: float) -> np.ndarray:
                omega = gaussian_pulse(t, t_c2, sigma, area_corr)
                return commutator_sop(omega * H_corr_X) + L_dis
            L_func = L_s2

        t_out, v_out = solve_lindblad_stage(L_func, v_current,
                                             (t_start, t_end), t_eval=t_eval)
        for t_i, v_i in zip(t_out, v_out):
            times.append(float(t_i))
            rho_vecs.append(v_i.copy())
        v_current = v_out[-1]

    elapsed = time.perf_counter() - wall_start
    return times, rho_vecs, stage_info, elapsed


# ---------------------------------------------------------------------------
# Observable extraction  (same helpers as five_qubit_cat_state.py)
# ---------------------------------------------------------------------------

def extract_rho(v: np.ndarray, N: int) -> np.ndarray:
    return v.reshape(2 ** N, 2 ** N)


def partial_trace(rho: np.ndarray, N: int, keep: int) -> np.ndarray:
    """2x2 reduced density matrix for qubit `keep`."""
    result = rho.reshape([2] * (2 * N))
    axes_to_trace = [k for k in range(N) if k != keep]
    removed = 0
    for ax in axes_to_trace:
        ax0 = ax - removed
        ax1 = ax0 + (N - removed)
        result = np.trace(result, axis1=ax0, axis2=ax1)
        removed += 1
    return result


def vn_entropy(rho2: np.ndarray) -> float:
    eigvals = np.clip(np.linalg.eigvalsh(rho2).real, 0.0, 1.0)
    return float(-sum(l * np.log2(l) for l in eigvals if l > 1e-15))


def compute_observables(rho_vecs: list, N: int) -> dict:
    """Single-qubit populations, coherences, entanglement entropies, purity."""
    n = len(rho_vecs)
    traces   = np.empty(n)
    pop_vals = np.zeros((n, N))
    coh_vals = np.zeros((n, N))
    Zk_vals  = np.zeros((n, N))
    Sk_vals  = np.zeros((n, N))
    purity   = np.empty(n)

    for i, v in enumerate(rho_vecs):
        rho = extract_rho(v, N)
        tr  = np.trace(rho).real
        traces[i] = tr
        rho_n = rho / max(abs(tr), 1e-15)
        for k in range(N):
            rho_k = partial_trace(rho_n, N, k)
            pop_vals[i, k] = rho_k[0, 0].real
            coh_vals[i, k] = abs(rho_k[0, 1])
            Zk_vals[i, k]  = rho_k[0, 0].real - rho_k[1, 1].real
            Sk_vals[i, k]  = vn_entropy(rho_k)
        purity[i] = np.real(np.trace(rho_n @ rho_n))

    return {"trace": traces, "pop": pop_vals, "coh": coh_vals,
            "Zk": Zk_vals, "Sk": Sk_vals, "purity": purity}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def shade_stages(ax, stage_info: dict, alpha: float = 0.13):
    ends   = stage_info["stage_ends"]
    colors = stage_info["stage_colors"]
    t_prev = 0.0
    for t_end, color in zip(ends, colors):
        ax.axvspan(t_prev, t_end, alpha=alpha, color=color, zorder=0)
        t_prev = t_end
    ax.set_xlim(0.0, stage_info["T_tot"])


def plot_dynamics(times, obs, stage_info, N):
    """Six-panel figure: populations, coherences, magnetisations,
    entanglement entropy, purity, trace."""
    t_arr = np.array(times)
    qc = plt.cm.plasma(np.linspace(0.1, 0.9, N))   # one colour per qubit

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"{N}-Qubit Cat State via Global Gaussian Drives + Parallel ZX Coupling\n"
        f"(T1 = 100 us,  T2 = 50 us,  J/(2pi) = 10 MHz)",
        fontsize=13, fontweight="bold",
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.36)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    ax1, ax2, ax3, ax4, ax5, ax6 = axes

    legend_patches = [
        mpatches.Patch(color=c, alpha=0.55, label=l)
        for c, l in zip(stage_info["stage_colors"], stage_info["stage_labels"])
    ]

    # -- Panel 1: populations P(|0>_k) ----------------------------------------
    shade_stages(ax1, stage_info)
    for k in range(N):
        ax1.plot(t_arr, obs["pop"][:, k], color=qc[k], lw=2, label=f"q{k}")
    ax1.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6)
    ax1.set_xlabel("Time (ns)"); ax1.set_ylabel("P(|0>_k)")
    ax1.set_title("Populations  P(|0>_k)(t)\n"
                  "Global drive pulls all qubits to 0.5 simultaneously")
    ax1.legend(fontsize=8, ncol=N, loc="lower right")
    ax1.set_ylim(-0.05, 1.05)

    # -- Panel 2: coherences |rho_k[0,1]| ------------------------------------
    shade_stages(ax2, stage_info)
    for k in range(N):
        ax2.plot(t_arr, obs["coh"][:, k], color=qc[k], lw=2, label=f"q{k}")
    ax2.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6, label="Max 0.5")
    ax2.set_xlabel("Time (ns)"); ax2.set_ylabel("|rho_k[0,1]|")
    ax2.set_title("Single-qubit coherences  |rho_k[0,1]|(t)\n"
                  "Rise during drive; fall as ZX entanglement destroys local purity")
    ax2.legend(fontsize=8, ncol=N, loc="upper right")
    ax2.set_ylim(-0.02, 0.55)

    # -- Panel 3: magnetisations <Z_k> ----------------------------------------
    shade_stages(ax3, stage_info)
    for k in range(N):
        ax3.plot(t_arr, obs["Zk"][:, k], color=qc[k], lw=2, label=f"q{k}")
    ax3.axhline(0.0, color="gray", ls=":", lw=1, alpha=0.6)
    ax3.set_xlabel("Time (ns)"); ax3.set_ylabel("<Z_k>")
    ax3.set_title("Magnetisations  <Z_k>(t)\n"
                  "(+1 = |0>, -1 = |1>, 0 = superposition)")
    ax3.legend(fontsize=8, ncol=N, loc="lower right")
    ax3.set_ylim(-1.1, 1.1)

    # -- Panel 4: entanglement entropy S(k) -----------------------------------
    shade_stages(ax4, stage_info)
    for k in range(N):
        ax4.plot(t_arr, obs["Sk"][:, k], color=qc[k], lw=2, label=f"S(q{k})")
    ax4.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7, label="Max 1 bit")
    ax4.set_xlabel("Time (ns)"); ax4.set_ylabel("S(k)  [bits]")
    ax4.set_title("Entanglement entropy  S(k)(t)\n"
                  "All qubits entangle simultaneously (not sequentially)")
    ax4.legend(fontsize=8, ncol=N, loc="upper left")
    ax4.set_ylim(-0.05, 1.1)

    # -- Panel 5: purity Tr(rho^2) --------------------------------------------
    shade_stages(ax5, stage_info)
    ax5.plot(t_arr, obs["purity"], color="darkorchid", lw=2, label="Tr(rho^2)")
    ax5.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7, label="Pure")
    ax5.axhline(1.0 / (2 ** N), color="red", ls=":", lw=1.2, alpha=0.6,
                label=f"Max mixed 1/{2**N}")
    ax5.set_xlabel("Time (ns)"); ax5.set_ylabel("Purity")
    ax5.set_title("Purity  Tr(rho^2)(t)")
    ax5.legend(fontsize=8); ax5.set_ylim(0.0, 1.05)

    # -- Panel 6: trace -------------------------------------------------------
    shade_stages(ax6, stage_info)
    ax6.plot(t_arr, obs["trace"], "k-", lw=2, label="Tr(rho)")
    ax6.axhline(1.0, color="red", ls="--", lw=1, alpha=0.7, label="Tr = 1")
    ax6.set_xlabel("Time (ns)"); ax6.set_ylabel("Tr(rho)")
    ax6.set_title("Trace preservation\n(Lindblad exact conservation)")
    ax6.legend(fontsize=8)
    dev = max(abs(obs["trace"] - 1.0).max(), 0.001)
    ax6.set_ylim(1.0 - dev * 2, 1.0 + dev * 2)

    # Stage legend at bottom
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.04), framealpha=0.88)

    out = _HERE / "cat_drives_dynamics.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [Plot saved to {out}]")


def save_density_matrix_txt(
    rho_final: np.ndarray,
    times: list,
    obs: dict,
    stage_info: dict,
    N: int,
    path: str = "outputs/cat_drives_result.txt",
) -> None:
    """Save the full density matrix and time-series observables to a text file.

    Sections
    --------
    1. Run parameters
    2. Density matrix  — Re(rho), Im(rho), |rho| in full or blocked format
    3. Summary statistics  — trace, purity, hermiticity, eigenvalues
    4. Single-qubit reduced states  — populations, coherences, entropies
    5. Time series  — t, Tr(rho), purity, and per-qubit observables
    """
    from pathlib import Path as _Path
    out = _Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    d   = 2 ** N
    t_arr = np.asarray(times)

    # ── helpers ────────────────────────────────────────────────────────────────
    def fmt_matrix(M: np.ndarray, label: str, fh, width: int = 9, prec: int = 6) -> None:
        """Write a d×d matrix with row/column labels."""
        fh.write(f"  {label}\n")
        # column header — split into blocks of 8 if d > 8
        block = 8
        for col_start in range(0, d, block):
            col_end = min(col_start + block, d)
            header = "         " + "".join(f"  [{j:3d}]" for j in range(col_start, col_end))
            fh.write(header + "\n")
            for i in range(d):
                row = f"  [{i:3d}]  " + "".join(
                    f"  {M[i, j]:+{width}.{prec}f}" for j in range(col_start, col_end)
                )
                fh.write(row + "\n")
            fh.write("\n")

    # ── open file ──────────────────────────────────────────────────────────────
    with open(out, "w", encoding="utf-8") as fh:

        # ── Section 1: parameters ──────────────────────────────────────────────
        fh.write("=" * 72 + "\n")
        fh.write("  LindbladTTN — five_qubit_cat_drives result\n")
        fh.write("=" * 72 + "\n\n")
        fh.write(f"  N qubits        : {N}\n")
        fh.write(f"  Hilbert dim     : 2^{N} = {d}\n")
        fh.write(f"  Liouville dim   : 4^{N} = {d**2}\n")
        fh.write(f"  Stages          : {len(stage_info['stage_labels'])}\n")
        for i, (label, t_end) in enumerate(
            zip(stage_info["stage_labels"], stage_info["stage_ends"])
        ):
            t_start = stage_info["stage_ends"][i - 1] if i > 0 else 0.0
            fh.write(f"    {i}: [{t_start:.2f}, {t_end:.2f}] ns  —  {label}\n")
        fh.write(f"  Total time      : {stage_info['T_tot']:.4f} ns\n")
        fh.write(f"  Snapshots saved : {len(times)}\n\n")

        # ── Section 2: density matrix ──────────────────────────────────────────
        fh.write("-" * 72 + "\n")
        fh.write(f"  Density matrix  rho_final  ({d} x {d})\n")
        fh.write("-" * 72 + "\n\n")
        fmt_matrix(rho_final.real, "Re(rho)", fh)
        fmt_matrix(rho_final.imag, "Im(rho)", fh)
        fmt_matrix(np.abs(rho_final), "|rho|  ", fh)

        # ── Section 3: summary statistics ─────────────────────────────────────
        fh.write("-" * 72 + "\n")
        fh.write("  Summary statistics\n")
        fh.write("-" * 72 + "\n")
        tr    = np.trace(rho_final).real
        pur   = np.real(np.trace(rho_final @ rho_final))
        herm  = np.linalg.norm(rho_final - rho_final.conj().T)
        eigv  = np.linalg.eigvalsh(rho_final).real
        p0    = rho_final[0,   0].real
        p1    = rho_final[d-1, d-1].real
        coh01 = abs(rho_final[0, d-1])
        ghz_f = 0.5 * (p0 + p1) + coh01
        fh.write(f"  Tr(rho)              = {tr:.10f}\n")
        fh.write(f"  Purity Tr(rho^2)     = {pur:.10f}\n")
        fh.write(f"  ||rho - rho†||       = {herm:.2e}   (0 = Hermitian)\n")
        fh.write(f"  min eigenvalue       = {eigv.min():.6e}  (>=0 = physical)\n")
        fh.write(f"  max eigenvalue       = {eigv.max():.6f}\n")
        fh.write(f"  P(|{'0'*N}>)          = {p0:.6f}\n")
        fh.write(f"  P(|{'1'*N}>)          = {p1:.6f}\n")
        fh.write(f"  |rho(|0..0>,|1..1>)| = {coh01:.6f}\n")
        fh.write(f"  GHZ fidelity proxy F = {ghz_f:.6f}  (ideal: 1.0)\n\n")

        # ── Section 4: single-qubit reduced states at t_final ─────────────────
        fh.write("-" * 72 + "\n")
        fh.write("  Single-qubit reduced states  rho_k  at t = final\n")
        fh.write("-" * 72 + "\n\n")
        fh.write(f"  {'Qubit':>5}  {'P(|0>)':>9}  {'P(|1>)':>9}"
                 f"  {'|coh|':>9}  {'<Z>':>9}  {'S [bits]':>10}\n")
        fh.write("  " + "-" * 57 + "\n")
        for k in range(N):
            fh.write(
                f"  {'q'+str(k):>5}  {obs['pop'][-1, k]:>9.6f}  "
                f"{1-obs['pop'][-1, k]:>9.6f}  "
                f"{obs['coh'][-1, k]:>9.6f}  "
                f"{obs['Zk'][-1, k]:>+9.6f}  "
                f"{obs['Sk'][-1, k]:>10.6f}\n"
            )
        fh.write("\n")

        # ── Section 5: time series ─────────────────────────────────────────────
        fh.write("-" * 72 + "\n")
        fh.write("  Time series  (one row per saved snapshot)\n")
        fh.write("-" * 72 + "\n")
        # column header
        qubit_cols = "".join(
            f"  P(|0>)_q{k}  |coh|_q{k}   Sk_q{k} " for k in range(N)
        )
        fh.write(f"  {'t [ns]':>10}  {'Tr(rho)':>10}  {'purity':>10}{qubit_cols}\n")
        fh.write("  " + "-" * (32 + 30 * N) + "\n")
        for i, t in enumerate(t_arr):
            qubit_vals = "".join(
                f"  {obs['pop'][i, k]:>10.6f}  {obs['coh'][i, k]:>9.6f}"
                f"  {obs['Sk'][i, k]:>8.6f}"
                for k in range(N)
            )
            fh.write(
                f"  {t:>10.4f}  {obs['trace'][i]:>10.8f}"
                f"  {obs['purity'][i]:>10.8f}{qubit_vals}\n"
            )
        fh.write("\n")

    print(f"  [Density matrix saved to {out}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N   = 5
    J   = 2 * np.pi * 0.010
    T1  = 100_000.0
    T2  = 50_000.0
    T_H = 10.0

    T_CX  = np.pi / (2 * J)
    sigma = T_H / 5.0
    T_tot = T_H + T_CX + T_H   # 3 stages

    sep = "-" * 64

    print()
    print(sep)
    print(f"  {N}-Qubit Cat State via Global Drives + Parallel ZX")
    print(sep)
    print(f"  System          : {N} superconducting transmon qubits")
    print(f"  ZX coupling     : J/(2pi) = {J/(2*np.pi)*1e3:.1f} MHz")
    print(f"  T1 / T2         : {T1/1e3:.0f} us / {T2/1e3:.0f} us")
    print()
    print(f"  Stage 0  [0, {T_H:.1f} ns]"
          f"        : Global Gaussian Rx(pi/2) on ALL {N} qubits")
    print(f"  Stage 1  [{T_H:.1f}, {T_H+T_CX:.1f} ns]"
          f"  : Parallel ZX on ALL {N-1} bonds simultaneously")
    print(f"  Stage 2  [{T_H+T_CX:.1f}, {T_tot:.1f} ns]"
          f"  : Local Rx correction drives on qubits 1..{N-1}")
    print()
    print(f"  Total time      : {T_tot:.2f} ns")
    print(f"  Direct Liouv.   : 4^{N} = {4**N:,}")
    print(sep)

    print()
    print("  Running simulation ...")
    times, rho_vecs, stage_info, elapsed = simulate_global_drive_cat(
        N, J=J, T1=T1, T2=T2, T_H=T_H, n_pts_per_stage=80
    )
    print(f"  Completed in {elapsed:.2f} s  ({len(times)} snapshots)")

    obs = compute_observables(rho_vecs, N)
    rho_final = extract_rho(rho_vecs[-1], N)
    d = 2 ** N

    # GHZ-basis fidelity proxy (maximised over phase)
    p0   = rho_final[0,  0].real
    p1   = rho_final[d-1, d-1].real
    coh  = abs(rho_final[0, d-1])
    ghz_f = 0.5 * (p0 + p1) + coh

    print()
    print(f"  Results at t = {T_tot:.2f} ns")
    print(f"    Tr(rho)         = {obs['trace'][-1]:.8f}")
    print(f"    Purity          = {obs['purity'][-1]:.4f}")
    print()
    print(f"    {'Qubit':>6}  {'P(|0>)':>8}  {'|coh|':>8}  {'<Z>':>8}  {'S [bits]':>10}")
    print("    " + "-" * 46)
    for k in range(N):
        print(f"    {'q'+str(k):>6}  {obs['pop'][-1,k]:>8.4f}  "
              f"{obs['coh'][-1,k]:>8.4f}  {obs['Zk'][-1,k]:>+8.4f}  "
              f"{obs['Sk'][-1,k]:>10.4f}")
    print()
    print(f"    GHZ-basis fidelity proxy F = {ghz_f:.4f}  (ideal: 1.0)")
    print(f"    P(|00000>) = {p0:.4f},  P(|11111>) = {p1:.4f}")
    print(f"    |rho(|00000>,|11111>)| = {coh:.4f}  (ideal cat: 0.5)")

    save_density_matrix_txt(
        rho_final, times, obs, stage_info, N,
        path=str(_HERE / "outputs" / "cat_drives_result.txt"),
    )

    plot_dynamics(times, obs, stage_info, N)

    print()
    print(sep)
    print("  Physical interpretation")
    print(sep)
    print(f"  Stage 0: Global Gaussian drive puts ALL {N} qubits simultaneously")
    print(f"           into superposition (|0>-i|1>)/sqrt(2).  Populations drop")
    print(f"           to 0.5 and coherences |rho_k[0,1]| rise to ~0.5 for all k.")
    print()
    print(f"  Stage 1: Parallel ZX coupling on ALL {N-1} bonds entangles adjacent")
    print(f"           pairs simultaneously.  Single-qubit coherences collapse as")
    print(f"           local quantum information transfers into multi-qubit")
    print(f"           correlations.  Entanglement entropy S(k) rises for all k.")
    print()
    print(f"  Stage 2: Local correction drives on qubits 1..{N-1} rotate the")
    print(f"           entangled state toward the computational (Z) basis.")
    print(f"           Contrast with the cascade example where entanglement")
    print(f"           spreads bond-by-bond in {N-1} sequential stages.")
    print(sep)
