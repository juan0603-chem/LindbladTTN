#!/usr/bin/env python3
"""
Five-qubit entanglement cascade — per-qubit Gaussian drives
============================================================

Extension of five_qubit_cat_state.py where EVERY qubit receives its own
Gaussian Rx(pi/2) drive at the moment it first enters the cascade, rather
than only qubit 0.  The ZX coupling and the target-qubit drive run
simultaneously during each stage — mirroring the "echo" pulse used in real
cross-resonance implementations.

Protocol
--------
Stage 0  [0,  T_H]:
    Gaussian Rx(pi/2) on qubit 0 only.
    H(t) = Omega(t) * X_0

Stage k  [T_H + (k-1)*T_CX,  T_H + k*T_CX]   for k = 1 .. N-1:
    ZX cross-resonance on bond (k-1, k)
    PLUS Gaussian Rx(pi/2) on the target qubit k.
    H(t) = (J/2) Z_{k-1} X_k  +  Omega(t) X_k

Visualisation
-------------
  - Per-qubit grid (4 rows x N cols):
      Row 0  : Population  P(|0>_k)(t)  with drive pulse overlay
      Row 1  : Coherence   |rho_k[0,1]|(t) with drive pulse overlay
      Row 2  : Entanglement entropy  S(k)(t)
      Row 3  : Wootters concurrence  C(k, k+1) for adjacent bonds;
               last column shows all bonds overlaid as a summary
  - Overview (2-panel): all pops + all cohs on single axes

System
------
  Superconducting transmon qubits
  ZX coupling  J/(2pi) = 10 MHz
  T1 = 100 us,  T2 = 50 us
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

# (Y x Y) used in concurrence calculation -- precomputed once
_YY = np.kron(Y, Y)


# ---------------------------------------------------------------------------
# Operator builders
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
# Liouville superoperators
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
                    - 0.5 * left_sop(LdL)
                    - 0.5 * right_sop(LdL))

def build_static_dissipator(N: int, gamma1: float, gamma_phi: float) -> np.ndarray:
    d = 2 ** N
    D = np.zeros((d * d, d * d), dtype=complex)
    for k in range(N):
        D += dissipator_sop(gamma1, single_site_op(N, k, Sm))
        if gamma_phi > 0:
            D += dissipator_sop(gamma_phi / 2.0, single_site_op(N, k, Z))
    return D


# ---------------------------------------------------------------------------
# Gaussian pulse helper
# ---------------------------------------------------------------------------

def gaussian_pulse(t: float, t_center: float, sigma: float, area: float) -> float:
    """Gaussian envelope normalised so integral(-inf..inf) = area."""
    return (area / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((t - t_center) / sigma) ** 2
    )


# ---------------------------------------------------------------------------
# scipy solve_ivp wrapper
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
    """Integrate drho/dt = L(t)*rho with complex->real splitting for scipy."""
    d2 = len(v0)
    y0 = np.concatenate([v0.real, v0.imag])

    def rhs(t, y):
        v  = y[:d2] + 1j * y[d2:]
        dv = L_func(t) @ v
        return np.concatenate([dv.real, dv.imag])

    sol = solve_ivp(rhs, t_span, y0, method=method,
                    t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"ODE failed: {sol.message}")
    return sol.t, sol.y[:d2].T + 1j * sol.y[d2:].T


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def simulate_cascade_with_drives(
    N: int,
    J: float  = 2 * np.pi * 0.010,
    T1: float = 100_000.0,
    T2: float =  50_000.0,
    T_H: float = 10.0,
    n_pts_per_stage: int = 80,
) -> tuple:
    """Cascade with simultaneous Gaussian drive on each target qubit.

    Stage 0   : Gaussian Rx(pi/2) on q0.
    Stage k   : ZX(k-1 -> k)  +  simultaneous Gaussian Rx(pi/2) on q_k.

    Returns
    -------
    times, rho_vecs, stage_info, elapsed
    """
    wall  = time.perf_counter()
    d     = 2 ** N
    d2    = d * d

    gamma1    = 1.0 / T1
    gamma_phi = max(0.0, 1.0 / T2 - 1.0 / (2.0 * T1))

    sigma = T_H / 5.0
    T_CX  = np.pi / (2.0 * J)

    L_dis = build_static_dissipator(N, gamma1, gamma_phi)

    psi0 = np.zeros(d, dtype=complex); psi0[0] = 1.0
    v_cur = np.outer(psi0, psi0.conj()).ravel(order="C")

    stage_bounds = [0.0, T_H] + [T_H + k * T_CX for k in range(1, N)]

    STAGE_COLORS = ["gold", "#7ec8e3", "#90d67a", "#ffb347", "#d4a0ff", "#f9a8d4"]

    times: list     = []
    rho_vecs: list  = []
    stage_labels: list = []
    stage_ends: list   = []

    for stage_idx, (t0, t1) in enumerate(
        zip(stage_bounds[:-1], stage_bounds[1:])
    ):
        t_eval = np.linspace(t0, t1, n_pts_per_stage + 1)[1:]
        t_c    = (t0 + t1) / 2.0

        if stage_idx == 0:
            H_drv = single_site_op(N, 0, X)
            def _make_L0(Hd, tc):
                def L0(t):
                    om = gaussian_pulse(t, tc, sigma, np.pi / 4.0)
                    return commutator_sop(om * Hd) + L_dis
                return L0
            L_func = _make_L0(H_drv, t_c)
            stage_labels.append("Gauss. q0")
        else:
            k      = stage_idx
            H_zx   = (J / 2.0) * zx_bond_op(N, k - 1)
            H_drv  = single_site_op(N, k, X)
            L_zx_c = commutator_sop(H_zx)

            def _make_Lk(Lzx, Hd, tc):
                def Lk(t):
                    om = gaussian_pulse(t, tc, sigma, np.pi / 4.0)
                    return Lzx + commutator_sop(om * Hd) + L_dis
                return Lk
            L_func = _make_Lk(L_zx_c, H_drv, t_c)
            stage_labels.append(f"ZX {k-1}->{k} + Gauss. q{k}")

        stage_ends.append(t1)

        t_out, v_out = solve_lindblad_stage(L_func, v_cur, (t0, t1),
                                             t_eval=t_eval)
        for ti, vi in zip(t_out, v_out):
            times.append(float(ti))
            rho_vecs.append(vi.copy())
        v_cur = v_out[-1]

    elapsed = time.perf_counter() - wall

    stage_info = {
        "T_H":          T_H,
        "T_CX":         T_CX,
        "T_tot":        T_H + (N - 1) * T_CX,
        "sigma":        sigma,
        "stage_ends":   stage_ends,
        "stage_labels": stage_labels,
        "stage_colors": STAGE_COLORS[:N],
    }
    return times, rho_vecs, stage_info, elapsed


# ---------------------------------------------------------------------------
# Observables  (single-qubit + pair concurrences)
# ---------------------------------------------------------------------------

def extract_rho(v: np.ndarray, N: int) -> np.ndarray:
    return v.reshape(2 ** N, 2 ** N)


def partial_trace(rho: np.ndarray, N: int, keep: int) -> np.ndarray:
    """2x2 reduced state for single qubit `keep`."""
    result  = rho.reshape([2] * (2 * N))
    removed = 0
    for ax in [k for k in range(N) if k != keep]:
        ax0 = ax - removed
        ax1 = ax0 + (N - removed)
        result  = np.trace(result, axis1=ax0, axis2=ax1)
        removed += 1
    return result


def partial_trace_pair(rho: np.ndarray, N: int, i: int, j: int) -> np.ndarray:
    """4x4 reduced density matrix for qubit pair (i, j) with i < j.

    Traces out all qubits except i and j.  The returned matrix has row/col
    order (|00>, |01>, |10>, |11>) in the {i, j} subspace.
    """
    assert i < j, "Require i < j"
    result  = rho.reshape([2] * (2 * N))
    removed = 0
    for qk in [k for k in range(N) if k not in (i, j)]:
        ax_bra = qk - removed
        ax_ket = ax_bra + (N - removed)
        result  = np.trace(result, axis1=ax_bra, axis2=ax_ket)
        removed += 1
    # Remaining axes: [bra_i, bra_j, ket_i, ket_j]
    return result.reshape(4, 4)


def wootters_concurrence(rho_pair: np.ndarray) -> float:
    """Wootters concurrence C in [0, 1] for a 4x4 two-qubit density matrix.

    C = max(0, sqrt(lambda_1) - sqrt(lambda_2) - sqrt(lambda_3) - sqrt(lambda_4))
    where lambda_k are eigenvalues (descending) of  rho * (Y x Y) * rho* * (Y x Y).
    """
    rho_tilde = _YY @ rho_pair.conj() @ _YY
    M         = rho_pair @ rho_tilde
    eigs      = np.linalg.eigvals(M)
    sqrt_eigs = np.sort(np.sqrt(np.clip(eigs.real, 0.0, None)))[::-1]
    return float(max(0.0, sqrt_eigs[0] - sqrt_eigs[1] - sqrt_eigs[2] - sqrt_eigs[3]))


def vn_entropy(rho2: np.ndarray) -> float:
    eigs = np.clip(np.linalg.eigvalsh(rho2).real, 0.0, 1.0)
    return float(-sum(l * np.log2(l) for l in eigs if l > 1e-15))


def compute_observables(rho_vecs: list, N: int) -> dict:
    """Single-qubit + pairwise observables at each saved snapshot.

    Returns
    -------
    dict with keys:
      trace  : (n,)       Tr(rho)
      pop    : (n, N)     P(|0>_k)
      coh    : (n, N)     |rho_k[0,1]|
      Zk     : (n, N)     <Z_k>
      Sk     : (n, N)     von Neumann entropy  [bits]
      purity : (n,)       Tr(rho^2)
      conc   : (n, N-1)   Wootters concurrence for adjacent pairs (k, k+1)
    """
    n = len(rho_vecs)
    traces = np.empty(n)
    pop    = np.zeros((n, N))
    coh    = np.zeros((n, N))
    Zk     = np.zeros((n, N))
    Sk     = np.zeros((n, N))
    purity = np.empty(n)
    conc   = np.zeros((n, N - 1))

    for i, v in enumerate(rho_vecs):
        rho   = extract_rho(v, N)
        tr    = np.trace(rho).real
        traces[i] = tr
        rho_n = rho / max(abs(tr), 1e-15)

        # Single-qubit quantities
        for k in range(N):
            rk        = partial_trace(rho_n, N, k)
            pop[i, k] = rk[0, 0].real
            coh[i, k] = abs(rk[0, 1])
            Zk[i, k]  = rk[0, 0].real - rk[1, 1].real
            Sk[i, k]  = vn_entropy(rk)

        purity[i] = np.real(np.trace(rho_n @ rho_n))

        # Adjacent-pair concurrences
        for pair in range(N - 1):
            rho_pair     = partial_trace_pair(rho_n, N, pair, pair + 1)
            conc[i, pair] = wootters_concurrence(rho_pair)

    return {"trace": traces, "pop": pop, "coh": coh,
            "Zk": Zk, "Sk": Sk, "purity": purity, "conc": conc}


def compute_drives(times: list, stage_info: dict, N: int) -> np.ndarray:
    """Normalised Gaussian drive envelope for each qubit (peak = 1, within stage window).

    Returns shape (n_times, N).
    """
    t_arr  = np.array(times)
    drives = np.zeros((len(times), N))
    sigma  = stage_info["sigma"]
    ends   = [0.0] + stage_info["stage_ends"]   # length N+1

    for k in range(N):
        t0  = ends[k]
        t1  = ends[k + 1]
        t_c = (t0 + t1) / 2.0
        envelope = np.exp(-0.5 * ((t_arr - t_c) / sigma) ** 2)
        mask     = (t_arr >= t0) & (t_arr <= t1)
        drives[:, k] = np.where(mask, envelope, 0.0)

    return drives   # shape (n_times, N), normalised peak = 1


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def shade_stages(ax, stage_info: dict, alpha: float = 0.13):
    ends   = stage_info["stage_ends"]
    colors = stage_info["stage_colors"]
    t_prev = 0.0
    for t_end, color in zip(ends, colors):
        ax.axvspan(t_prev, t_end, alpha=alpha, color=color, zorder=0)
        t_prev = t_end
    ax.set_xlim(0.0, stage_info["T_tot"])


def add_drive_overlay(ax, t_arr, drive_k, stage_color):
    """Overlay normalised Gaussian pulse shape on an existing axis (twinx)."""
    ax2 = ax.twinx()
    ax2.fill_between(t_arr, drive_k, color=stage_color,
                     alpha=0.18, zorder=1)
    ax2.plot(t_arr, drive_k, color=stage_color,
             lw=1.2, alpha=0.65, zorder=2, ls="--")
    ax2.set_ylim(0.0, 4.5)          # keep pulse small at top of panel
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["0", "pk"], fontsize=6, color="dimgray")
    ax2.tick_params(axis="y", length=2, labelsize=6, colors="dimgray")
    ax2.spines["right"].set_color("dimgray")
    ax2.spines["right"].set_linewidth(0.6)
    return ax2


# ---------------------------------------------------------------------------
# Main per-qubit grid plot  (4 rows x N cols)
# ---------------------------------------------------------------------------

def plot_per_qubit(times, obs, stage_info, N, drives):
    """4 x N per-qubit grid.

    Row 0 : Population     P(|0>_k)   + drive overlay
    Row 1 : Coherence      |rho[0,1]| + drive overlay
    Row 2 : Entropy        S(k)
    Row 3 : Concurrence    C(k, k+1) for cols 0..N-2;
                           all bonds overlaid in col N-1 (summary)
    """
    t_arr = np.array(times)
    qc    = plt.cm.plasma(np.linspace(0.1, 0.9, N))

    # Colours for the N-1 bonds (reuse qubit colours of left qubit)
    bond_colors = [qc[k] for k in range(N - 1)]

    n_rows = 4
    fig, axes = plt.subplots(
        n_rows, N,
        figsize=(4.0 * N, 11),
        sharex=True,
    )

    fig.suptitle(
        f"{N}-Qubit Cascade with Per-Qubit Gaussian Drives\n"
        f"(T1 = 100 us,  T2 = 50 us,  J/(2pi) = 10 MHz,  "
        f"sigma = {stage_info['sigma']:.1f} ns)\n"
        r"$\mathbf{Dashed\ fill}$ = normalised Gaussian drive envelope",
        fontsize=12, fontweight="bold", y=1.02,
    )

    row_labels = [
        "Population  P(|0>_k)",
        "Coherence  |rho_k[0,1]|",
        "Entropy  S(k)  [bits]",
        "Concurrence  C(pair)",
    ]
    row_ylims  = [(-0.05, 1.05), (-0.02, 0.55), (-0.05, 1.10), (-0.02, 1.05)]
    row_yhline = [0.5,           0.5,            1.0,            None]
    row_keys   = ["pop",         "coh",          "Sk",           None]

    for k in range(N):
        stage_col = stage_info["stage_colors"][k]   # colour of THIS qubit's stage

        # ── Rows 0 & 1: single-qubit quantities + drive ────────────────────
        for row in range(2):
            ax = axes[row, k]
            shade_stages(ax, stage_info)
            ax.plot(t_arr, obs[row_keys[row]][:, k],
                    color=qc[k], lw=2.0, zorder=3)
            ax.axhline(row_yhline[row], color="gray", ls="--",
                       lw=0.9, alpha=0.5)
            ax.set_ylim(*row_ylims[row])
            ax.tick_params(labelsize=8)
            if k == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)
            if row == 0:
                ax.set_title(f"Qubit {k}", fontsize=11,
                             fontweight="bold", color=qc[k])
            # Drive overlay
            add_drive_overlay(ax, t_arr, drives[:, k], stage_col)

        # ── Row 2: entropy ──────────────────────────────────────────────────
        ax = axes[2, k]
        shade_stages(ax, stage_info)
        ax.plot(t_arr, obs["Sk"][:, k], color=qc[k], lw=2.0, zorder=3)
        ax.axhline(1.0, color="gray", ls="--", lw=0.9, alpha=0.5)
        ax.set_ylim(*row_ylims[2])
        ax.tick_params(labelsize=8)
        if k == 0:
            ax.set_ylabel(row_labels[2], fontsize=9)

        # ── Row 3: concurrence ──────────────────────────────────────────────
        ax = axes[3, k]
        shade_stages(ax, stage_info)
        ax.set_ylim(*row_ylims[3])
        ax.tick_params(labelsize=8)
        if k == 0:
            ax.set_ylabel(row_labels[3], fontsize=9)
        ax.set_xlabel("Time (ns)", fontsize=9)
        ax.axhline(1.0, color="gray", ls="--", lw=0.9, alpha=0.5)

        if k < N - 1:
            # Primary bond for this column: C(k, k+1)
            ax.plot(t_arr, obs["conc"][:, k],
                    color=bond_colors[k], lw=2.2, zorder=3,
                    label=f"C({k},{k+1})")
            # All other bonds as thin gray lines for context
            for pair in range(N - 1):
                if pair != k:
                    ax.plot(t_arr, obs["conc"][:, pair],
                            color="lightgray", lw=0.8, zorder=2, alpha=0.7)
            ax.legend(fontsize=8, loc="upper left")
        else:
            # Last column: overlay ALL bonds as summary
            for pair in range(N - 1):
                ax.plot(t_arr, obs["conc"][:, pair],
                        color=bond_colors[pair], lw=1.8, zorder=3,
                        label=f"C({pair},{pair+1})")
            ax.legend(fontsize=7, loc="upper left", ncol=1)
            ax.set_title("All bonds", fontsize=9, color="gray",
                         style="italic")

    # Stage legend at bottom
    patches = [
        mpatches.Patch(color=c, alpha=0.6, label=l)
        for c, l in zip(stage_info["stage_colors"], stage_info["stage_labels"])
    ]
    fig.legend(handles=patches, loc="lower center", ncol=N,
               fontsize=8, bbox_to_anchor=(0.5, -0.05), framealpha=0.88)

    plt.tight_layout()
    out = _HERE / "cascade_gaussian_drives_per_qubit.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [Per-qubit grid saved to {out}]")


# ---------------------------------------------------------------------------
# Overview plot  (2-panel: populations + coherences)
# ---------------------------------------------------------------------------

def plot_overview(times, obs, stage_info, N, drives):
    """Two-panel overview: all populations + all coherences, with drive overlays."""
    t_arr = np.array(times)
    qc    = plt.cm.plasma(np.linspace(0.1, 0.9, N))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(
        f"{N}-Qubit Cascade with Per-Qubit Gaussian Drives — Populations & Coherences\n"
        f"(T1 = 100 us,  T2 = 50 us,  J/(2pi) = 10 MHz)",
        fontsize=13, fontweight="bold",
    )

    for k in range(N):
        sc = stage_info["stage_colors"][k]
        ax1.plot(t_arr, obs["pop"][:, k], color=qc[k], lw=2.0, label=f"q{k}")
        ax2.plot(t_arr, obs["coh"][:, k], color=qc[k], lw=2.0, label=f"q{k}")
        # Drive: fill between 0 and (normalised envelope * 0.08) so it sits as
        # a small bump at the bottom of each panel — no twinx needed here
        scale1 = 0.08   # height in population units
        scale2 = 0.04   # height in coherence units
        ax1.fill_between(t_arr, drives[:, k] * scale1, color=sc,
                         alpha=0.25, zorder=0)
        ax2.fill_between(t_arr, drives[:, k] * scale2, color=sc,
                         alpha=0.25, zorder=0)

    for ax in (ax1, ax2):
        shade_stages(ax, stage_info)

    ax1.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6)
    ax1.set_ylabel("P(|0>_k)", fontsize=11)
    ax1.set_title("Populations — shaded bumps show each qubit's Gaussian drive", fontsize=10)
    ax1.legend(fontsize=9, ncol=N, loc="lower right")
    ax1.set_ylim(-0.05, 1.05)

    ax2.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6)
    ax2.set_ylabel("|rho_k[0,1]|", fontsize=11)
    ax2.set_xlabel("Time (ns)", fontsize=11)
    ax2.set_title("Coherences — each rises during local Gaussian, collapses as ZX entangles",
                  fontsize=10)
    ax2.legend(fontsize=9, ncol=N, loc="upper right")
    ax2.set_ylim(-0.02, 0.55)

    patches = [
        mpatches.Patch(color=c, alpha=0.6, label=l)
        for c, l in zip(stage_info["stage_colors"], stage_info["stage_labels"])
    ]
    fig.legend(handles=patches, loc="lower center", ncol=N,
               fontsize=9, bbox_to_anchor=(0.5, -0.06), framealpha=0.88)

    plt.tight_layout()
    out = _HERE / "cascade_gaussian_drives_overview.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [Overview saved to {out}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N   = 5
    J   = 2 * np.pi * 0.010
    T1  = 100_000.0
    T2  =  50_000.0
    T_H = 10.0

    T_CX  = np.pi / (2 * J)
    sigma = T_H / 5.0
    T_tot = T_H + (N - 1) * T_CX + 1000 #for relaxation obsevation

    sep = "-" * 66
    print()
    print(sep)
    print(f"  {N}-Qubit Cascade with Per-Qubit Gaussian Drives")
    print(sep)
    print(f"  System         : {N} superconducting transmon qubits")
    print(f"  ZX coupling    : J/(2pi) = {J/(2*np.pi)*1e3:.1f} MHz")
    print(f"  T1 / T2        : {T1/1e3:.0f} us / {T2/1e3:.0f} us")
    print(f"  Gaussian sigma : {sigma:.2f} ns  (pi/4 area each)")
    print()
    print(f"  Stage 0  [0, {T_H:.1f} ns]")
    print(f"           Gaussian Rx(pi/2) on q0 only")
    for k in range(1, N):
        t0 = T_H + (k - 1) * T_CX
        t1 = T_H + k * T_CX
        print(f"  Stage {k}  [{t0:.1f}, {t1:.1f} ns]")
        print(f"           ZX({k-1}->{k}) + Gaussian Rx(pi/2) on q{k}")
    print()
    print(f"  Total time     : {T_tot:.2f} ns")
    print(f"  Liouville dim  : 4^{N} = {4**N:,}")
    print(sep)

    print()
    print("  Running simulation ...")
    times, rho_vecs, stage_info, elapsed = simulate_cascade_with_drives(
        N, J=J, T1=T1, T2=T2, T_H=T_H, n_pts_per_stage=80
    )
    print(f"  Completed in {elapsed:.2f} s  ({len(times)} snapshots)")

    print("  Computing observables (includes pairwise concurrences) ...")
    obs    = compute_observables(rho_vecs, N)
    drives = compute_drives(times, stage_info, N)

    rho_final = extract_rho(rho_vecs[-1], N)

    print()
    print(f"  Results at t = {T_tot:.2f} ns")
    print(f"    Tr(rho)  = {obs['trace'][-1]:.8f}")
    print(f"    Purity   = {obs['purity'][-1]:.4f}")
    print()
    print(f"    {'Qubit':>6}  {'P(|0>)':>8}  {'|coh|':>8}  {'<Z>':>8}  {'S [bits]':>10}")
    print("    " + "-" * 46)
    for k in range(N):
        print(f"    {'q'+str(k):>6}  {obs['pop'][-1,k]:>8.4f}  "
              f"{obs['coh'][-1,k]:>8.4f}  {obs['Zk'][-1,k]:>+8.4f}  "
              f"{obs['Sk'][-1,k]:>10.4f}")
    print()
    print(f"    {'Bond':>8}  {'C (final)':>10}")
    print("    " + "-" * 20)
    for pair in range(N - 1):
        print(f"    ({pair},{pair+1}):  {obs['conc'][-1, pair]:>10.4f}")

    print()
    print("  Generating plots ...")
    plot_overview(times, obs, stage_info, N, drives)
    plot_per_qubit(times, obs, stage_info, N, drives)

    print()
    print(sep)
    print("  Physical notes on concurrence")
    print(sep)
    print(f"  Concurrence C in [0,1] is the Wootters entanglement measure for")
    print(f"  two-qubit reduced states.  C=0 means separable, C=1 means")
    print(f"  maximally entangled (Bell state).  It is strictly stronger than")
    print(f"  entropy: C>0 guarantees entanglement while S(k) can be nonzero")
    print(f"  for classically mixed states.")
    print()
    print(f"  In this cascade each bond C(k,k+1) rises during Stage k+1 when")
    print(f"  the ZX gate acts.  The simultaneous Gaussian drive on the target")
    print(f"  qubit modifies the entangling trajectory compared to the bare")
    print(f"  cascade: the drive competes with the ZX coupling, so C(k,k+1)")
    print(f"  may not reach its maximum mid-stage before continuing to evolve.")
    print(sep)
