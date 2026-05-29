# coding: utf-8
"""Pulse-shape library for time-dependent control (M6).

Each pulse returns a *callable* ``f(t) -> float`` (the envelope) that can
be plugged into ``LindbladTTN(drives=[(f, V)])`` or its heterogeneous
counterpart ``drives_nd=[(f, V_terms)]``.

Common pulses
-------------
* :func:`constant_pulse` — constant envelope on ``[t_start, t_end]``.
* :func:`gaussian` — Gaussian envelope; standard ``π/2`` or ``π`` rotations.
* :func:`drag` — DRAG-corrected Gaussian (adds an i·(d/dt envelope)/α
  imaginary component to suppress |0⟩↔|2⟩ leakage in transmons).
* :func:`square_rise` — flat-top with cosine rise/fall.
* :func:`cosine_drive` — pure carrier ``A cos(ωt + φ)``.
* :func:`sequence` — concatenation of multiple pulses on disjoint windows.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


PulseFn = Callable[[float], float]


# ---------------------------------------------------------------------------
# Basic shapes
# ---------------------------------------------------------------------------

def constant_pulse(amp: float, t_start: float = 0.0, t_end: float = float("inf")) -> PulseFn:
    """Constant ``amp`` on ``[t_start, t_end]``, zero outside."""
    def f(t: float) -> float:
        return float(amp) if (t_start <= t < t_end) else 0.0
    return f


def gaussian(amp: float, t0: float, sigma: float, truncate_at: float = 3.0) -> PulseFn:
    """Gaussian envelope centred at ``t0`` with width ``sigma``.

    ``f(t) = amp · exp(−(t−t0)² / (2 σ²))`` for ``|t − t0| ≤ truncate_at · σ``,
    else 0.  Truncation makes the integral finite for use as a discrete pulse.
    """
    t_low = t0 - truncate_at * sigma
    t_high = t0 + truncate_at * sigma
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    a = float(amp)

    def f(t: float) -> float:
        if t < t_low or t > t_high:
            return 0.0
        dt = t - t0
        return a * float(np.exp(-dt * dt * inv2s2))
    return f


def drag(
    amp: float,
    t0: float,
    sigma: float,
    anharm: float,
    beta: float = 1.0,
    truncate_at: float = 3.0,
) -> tuple[PulseFn, PulseFn]:
    """DRAG-corrected Gaussian envelope.

    Returns two callables ``(f_I(t), f_Q(t))`` representing the in-phase and
    quadrature components.  The Q-component is ``-(d/dt) f_I / α`` scaled by
    ``β`` and used to suppress leakage to ``|2⟩`` for weakly anharmonic qubits
    (transmons).

    Parameters
    ----------
    amp : float
        Peak amplitude of the in-phase Gaussian.
    t0 : float
        Centre time.
    sigma : float
        Gaussian width.
    anharm : float
        Qubit anharmonicity (typically negative; same units as time⁻¹).
    beta : float
        DRAG coefficient (often 0.5–1.0); set to 0 for pure Gaussian.
    truncate_at : float
        Truncation in units of sigma.
    """
    fI = gaussian(amp, t0, sigma, truncate_at)
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    a = float(amp)
    beta_over_alpha = beta / anharm if anharm != 0.0 else 0.0

    def fQ(t: float) -> float:
        dt = t - t0
        if dt < -truncate_at * sigma or dt > truncate_at * sigma:
            return 0.0
        # d/dt [amp · exp(-dt²/2σ²)] = amp · (-dt/σ²) · exp(-dt²/2σ²)
        env = a * float(np.exp(-dt * dt * inv2s2))
        deriv = env * (-dt / (sigma * sigma))
        return -beta_over_alpha * deriv
    return fI, fQ


def square_rise(
    amp: float,
    t_start: float,
    t_end: float,
    t_rise: float = 0.0,
) -> PulseFn:
    """Flat-top pulse with cosine-shaped rise and fall.

    For ``t in [t_start, t_start + t_rise]`` the envelope rises smoothly from
    0 to ``amp`` via ``½ amp (1 - cos(π Δt / t_rise))``; analogously for fall.
    """
    a = float(amp)
    t_flat_end = t_end - t_rise

    def f(t: float) -> float:
        if t < t_start or t >= t_end:
            return 0.0
        if t_rise <= 0:
            return a if t_start <= t < t_end else 0.0
        if t < t_start + t_rise:
            phi = float(np.pi * (t - t_start) / t_rise)
            return 0.5 * a * (1.0 - float(np.cos(phi)))
        if t > t_flat_end:
            phi = float(np.pi * (t - t_flat_end) / t_rise)
            return 0.5 * a * (1.0 + float(np.cos(phi)))
        return a
    return f


def cosine_drive(amp: float, omega: float, phase: float = 0.0,
                 t_start: float = 0.0, t_end: float = float("inf")) -> PulseFn:
    """Continuous-wave carrier ``A · cos(ω t + φ)`` gated on ``[t_start, t_end]``."""
    a = float(amp)
    om = float(omega)
    ph = float(phase)

    def f(t: float) -> float:
        if t < t_start or t >= t_end:
            return 0.0
        return a * float(np.cos(om * t + ph))
    return f


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def sequence(*pulses: PulseFn) -> PulseFn:
    """Sum of pulse callables — natural way to build composite waveforms."""
    if not pulses:
        return lambda t: 0.0

    def f(t: float) -> float:
        return float(sum(p(t) for p in pulses))
    return f


# ---------------------------------------------------------------------------
# Integration check (utility)
# ---------------------------------------------------------------------------

def integrate_pulse(f: PulseFn, t_start: float, t_end: float, n: int = 1024) -> float:
    """Trapezoidal integral of ``f(t)`` on ``[t_start, t_end]``.

    Useful for choosing pulse amplitudes that realise a target rotation
    angle ``θ = ∫ Ω(t) dt`` (e.g. a π/2 pulse).
    """
    ts = np.linspace(t_start, t_end, n)
    vals = np.array([f(t) for t in ts])
    return float(np.trapz(vals, ts))
