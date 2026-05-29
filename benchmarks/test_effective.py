# coding: utf-8
"""Validation of effective-Hamiltonian tools (M5)."""

from __future__ import annotations

import numpy as np
import pytest

from lindblad_ttn.effective import dispersive_shift, schrieffer_wolff, magnus_average


def test_dispersive_shift_jc():
    """Dispersive χ from numerical diagonalisation matches g²/Δ to leading order."""
    omega_q = 5.0
    omega_c = 4.7
    g = 0.05
    Delta = omega_q - omega_c
    chi_expected = g * g / Delta

    # Build full JC Hamiltonian (qubit dim 2, cavity dim 8)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    sp = np.array([[0, 1], [0, 0]], dtype=complex)
    sm = np.array([[0, 0], [1, 0]], dtype=complex)
    Nc = 8
    a = np.zeros((Nc, Nc), dtype=complex)
    for n in range(1, Nc):
        a[n - 1, n] = np.sqrt(n)
    adag = a.T.conj()
    H = (
        0.5 * omega_q * np.kron(sz, np.eye(Nc))
        + omega_c * np.kron(np.eye(2), adag @ a)
        + g * np.kron(sp, a)
        + g * np.kron(sm, adag)
    )
    chi = dispersive_shift(H, qubit_dim=2, cavity_dim=Nc, n_max=1)
    rel = abs(chi - chi_expected) / abs(chi_expected)
    # χ from numerics includes higher-order g²/Δ corrections; allow ~10% rel
    # to the leading-order formula g²/Δ.
    assert rel < 0.10, f"dispersive_shift: chi={chi:.4e}, expected={chi_expected:.4e}"


def test_magnus_first_order_cosine_drive():
    """Magnus order-1 of A·cos(ωt)·σx over period 2π/ω is zero."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)

    def H_t(t):
        return 1.0 * np.cos(2.0 * t) * sx

    H_avg = magnus_average(H_t, period=np.pi, n_points=256, order=1)
    assert np.max(np.abs(H_avg)) < 1e-2, f"order-1 Magnus should ≈ 0, got max={np.max(np.abs(H_avg)):.2e}"


def test_sw_block_diagonalises_two_level():
    """Two-level toy: H0 + V, with off-diagonal V. SW removes it to 2nd order."""
    H0 = np.diag([0.0, 1.0])
    V = 0.05 * np.array([[0, 1], [1, 0]], dtype=complex)
    PA = np.array([[1, 0], [0, 0]], dtype=complex)  # subspace A = |0>
    H_eff = schrieffer_wolff(H0, V, projector_A=PA, order=2)
    # Off-diagonal block should be much smaller than V
    od = abs(H_eff[0, 1])
    assert od < 0.005, f"SW off-diagonal too large: {od:.2e}"
