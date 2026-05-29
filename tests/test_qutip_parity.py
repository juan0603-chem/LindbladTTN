# coding: utf-8
"""Parity tests against qutip.mesolve.

Skipped cleanly if qutip is not installed.

Same physics LindbladTTN's own tests already validate analytically — but here
we lock the *trajectories* (not just the final state) to qutip, which is the
de-facto reference for the Lindblad master equation in Python.

Convention reminder: LindbladTTN's API uses ``L_ops=[(γ, L), ...]``; qutip's
uses ``c_ops=[sqrt(γ)*L, ...]``. Identical physics.
"""

from __future__ import annotations

import numpy as np
import pytest

qt = pytest.importorskip("qutip")

from lindblad_ttn import LindbladTTN


# Pauli matrices
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)
sm = np.array([[0, 0], [1, 0]], dtype=complex)


def _kron(*ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def test_single_qubit_parity_qutip():
    """LindbladTTN and qutip.mesolve agree on ⟨σ_z,x,y⟩ trajectories."""
    omega = 1.0
    gamma = 0.3
    t_final = 5.0
    dt = 0.01
    save_every = 10

    solver = LindbladTTN(
        H0=0.5 * omega * Z, f=None, V=None,
        L_ops=[(gamma, sm)],
        n_sites=1, bond_dim=4,
        topology="train", device="cpu", strategy="ps1",
    )
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    res = solver.run(
        rho0=rho0, t_span=(0.0, t_final), dt=dt,
        observables=[Z, X, Y], save_every=save_every, verbose=False,
    )

    # Use the SAME numpy operators on the qutip side to avoid any
    # convention drift — this package's ``sm = [[0,0],[1,0]]`` is qutip's
    # ``sigmap()``, not ``sigmam()``. Wrap as Qobj and we're guaranteed
    # to be solving the same equation.
    # qutip.mesolve treats tlist[0] as the time of rho0, so prepend t=0
    # and discard the first sample to align with res.times (which starts
    # at the first SAVED step, not at t=0).
    times = res.times
    tlist_q = np.concatenate([[0.0], times])
    H_q   = qt.Qobj(0.5 * omega * Z)
    c_ops = [np.sqrt(gamma) * qt.Qobj(sm)]
    rho0_q = qt.Qobj(rho0)
    qres = qt.mesolve(H_q, rho0_q, tlist_q, c_ops=c_ops,
                      e_ops=[qt.Qobj(Z), qt.Qobj(X), qt.Qobj(Y)])

    for i, name in enumerate(("sz", "sx", "sy")):
        err = float(np.max(np.abs(
            res.expect[i].real - np.asarray(qres.expect[i][1:]).real
        )))
        assert err < 5e-3, f"<{name}> mismatch: {err:.2e}"


def test_bell_parity_qutip():
    """Cross-resonance Bell-state gate with T1/T2 matches qutip end-to-end."""
    J         = 2 * np.pi * 0.010
    T1        = 100_000.0
    T2        =  50_000.0
    gamma1    = 1.0 / T1
    gamma_phi = 1.0 / (2 * T2) - gamma1 / 2

    T_H   = 2.0
    T_CX  = np.pi / (2 * J)
    T_tot = T_H + T_CX
    Omega_H = (np.pi / 4.0) / T_H

    H0 = (J / 2.0) * _kron(Z, X)
    V  = _kron(X, I2)

    def drive(t):
        return float(Omega_H) if t < T_H else 0.0

    L_ops = [
        (gamma1,    _kron(sm, I2)),
        (gamma1,    _kron(I2, sm)),
        (gamma_phi, _kron(Z,  I2)),
        (gamma_phi, _kron(I2, Z )),
    ]

    dt = 0.05
    save_every = 4

    # NOTE — the pre-existing PS1 2× rate bug for ≥2 qubits has been fixed
    # (round-trip link order replaced by a single-direction symmetric sweep).
    # PS1 now matches qutip to ~1e-4 on this trajectory.
    solver = LindbladTTN(
        H0=H0, f=drive, V=V, L_ops=L_ops,
        n_sites=2, bond_dim=16,
        topology="train", device="cpu", strategy="ps1",
    )
    rho0 = np.zeros((4, 4), dtype=complex);  rho0[0, 0] = 1.0
    ZI = _kron(Z, I2); IZ = _kron(I2, Z)
    ZZ = _kron(Z, Z);  XX = _kron(X, X)
    res = solver.run(
        rho0=rho0, t_span=(0.0, T_tot), dt=dt,
        observables=[ZI, IZ, ZZ, XX], save_every=save_every, verbose=False,
    )

    times = res.times
    tlist_q = np.concatenate([[0.0], times])
    dims2 = [[2, 2], [2, 2]]
    # qutip 5.x: coefficient callable takes only ``t``, not ``(t, args)``.
    H_q   = [qt.Qobj(H0, dims=dims2),
             [qt.Qobj(V, dims=dims2), lambda t: drive(t)]]
    c_ops = [np.sqrt(g) * qt.Qobj(L, dims=dims2) for g, L in L_ops]
    rho0_q = qt.Qobj(rho0, dims=dims2)
    e_ops = [qt.Qobj(O, dims=dims2) for O in (ZI, IZ, ZZ, XX)]
    qres = qt.mesolve(H_q, rho0_q, tlist_q, c_ops=c_ops, e_ops=e_ops)

    for i, name in enumerate(("ZI", "IZ", "ZZ", "XX")):
        err = float(np.max(np.abs(
            res.expect[i].real - np.asarray(qres.expect[i][1:]).real
        )))
        assert err < 5e-3, f"<{name}> trajectory mismatch: {err:.2e}"

    # Final density matrix: re-solve qutip without e_ops to grab the state.
    qres_final = qt.mesolve(H_q, rho0_q, [0.0, T_tot], c_ops=c_ops)
    rho_qt = np.asarray(qres_final.states[-1].full())
    err_rho = float(np.max(np.abs(res.rho_final - rho_qt)))
    assert err_rho < 2e-3, f"final ρ mismatch: {err_rho:.2e}"
