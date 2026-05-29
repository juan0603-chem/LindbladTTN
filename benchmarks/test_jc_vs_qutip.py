# coding: utf-8
"""Jaynes–Cummings: LindbladTTN heterogeneous path vs QuTiP."""

from __future__ import annotations

import numpy as np
import pytest

qt = pytest.importorskip("qutip")

from lindblad_ttn import LindbladTTN
from lindblad_ttn.sites import boson_site, spin_half_site


def test_jc_rabi_vacuum_matches_qutip():
    """Vacuum Rabi oscillations under the JC Hamiltonian, no dissipation."""
    q = spin_half_site("q0")
    c = boson_site(6, "c0")

    omega_q, omega_c, g = 1.0, 1.0, 0.1
    H_terms = [
        (0.5 * omega_q, {q.name: q.sz}),
        (omega_c,       {c.name: c.adag @ c.a}),
        (g,             {q.name: q.sp, c.name: c.a}),
        (g,             {q.name: q.sm, c.name: c.adag}),
    ]
    solver = LindbladTTN(
        sites=[q, c], H_terms=H_terms, L_terms=[],
        bond_dim=12, topology="train", strategy="ps1",
    )

    rho_q = np.array([[1, 0], [0, 0]], dtype=complex)  # |e>
    rho_c = np.zeros((6, 6), dtype=complex); rho_c[0, 0] = 1.0  # vacuum
    rho0 = np.kron(rho_q, rho_c)
    sz_full = np.kron(np.array([[1, 0], [0, -1]], dtype=complex), np.eye(6, dtype=complex))

    result = solver.run(
        rho0=rho0, t_span=(0.0, 10.0), dt=0.005,
        observables=[sz_full], save_every=200, verbose=False,
    )

    # QuTiP reference
    sz_q = qt.sigmaz()
    sp_q = qt.Qobj(np.array([[0, 1], [0, 0]], dtype=complex))
    sm_q = qt.Qobj(np.array([[0, 0], [1, 0]], dtype=complex))
    a = qt.destroy(6)
    H_q = (
        0.5 * omega_q * qt.tensor(sz_q, qt.qeye(6))
        + omega_c * qt.tensor(qt.qeye(2), a.dag() * a)
        + g * qt.tensor(sp_q, a)
        + g * qt.tensor(sm_q, a.dag())
    )
    rho0_q = qt.Qobj(rho0, dims=[[2, 6], [2, 6]])
    tlist = np.concatenate([[0.0], result.times])
    qres = qt.mesolve(H_q, rho0_q, tlist, c_ops=[],
                      e_ops=[qt.Qobj(sz_full, dims=[[2, 6], [2, 6]])])
    err = float(np.max(np.abs(result.expect[0].real - np.asarray(qres.expect[0][1:]).real)))
    assert err < 5e-3, f"<sigma_z> vs QuTiP mismatch: {err:.2e}"


def test_jc_with_decay_matches_qutip():
    """JC with cavity photon loss; trace and observables track QuTiP."""
    q = spin_half_site("q0")
    c = boson_site(8, "c0")

    omega_q, omega_c, g, kappa = 1.0, 1.0, 0.15, 0.03
    H_terms = [
        (0.5 * omega_q, {q.name: q.sz}),
        (omega_c,       {c.name: c.adag @ c.a}),
        (g,             {q.name: q.sp, c.name: c.a}),
        (g,             {q.name: q.sm, c.name: c.adag}),
    ]
    L_terms = [(kappa, {c.name: c.a})]
    solver = LindbladTTN(
        sites=[q, c], H_terms=H_terms, L_terms=L_terms,
        bond_dim=16, topology="train", strategy="ps1",
    )

    rho_q = np.array([[1, 0], [0, 0]], dtype=complex)
    rho_c = np.zeros((8, 8), dtype=complex); rho_c[0, 0] = 1.0
    rho0 = np.kron(rho_q, rho_c)
    sz_full = np.kron(np.array([[1, 0], [0, -1]], dtype=complex), np.eye(8, dtype=complex))
    n_full = np.kron(np.eye(2), np.diag(np.arange(8)).astype(complex))

    result = solver.run(
        rho0=rho0, t_span=(0.0, 25.0), dt=0.01,
        observables=[sz_full, n_full], save_every=100, verbose=False,
    )

    # QuTiP reference
    sz_q = qt.sigmaz()
    sp_q = qt.Qobj(np.array([[0, 1], [0, 0]], dtype=complex))
    sm_q = qt.Qobj(np.array([[0, 0], [1, 0]], dtype=complex))
    a = qt.destroy(8)
    H_q = (
        0.5 * omega_q * qt.tensor(sz_q, qt.qeye(8))
        + omega_c * qt.tensor(qt.qeye(2), a.dag() * a)
        + g * qt.tensor(sp_q, a)
        + g * qt.tensor(sm_q, a.dag())
    )
    c_ops = [np.sqrt(kappa) * qt.tensor(qt.qeye(2), a)]
    rho0_q = qt.Qobj(rho0, dims=[[2, 8], [2, 8]])
    tlist = np.concatenate([[0.0], result.times])
    qres = qt.mesolve(H_q, rho0_q, tlist, c_ops=c_ops,
                      e_ops=[qt.Qobj(sz_full, dims=[[2, 8], [2, 8]]),
                             qt.Qobj(n_full, dims=[[2, 8], [2, 8]])])
    err_sz = float(np.max(np.abs(result.expect[0].real - np.asarray(qres.expect[0][1:]).real)))
    err_n = float(np.max(np.abs(result.expect[1].real - np.asarray(qres.expect[1][1:]).real)))
    assert err_sz < 1e-2, f"<sigma_z> mismatch: {err_sz:.2e}"
    assert err_n < 1e-2, f"<n> mismatch: {err_n:.2e}"
    assert np.all(np.abs(result.norm - 1.0) < 1e-3), \
        f"trace drift: {np.max(np.abs(result.norm-1.0)):.2e}"
