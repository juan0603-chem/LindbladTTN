# coding: utf-8
"""Analytical validation: single qubit decay.

H0 = (omega/2) * sigma_z
L  = [(gamma, sigma_minus)]

Analytical solution:
    rho_11(t) = rho_11(0) * exp(-gamma*t)
    rho_01(t) = rho_01(0) * exp(-(gamma/2 + i*omega)*t)
"""

import numpy as np
import pytest
from lindblad_ttn import LindbladTTN


# Pauli matrices
sz = np.array([[1, 0], [0, -1]], dtype=complex)
sm = np.array([[0, 0], [1, 0]], dtype=complex)  # sigma_minus = |0><1|

OMEGA = 1.0
GAMMA = 0.3


def analytical_rho(t, rho0, omega, gamma):
    """Compute analytical solution for single qubit decay.

    Convention: H = omega/2 * sz = diag(+omega/2, -omega/2).
    State |0> has energy +omega/2 (excited), |1> has -omega/2 (ground).
    sm = [[0,0],[1,0]] = |1><0| is the lowering operator (excited -> ground).
    => rho[0,0] (excited population) decays at rate gamma.
    """
    r00_0 = rho0[0, 0]
    r01_0 = rho0[0, 1]
    r10_0 = rho0[1, 0]

    r00_t = r00_0 * np.exp(-gamma * t)
    r11_t = 1.0 - r00_t  # trace preservation
    r01_t = r01_0 * np.exp(-(gamma / 2.0 + 1j * omega) * t)
    r10_t = r10_0 * np.exp(-(gamma / 2.0 - 1j * omega) * t)

    return np.array([[r00_t, r01_t], [r10_t, r11_t]])


@pytest.fixture
def solver():
    return LindbladTTN(
        H0=0.5 * OMEGA * sz,
        f=None,
        V=None,
        L_ops=[(GAMMA, sm)],
        n_sites=1,
        bond_dim=4,
        topology="train",
        device="cpu",
        strategy="ps1",
    )


def test_single_qubit_decay(solver):
    """rho11 and rho01 must match analytical solution within 1e-4."""
    rho0 = np.array([[0.3, 0.4 + 0.2j], [0.4 - 0.2j, 0.7]], dtype=complex)
    rho0 /= np.trace(rho0)

    T = 5.0 / GAMMA
    dt = 0.005
    save_every = 20

    result = solver.run(
        rho0=rho0,
        t_span=(0.0, T),
        dt=dt,
        observables=[],
        save_every=save_every,
        verbose=False,
    )

    times = result.times
    rho_final = result.rho_final

    # Check final state against analytical solution
    rho_exact = analytical_rho(times[-1], rho0, OMEGA, GAMMA)
    err = np.max(np.abs(rho_final - rho_exact))
    assert err < 5e-3, f"Final state error: {err:.2e}"

    # Check trace preservation
    norms = result.norm
    assert np.all(np.abs(norms - 1.0) < 1e-4), f"Trace drift: {np.max(np.abs(norms-1.0)):.2e}"


def test_single_qubit_norm(solver):
    """Trace should remain 1 throughout evolution."""
    rho0 = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)
    result = solver.run(
        rho0=rho0,
        t_span=(0.0, 2.0),
        dt=0.01,
        verbose=False,
    )
    assert np.all(np.abs(result.norm - 1.0) < 1e-3)


def test_single_qubit_ground_state_is_fixed(solver):
    """Ground state rho = |1><1| is a fixed point of the Lindblad equation.

    Convention: sm = |1><0| lowers excited|0> -> ground|1>.
    The steady state is |1><1| (all population in ground state).
    """
    rho0 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    result = solver.run(
        rho0=rho0,
        t_span=(0.0, 2.0),
        dt=0.01,
        verbose=False,
    )
    # rho_final should still be |1><1|
    err = np.max(np.abs(result.rho_final - rho0))
    assert err < 1e-3, f"Fixed point error: {err:.2e}"
