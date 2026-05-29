# coding: utf-8
"""Physical consistency tests: two-qubit driven system.

H(t) = omega*(sz⊗I + I⊗sz) + cos(t)*(sx⊗I)
L    = [(gamma, sm⊗I), (gamma, I⊗sm)]

At every saved step assert:
  - |tr(rho) - 1| < 1e-5   (trace preservation)
  - ||rho - rho†|| < 1e-5  (Hermiticity)
  - min(eigvals(rho)) > -1e-4  (positivity)
"""

import numpy as np
import pytest
from lindblad_ttn import LindbladTTN

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
sm = np.array([[0, 0], [1, 0]], dtype=complex)

OMEGA = 0.5
GAMMA = 0.1


@pytest.fixture
def solver():
    H0 = OMEGA * (np.kron(sz, I2) + np.kron(I2, sz))
    V = np.kron(sx, I2)
    f = np.cos
    L_ops = [(GAMMA, np.kron(sm, I2)), (GAMMA, np.kron(I2, sm))]

    return LindbladTTN(
        H0=H0,
        f=f,
        V=V,
        L_ops=L_ops,
        n_sites=2,
        bond_dim=8,
        topology="train",
        device="cpu",
        strategy="ps1",
    )


def test_trace_preserved(solver):
    """Trace must stay 1 at every step."""
    rho0 = np.eye(4, dtype=complex) / 4.0
    result = solver.run(
        rho0=rho0,
        t_span=(0.0, 2.0),
        dt=0.02,
        verbose=False,
    )
    assert np.all(np.abs(result.norm - 1.0) < 1e-4), \
        f"Max trace drift: {np.max(np.abs(result.norm-1.0)):.2e}"


def test_hermiticity(solver):
    """rho_final must be Hermitian."""
    rho0 = np.eye(4, dtype=complex) / 4.0
    result = solver.run(
        rho0=rho0,
        t_span=(0.0, 1.0),
        dt=0.02,
        verbose=False,
    )
    rho_f = result.rho_final
    herm_err = np.max(np.abs(rho_f - rho_f.conj().T))
    assert herm_err < 1e-4, f"Hermiticity error: {herm_err:.2e}"


def test_positivity(solver):
    """rho_final must be positive semidefinite."""
    rho0 = np.eye(4, dtype=complex) / 4.0
    result = solver.run(
        rho0=rho0,
        t_span=(0.0, 1.0),
        dt=0.02,
        verbose=False,
    )
    rho_f = result.rho_final
    eigvals = np.linalg.eigvalsh(0.5 * (rho_f + rho_f.conj().T))
    min_eig = float(np.min(eigvals))
    assert min_eig > -1e-3, f"Min eigenvalue: {min_eig:.2e}"
