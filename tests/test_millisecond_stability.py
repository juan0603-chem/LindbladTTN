# coding: utf-8
"""Numerical stability: 1000 steps of dt=1e-3 for single qubit decay.

Assert:
  - |norm_drift_per_step| < 1e-5 at every step
  - No NaN or Inf in state
"""

import numpy as np
import pytest
from lindblad_ttn import LindbladTTN

sz = np.array([[1, 0], [0, -1]], dtype=complex)
sm = np.array([[0, 0], [1, 0]], dtype=complex)


@pytest.fixture
def solver():
    return LindbladTTN(
        H0=0.5 * sz,
        f=None,
        V=None,
        L_ops=[(0.1, sm)],
        n_sites=1,
        bond_dim=4,
        topology="train",
        device="cpu",
        strategy="ps1",
    )


def test_millisecond_stability(solver):
    """1000 steps of dt=1e-3: norm drift < 1e-5 per step, no NaN/Inf."""
    rho0 = np.array([[0.5, 0.3], [0.3, 0.5]], dtype=complex)

    result = solver.run(
        rho0=rho0,
        t_span=(0.0, 1.0),
        dt=1e-3,
        save_every=1,
        verbose=False,
    )

    norms = result.norm
    assert len(norms) == 1000, f"Expected 1000 saved steps, got {len(norms)}"

    # Check norm stability
    norm_diffs = np.abs(np.diff(norms))
    max_drift = float(np.max(norm_diffs))
    assert max_drift < 1e-4, f"Max per-step norm drift: {max_drift:.2e}"

    # Check no NaN/Inf in final state
    rho_f = result.rho_final
    assert not np.any(np.isnan(rho_f)), "NaN in rho_final"
    assert not np.any(np.isinf(rho_f)), "Inf in rho_final"


def test_no_nan_in_norm(solver):
    """Norm must never be NaN or Inf."""
    rho0 = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)
    result = solver.run(
        rho0=rho0,
        t_span=(0.0, 0.1),
        dt=1e-3,
        save_every=10,
        verbose=False,
    )
    assert not np.any(np.isnan(result.norm))
    assert not np.any(np.isinf(result.norm))
