# coding: utf-8
"""Scaling test: N-qubit pure decay, no Hamiltonian.

Runs N = 2, 4, 8 qubits with bond_dim=8 and asserts no OOM/NaN errors.
Prints wall-clock time and max bond dimension.
"""

import time

import numpy as np
import pytest
from lindblad_ttn import LindbladTTN

GAMMA = 0.05
I2 = np.eye(2, dtype=complex)
sm = np.array([[0, 0], [1, 0]], dtype=complex)


def kron_list(ops):
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


@pytest.mark.parametrize("n_sites", [2, 4, 8])
def test_scaling_no_nan(n_sites):
    """No OOM or NaN for N-qubit decay with bond_dim=8."""
    d = 2 ** n_sites
    L_ops = []
    for i in range(n_sites):
        ops = [I2] * n_sites
        ops[i] = sm
        L_ops.append((GAMMA, kron_list(ops)))

    solver = LindbladTTN(
        H0=None,
        f=None,
        V=None,
        L_ops=L_ops,
        n_sites=n_sites,
        bond_dim=8,
        topology="train",
        device="cpu",
        strategy="ps1",
    )

    rho0 = np.eye(d, dtype=complex) / d

    t0 = time.perf_counter()
    result = solver.run(
        rho0=rho0,
        t_span=(0.0, 1.0),
        dt=0.1,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nN={n_sites}: {elapsed:.2f}s, max_bond={result.bond_dims[-1]}")

    # No NaN / Inf in final state
    assert not np.any(np.isnan(result.rho_final)), "NaN in rho_final"
    assert not np.any(np.isinf(result.rho_final)), "Inf in rho_final"

    # Norm roughly preserved
    assert abs(result.norm[-1] - 1.0) < 0.5, f"Norm too far from 1: {result.norm[-1]}"
