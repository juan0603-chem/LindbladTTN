# coding: utf-8
"""Within-simulation parallelism + GPU ergonomics.

Covers:
  • parallel_nodes=True must match the serial path bitwise (math unchanged)
  • dtype='complex64' runs without exceptions and preserves trace
  • GPU smoke test — only runs when CUDA is available
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lindblad_ttn import LindbladTTN


# Pauli matrices
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)
sm = np.array([[0, 0], [1, 0]], dtype=complex)


def _kron(*ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def _build_single_qubit(parallel: bool) -> LindbladTTN:
    return LindbladTTN(
        H0=0.5 * Z, f=None, V=None,
        L_ops=[(0.3, sm)],
        n_sites=1, bond_dim=4,
        topology="train", device="cpu", strategy="vmf",
        parallel_nodes=parallel,
    )


def _build_two_qubit(parallel: bool) -> LindbladTTN:
    # NOTE — topology="train" rather than "tree". The 2-site tree topology
    # in this package has a frame-factory shape bug; the linear train works.
    return LindbladTTN(
        H0=0.5 * _kron(Z, I2) + 0.1 * _kron(X, X),
        f=None, V=None,
        L_ops=[(0.1, _kron(sm, I2)), (0.1, _kron(I2, sm))],
        n_sites=2, bond_dim=8,
        topology="train", device="cpu", strategy="vmf",
        parallel_nodes=parallel,
    )


@pytest.mark.parametrize("builder", [_build_single_qubit, _build_two_qubit])
def test_vmf_parallel_matches_serial(builder):
    """parallel_nodes=True must produce numerically identical results to serial.

    The math is unchanged — no reduction-order rearrangement — so even with
    ODE adaptive stepping the trajectories must agree to ~1e-12.
    """
    solver_s = builder(parallel=False)
    solver_p = builder(parallel=True)

    n = solver_s.n_sites
    rho0 = np.eye(2 ** n, dtype=complex) / (2 ** n)

    res_s = solver_s.run(rho0=rho0, t_span=(0.0, 1.0), dt=0.05, verbose=False)
    res_p = solver_p.run(rho0=rho0, t_span=(0.0, 1.0), dt=0.05, verbose=False)

    assert np.allclose(res_s.rho_final, res_p.rho_final, atol=1e-10, rtol=0), (
        f"max diff = {np.max(np.abs(res_s.rho_final - res_p.rho_final)):.2e}"
    )


def test_dtype_complex64_runs():
    """dtype='complex64' completes without errors and preserves trace ~ 1.

    complex64 trades accuracy for speed (~3× on GPU); we only assert that the
    pipeline runs and the trace stays close to 1.
    """
    solver = LindbladTTN(
        H0=0.5 * Z, f=None, V=None,
        L_ops=[(0.3, sm)],
        n_sites=1, bond_dim=4,
        topology="train", device="cpu", strategy="ps1",
        dtype="complex64",
    )
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    res = solver.run(
        rho0=rho0, t_span=(0.0, 1.0), dt=0.05,
        observables=[], save_every=2, verbose=False,
    )
    assert np.all(np.abs(res.norm - 1.0) < 1e-4), (
        f"trace drift: {np.max(np.abs(res.norm - 1.0)):.2e}"
    )
    # restore default for any tests that share this process
    from lindblad_ttn.core.backend import set_dtype
    set_dtype(torch.complex128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_smoke():
    """Smoke test: run 10 steps on CUDA, check shape and trace preservation."""
    try:
        solver = LindbladTTN(
            H0=0.5 * Z, f=None, V=None,
            L_ops=[(0.3, sm)],
            n_sites=1, bond_dim=4,
            topology="train", device="cuda", strategy="ps1",
            dtype="complex64",
        )
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        res = solver.run(
            rho0=rho0, t_span=(0.0, 1.0), dt=0.1, verbose=False,
        )
        assert res.rho_final.shape == (2, 2)
        assert abs(float(np.trace(res.rho_final).real) - 1.0) < 1e-3
    finally:
        # restore globals so subsequent tests run on CPU/complex128
        from lindblad_ttn.core.backend import set_device, set_dtype
        set_device("cpu")
        set_dtype(torch.complex128)
