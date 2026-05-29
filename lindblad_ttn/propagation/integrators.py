# coding: utf-8
"""Integration primitives for TDVP propagation.

Provides:
- :func:`krylov_expm_apply` — Krylov subspace matrix exponential
- :func:`torchdiffeq_step`  — single adaptive ODE step
- :func:`rk4_step`          — fixed-step RK4
"""

from __future__ import annotations

from typing import Callable

import torch

from lindblad_ttn.core.backend import DEVICE, DTYPE, expm_krylov, opt_odeint


def krylov_expm_apply(
    H_matvec: Callable[[torch.Tensor], torch.Tensor],
    v: torch.Tensor,
    dt: float,
    krylov_dim: int = 20,
) -> torch.Tensor:
    """Apply ``exp(H * dt)`` to ``v`` via the Arnoldi/Krylov subspace method.

    Parameters
    ----------
    H_matvec : callable
        Computes ``H @ v`` without forming ``H`` explicitly.
    v : torch.Tensor
        Input vector of shape ``(n,)``.
    dt : float
        Exponent parameter (negative for backward integration).
    krylov_dim : int
        Krylov subspace dimension.

    Returns
    -------
    torch.Tensor
        ``exp(H * dt) @ v``, shape ``(n,)``.
    """
    return expm_krylov(H_matvec, v, dt, krylov_dim=krylov_dim)


def krylov_expm_apply_dense(
    H: torch.Tensor,
    v: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Apply ``exp(H * dt)`` using a dense matrix exponential.

    Parameters
    ----------
    H : torch.Tensor
        Dense matrix, shape ``(n, n)``.
    v : torch.Tensor
        Vector, shape ``(n,)``.
    dt : float

    Returns
    -------
    torch.Tensor
        Shape ``(n,)``.
    """
    return torch.linalg.matrix_exp(H * dt) @ v


def torchdiffeq_step(
    f: Callable[[float, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t0: float,
    dt: float,
    method: str = "dopri5",
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> torch.Tensor:
    """Single adaptive ODE step using torchdiffeq.

    Parameters
    ----------
    f : callable
        RHS function ``f(t, y) -> dy/dt``.
    y0 : torch.Tensor
        Initial state.
    t0 : float
        Initial time.
    dt : float
        Step size.
    method : str
        Integration method (``'dopri5'``, ``'dopri8'``, ``'bosh3'``, etc.).
    atol : float
    rtol : float

    Returns
    -------
    torch.Tensor
        State at ``t0 + dt``.
    """
    return opt_odeint(f, y0, t0, dt, method=method, atol=atol, rtol=rtol)


def rk4_step(
    f: Callable[[float, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t0: float,
    dt: float,
) -> torch.Tensor:
    """Fixed-step RK4 using the 3/8 rule.

    Parameters
    ----------
    f : callable
        RHS function.
    y0 : torch.Tensor
    t0 : float
    dt : float

    Returns
    -------
    torch.Tensor
    """
    k1 = f(t0, y0) * dt
    k2 = f(t0 + dt / 3.0, y0 + k1 / 3.0) * dt
    k3 = f(t0 + dt * 2.0 / 3.0, y0 - k1 / 3.0 + k2) * dt
    k4 = f(t0 + dt, y0 + k1 - k2 + k3) * dt
    return y0 + (k1 + 3.0 * k2 + 3.0 * k3 + k4) / 8.0
