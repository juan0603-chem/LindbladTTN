# coding: utf-8
"""Backend for tensor operations.

Manages device/dtype globals and provides utility functions for tensor
arithmetic, SVD truncation, Krylov exponentiation, and ODE integration.

All tensor operations use ``torch.complex128`` by default.  Autograd is
disabled globally at import time.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable

import numpy as np
import torch
import torchdiffeq

torch.set_grad_enabled(False)

_SVD_CPU_FALLBACK_WARNED = False


def _svd_on_device(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD that stays on ``matrix.device`` when possible.

    Falls back to CPU SVD with a one-time warning for older torch builds where
    complex SVD is not supported on CUDA.
    """
    global _SVD_CPU_FALLBACK_WARNED
    try:
        return torch.linalg.svd(matrix, full_matrices=False)
    except RuntimeError:
        if not _SVD_CPU_FALLBACK_WARNED:
            warnings.warn(
                "torch.linalg.svd failed on device %r; falling back to CPU SVD. "
                "This is expected on older torch builds for complex tensors on "
                "CUDA. Performance will be degraded." % str(matrix.device),
                RuntimeWarning,
                stacklevel=2,
            )
            _SVD_CPU_FALLBACK_WARNED = True
        cpu_mat = matrix.cpu()
        U, S, Vh = torch.linalg.svd(cpu_mat, full_matrices=False)
        device = matrix.device
        return U.to(device), S.to(device), Vh.to(device)

# ---------------------------------------------------------------------------
# Global device / dtype state
# ---------------------------------------------------------------------------

DEVICE: str = "cpu"
DTYPE: torch.dtype = torch.complex128


def _propagate_to_importers(name: str, value) -> None:
    """Rebind a global in every module that has imported it from this one.

    Several submodules import ``DTYPE`` / ``DEVICE`` directly with
    ``from lindblad_ttn.core.backend import DTYPE``. That captures the value
    at import time, so a later ``set_dtype`` only updates *this* module's
    binding. We walk ``sys.modules`` and patch each stale reference so the
    runtime stays consistent.
    """
    import sys
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        mod_name = getattr(mod, "__name__", "")
        if not mod_name.startswith("lindblad_ttn"):
            continue
        if hasattr(mod, name):
            try:
                setattr(mod, name, value)
            except Exception:
                pass


def set_device(device: str) -> None:
    """Set the global compute device.

    Parameters
    ----------
    device : str
        PyTorch device string, e.g. ``'cpu'``, ``'cuda'``, ``'cuda:0'``.
    """
    global DEVICE
    DEVICE = device
    _propagate_to_importers("DEVICE", device)


def set_dtype(dtype: torch.dtype) -> None:
    """Set the global tensor dtype.

    Parameters
    ----------
    dtype : torch.dtype
        Desired dtype.  Recommended: ``torch.complex128`` (default) or
        ``torch.complex64`` for GPU runs.
    """
    global DTYPE
    DTYPE = dtype
    _propagate_to_importers("DTYPE", dtype)


# ---------------------------------------------------------------------------
# Array conversion helpers
# ---------------------------------------------------------------------------

def to_torch(array) -> torch.Tensor:
    """Convert a numpy array or scipy sparse matrix to a complex torch tensor.

    Parameters
    ----------
    array : array-like or scipy.sparse matrix
        Input data.

    Returns
    -------
    torch.Tensor
        Dense complex tensor on the global device with the global dtype.
    """
    import scipy.sparse as sp
    if sp.issparse(array):
        array = array.toarray()
    arr = np.asarray(array)
    return torch.tensor(arr, dtype=DTYPE, device=DEVICE)


def zeros(*shape: int) -> torch.Tensor:
    """Return a zero tensor with the global dtype and device.

    Parameters
    ----------
    *shape : int
        Dimensions of the tensor.

    Returns
    -------
    torch.Tensor
    """
    return torch.zeros(shape, dtype=DTYPE, device=DEVICE)


def eye(n: int, m: int | None = None) -> torch.Tensor:
    """Return an identity matrix with the global dtype and device.

    Parameters
    ----------
    n : int
        Number of rows.
    m : int, optional
        Number of columns (default: ``n``).

    Returns
    -------
    torch.Tensor
    """
    if m is None:
        m = n
    return torch.eye(n, m, dtype=DTYPE, device=DEVICE)


# ---------------------------------------------------------------------------
# Tensor product
# ---------------------------------------------------------------------------

def kron_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Kronecker (tensor) product of two matrices.

    Parameters
    ----------
    A : torch.Tensor
        Shape ``(m, n)``.
    B : torch.Tensor
        Shape ``(p, q)``.

    Returns
    -------
    torch.Tensor
        Shape ``(m*p, n*q)``.
    """
    m, n = A.shape
    p, q = B.shape
    return (A.unsqueeze(1).unsqueeze(3) * B.unsqueeze(0).unsqueeze(2)).reshape(m * p, n * q)


# ---------------------------------------------------------------------------
# SVD with optional truncation
# ---------------------------------------------------------------------------

def svd_truncate(
    matrix: torch.Tensor,
    max_rank: int | None = None,
    atol: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """SVD of a matrix with optional rank truncation.

    Truncation keeps the largest singular values. When both ``max_rank`` and
    ``atol`` are provided, the stricter criterion is applied.

    Parameters
    ----------
    matrix : torch.Tensor
        2-D input matrix of shape ``(m, n)``.
    max_rank : int, optional
        Maximum number of singular values to keep.
    atol : float
        Absolute tolerance: discard singular values smaller than ``atol``.

    Returns
    -------
    U : torch.Tensor
        Left singular vectors, shape ``(m, r)``.
    S : torch.Tensor
        Singular values, shape ``(r,)``.
    Vh : torch.Tensor
        Right singular vectors, shape ``(r, n)``.
    truncation_error : float
        Frobenius norm of the discarded part.
    """
    U, S, Vh = _svd_on_device(matrix)

    # Determine truncation rank
    rank = S.shape[0]
    if atol > 0.0:
        # keep values strictly above atol
        keep = int((S > atol).sum().item())
        rank = max(1, keep)
    if max_rank is not None:
        rank = min(rank, max_rank)

    trunc_err = float(torch.norm(S[rank:]).item()) if rank < S.shape[0] else 0.0
    return U[:, :rank], S[:rank], Vh[:rank, :], trunc_err


# ---------------------------------------------------------------------------
# Krylov subspace exponentiation
# ---------------------------------------------------------------------------

def expm_krylov(
    H_matvec: Callable[[torch.Tensor], torch.Tensor],
    v: torch.Tensor,
    dt: float,
    krylov_dim: int = 20,
) -> torch.Tensor:
    """Apply ``exp(H * dt)`` to vector ``v`` via the Arnoldi/Krylov method.

    For small dimensions (``len(v) <= krylov_dim``), falls back to
    ``torch.linalg.matrix_exp``.

    Parameters
    ----------
    H_matvec : callable
        Function ``v -> H @ v``.  ``H`` need not be explicitly formed.
    v : torch.Tensor
        Input vector, shape ``(n,)``.
    dt : float
        Time step (can be negative for backward integration).
    krylov_dim : int
        Maximum Krylov subspace dimension.

    Returns
    -------
    torch.Tensor
        ``exp(H * dt) @ v``, shape ``(n,)``.

    Notes
    -----
    Uses the Arnoldi iteration to build an orthonormal basis ``Q`` for the
    Krylov subspace ``{v, Hv, H²v, …}``, then exponentiates the small
    projected matrix ``H_K = Qᴴ H Q`` exactly.
    """
    n = v.shape[0]
    kdim = min(krylov_dim, n)

    if n <= krylov_dim:
        # Build H explicitly and use matrix_exp
        cols = []
        basis = eye(n)
        for i in range(n):
            cols.append(H_matvec(basis[:, i]))
        H_dense = torch.stack(cols, dim=1)
        return torch.linalg.matrix_exp(H_dense * dt) @ v

    # Arnoldi iteration
    dtype = v.dtype
    Q = torch.zeros(n, kdim + 1, dtype=dtype, device=DEVICE)
    H_k = torch.zeros(kdim + 1, kdim, dtype=dtype, device=DEVICE)

    norm_v = torch.norm(v)
    if norm_v < 1e-30:
        return v.clone()
    Q[:, 0] = v / norm_v

    breakdown = kdim
    for j in range(kdim):
        w = H_matvec(Q[:, j])
        for i in range(j + 1):
            H_k[i, j] = torch.dot(Q[:, i].conj(), w)
            w = w - H_k[i, j] * Q[:, i]
        h_next = torch.norm(w)
        H_k[j + 1, j] = h_next
        if h_next < 1e-12:
            breakdown = j + 1
            break
        if j + 1 < kdim:
            Q[:, j + 1] = w / h_next

    k = breakdown
    H_small = H_k[:k, :k]
    e1 = torch.zeros(k, dtype=dtype, device=DEVICE)
    e1[0] = norm_v
    exp_H_small = torch.linalg.matrix_exp(H_small * dt)
    coeffs = exp_H_small @ e1
    return Q[:, :k] @ coeffs


# ---------------------------------------------------------------------------
# ODE integration (mirrors tenso's opt_odeint with complex trick)
# ---------------------------------------------------------------------------

def opt_odeint(
    func: Callable[[float, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t0: float,
    dt: float,
    method: str = "dopri5",
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> torch.Tensor:
    """Integrate a complex-valued ODE by one step from ``t0`` to ``t0 + dt``.

    Complex tensors are split into ``(real, imag)`` pairs before calling
    ``torchdiffeq.odeint``, then recombined.  This mirrors the trick used in
    tenso's backend.

    Parameters
    ----------
    func : callable
        RHS of the ODE: ``dy/dt = func(t, y)``.  Must return a tensor with
        the same shape as ``y``.
    y0 : torch.Tensor
        Initial state (complex or real).
    t0 : float
        Initial time.
    dt : float
        Step size (positive for forward, negative for backward).
    method : str
        Integration method.  Supported:
        ``'dopri5'`` (default), ``'dopri8'``, ``'bosh3'``,
        ``'adaptive_heun'``, ``'euler'``, ``'midpoint'``,
        ``'rk4'``, ``'iter{N}'`` (Taylor up to order N).
    atol : float
        Absolute tolerance (adaptive methods only).
    rtol : float
        Relative tolerance (adaptive methods only).

    Returns
    -------
    torch.Tensor
        State at ``t0 + dt``, same shape as ``y0``.
    """
    is_complex = y0.is_complex()

    if method == "rk4":
        k1 = func(t0, y0) * dt
        k2 = func(t0 + dt / 3.0, y0 + k1 / 3.0) * dt
        k3 = func(t0 + dt * 2.0 / 3.0, y0 - k1 / 3.0 + k2) * dt
        k4 = func(t0 + dt, y0 + k1 - k2 + k3) * dt
        return y0 + (k1 + 3.0 * k2 + 3.0 * k3 + k4) / 8.0

    if method.startswith("iter"):
        order = int(method[4:])
        result = y0.clone()
        yn = y0.clone()
        for n in range(1, order + 1):
            yn = func(t0, yn) * dt / n
            result = result + yn
        return result

    # torchdiffeq path — requires real-valued tensors
    if is_complex:
        y0_re = y0.real
        y0_im = y0.imag
        y0_real = torch.cat([y0_re.flatten(), y0_im.flatten()])
        shape = y0.shape
        size = y0.numel()

        def real_func(t: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
            t_val = float(t.item())
            y_c = torch.complex(
                y_real[:size].reshape(shape),
                y_real[size:].reshape(shape),
            )
            dy = func(t_val, y_c)
            return torch.cat([dy.real.flatten(), dy.imag.flatten()])

        t_span = torch.tensor([t0, t0 + dt], dtype=torch.float64, device=DEVICE)
        sol = torchdiffeq.odeint(
            real_func, y0_real, t_span, method=method, atol=atol, rtol=rtol
        )
        y1_real = sol[1]
        return torch.complex(
            y1_real[:size].reshape(shape),
            y1_real[size:].reshape(shape),
        )
    else:
        t_span = torch.tensor([t0, t0 + dt], dtype=torch.float64, device=DEVICE)
        sol = torchdiffeq.odeint(
            lambda t, y: func(float(t.item()), y),
            y0,
            t_span,
            method=method,
            atol=atol,
            rtol=rtol,
        )
        return sol[1]


# ---------------------------------------------------------------------------
# Contraction helpers (mirrors tenso opt_transform / opt_multitransform)
# ---------------------------------------------------------------------------

def transform(op: torch.Tensor, tensor: torch.Tensor,
              op_ax: int, tensor_ax: int) -> torch.Tensor:
    """Contract ``op`` with ``tensor`` along one axis, then move result back.

    Parameters
    ----------
    op : torch.Tensor
        Operator matrix, shape ``(a, b)``.  Axis ``op_ax`` is contracted.
    tensor : torch.Tensor
        Target tensor.
    op_ax : int
        Which axis of ``op`` to contract (typically 1 for column contraction).
    tensor_ax : int
        Which axis of ``tensor`` to contract.

    Returns
    -------
    torch.Tensor
        Same shape as ``tensor`` but with the contracted axis replaced by the
        free axis of ``op``.
    """
    dotted = torch.tensordot(tensor, op, dims=([tensor_ax], [op_ax]))
    return dotted.movedim(-1, tensor_ax)


def multitransform(op_dict: dict[int, torch.Tensor],
                   tensor: torch.Tensor) -> torch.Tensor:
    """Apply a sequence of single-axis transforms.

    Parameters
    ----------
    op_dict : dict[int, torch.Tensor]
        Mapping from tensor axis to operator matrix.
    tensor : torch.Tensor
        Target tensor.

    Returns
    -------
    torch.Tensor
        Result after applying all operators.
    """
    result = tensor
    for ax, mat in op_dict.items():
        result = transform(mat, result, 1, ax)
    return result
