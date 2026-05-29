# coding: utf-8
"""Utilities for time-dependent Hamiltonians.

A time-dependent Hamiltonian of the form

    H(t) = H₀ + Σᵢ fᵢ(t) · Vᵢ

is supported through :class:`TimeDependentSoP`. At every timestep the
instantaneous Liouvillian is rebuilt as ``L₀ + Σᵢ fᵢ(t)·L_Vᵢ`` — this
mirrors the pytenso ``f_list`` pattern (a list of independent time-dependent
SoP terms evaluated together), letting one solver run a multi-pulse circuit
in a single ``run()`` call.
"""

from __future__ import annotations

from typing import Callable, Sequence

from lindblad_ttn.core.sop import SumOfProducts


class TimeDependentSoP:
    """Wraps a static SoP and zero-or-more time-dependent drive SoPs.

    ``evaluate(t)`` returns the instantaneous Liouvillian
    ``L(t) = L₀ + Σᵢ fᵢ(t)·L_Vᵢ``.

    Parameters
    ----------
    sop_static : SumOfProducts
        The time-independent part (includes dissipators).
    drives : sequence of (callable, SumOfProducts), optional
        Each pair (fᵢ, L_Vᵢ) is one independent time-dependent channel.
        Use this for multi-pulse circuits.
    sop_drive : SumOfProducts, optional
        Legacy single-drive form (kept for backward compatibility). Mutually
        exclusive with ``drives``. Promoted internally to ``drives=[(f, sop_drive)]``.
    f : callable, optional
        Legacy single-drive envelope. Required iff ``sop_drive`` is given.
    """

    def __init__(
        self,
        sop_static: SumOfProducts,
        sop_drive: SumOfProducts | None = None,
        f: Callable[[float], float] | None = None,
        drives: Sequence[tuple[Callable[[float], float], SumOfProducts]] | None = None,
    ) -> None:
        if drives is not None and (sop_drive is not None or f is not None):
            raise ValueError(
                "Pass either 'drives' or the (f, sop_drive) pair, not both."
            )
        if sop_drive is not None and f is None:
            raise ValueError("sop_drive provided but f is None.")
        if f is not None and sop_drive is None:
            raise ValueError("f provided but sop_drive is None.")

        self._sop_static = sop_static

        if drives is not None:
            self._drives: list[tuple[Callable[[float], float], SumOfProducts]] = [
                (fn, sop) for (fn, sop) in drives
            ]
        elif sop_drive is not None and f is not None:
            self._drives = [(f, sop_drive)]
        else:
            self._drives = []

    @property
    def is_time_dependent(self) -> bool:
        """True if any time-dependent drive is registered."""
        return len(self._drives) > 0

    @property
    def n_drives(self) -> int:
        return len(self._drives)

    def evaluate(self, t: float) -> SumOfProducts:
        """Return the Liouvillian SoP at time ``t``.

        ``O(n_drives · n_terms_drive)`` — concatenates the static term list
        with each drive's terms scaled by ``fᵢ(t)``. Drives whose envelope
        falls below ``1e-14`` at ``t`` are skipped (matches pytenso heom.py).
        """
        if not self._drives:
            return self._sop_static
        sop = self._sop_static
        for f_i, sop_v_i in self._drives:
            f_val = float(f_i(t))
            if abs(f_val) < 1e-14:
                continue
            sop = sop + (f_val * sop_v_i)
        return sop

    def evaluate_midpoint(self, t: float, dt: float) -> SumOfProducts:
        """Return the Liouvillian SoP at the midpoint ``t + dt/2``.

        Midpoint rule → second-order accuracy in time.
        """
        return self.evaluate(t + dt / 2.0)
