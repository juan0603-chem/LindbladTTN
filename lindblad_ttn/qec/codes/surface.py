# coding: utf-8
"""Rotated surface code (M8).

Distance-d rotated surface code on a ``d × d`` lattice of data qubits.
Stabilizers are weight-4 plaquettes (with weight-2 edges along the
boundary).  Boundary-X plaquettes generate ``X_i X_j ... `` strings;
boundary-Z plaquettes generate ``Z_i Z_j ...``.

Layout for d=3 (Z plaquettes ●, X plaquettes ○)::

    0 - 1 - 2
    | ○ | ● |
    3 - 4 - 5
    | ● | ○ |
    6 - 7 - 8

Logical operators:
    L̄_Z = Z_0 · Z_3 · Z_6   (left column)
    L̄_X = X_0 · X_1 · X_2   (top row)
"""

from __future__ import annotations

from lindblad_ttn.qec.stabilizers import PauliString, StabilizerCode


class SurfaceCode(StabilizerCode):
    """Distance-``d`` rotated surface code with ``d² + (d² − 1)`` qubits."""

    def __init__(self, distance: int = 3) -> None:
        super().__init__()
        if distance < 2:
            raise ValueError(f"distance must be ≥ 2, got {distance}.")
        if distance % 2 == 0:
            # Even-distance rotated codes have asymmetric logical strings;
            # for simplicity we restrict to odd d.
            raise ValueError("Only odd distance supported in this implementation.")
        d = distance
        self.distance = d
        self.n_data = d * d
        # (d² − 1) ancillas, alternating X / Z plaquettes
        self.n_ancilla = d * d - 1
        self.n_logical = 1
        n = self.n_data

        # Generate plaquettes.  In a (d-1)×(d-1) grid of plaquettes, colouring
        # is checkerboard with X/Z alternating; corner plaquettes are weight-2.
        for ri in range(d - 1):
            for cj in range(d - 1):
                kind = "Z" if (ri + cj) % 2 == 0 else "X"
                qubits = [
                    ri * d + cj,
                    ri * d + cj + 1,
                    (ri + 1) * d + cj,
                    (ri + 1) * d + cj + 1,
                ]
                self._generators.append(
                    PauliString(n, {q: kind for q in qubits})
                )

        # Boundary plaquettes (weight-2)
        # Top edge: Z plaquettes between columns
        for cj in range(d - 1):
            if cj % 2 == 1:
                self._generators.append(
                    PauliString(n, {cj: "Z", cj + 1: "Z"})
                )
        # Bottom edge
        for cj in range(d - 1):
            if (d - 2 + cj) % 2 == 1:
                top = (d - 1) * d + cj
                self._generators.append(
                    PauliString(n, {top: "Z", top + 1: "Z"})
                )
        # Left edge: X
        for ri in range(d - 1):
            if ri % 2 == 0:
                self._generators.append(
                    PauliString(n, {ri * d: "X", (ri + 1) * d: "X"})
                )
        # Right edge: X
        for ri in range(d - 1):
            if (ri + d - 2) % 2 == 0:
                right = ri * d + d - 1
                self._generators.append(
                    PauliString(n, {right: "X", right + d: "X"})
                )

        # Logical operators
        self._logical_Z = [
            PauliString(n, {ri * d: "Z" for ri in range(d)})
        ]
        self._logical_X = [
            PauliString(n, {cj: "X" for cj in range(d)})
        ]
