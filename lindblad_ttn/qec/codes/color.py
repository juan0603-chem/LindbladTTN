# coding: utf-8
"""4.8.8 colour code (M8) — minimal distance-3 instance.

Implemented as a textbook 7-qubit Steane code (the smallest colour code on
a 4.8.8 tiling collapses to Steane).  Transversal Cliffords work cleanly.

Data layout (Steane CSS):

    Stabilizers:
        S1  = X_0 X_2 X_4 X_6
        S2  = X_1 X_2 X_5 X_6
        S3  = X_3 X_4 X_5 X_6
        S4  = Z_0 Z_2 Z_4 Z_6
        S5  = Z_1 Z_2 Z_5 Z_6
        S6  = Z_3 Z_4 Z_5 Z_6

    Logical operators:
        L̄_X = X_0 X_1 X_2 X_3 X_4 X_5 X_6
        L̄_Z = Z_0 Z_1 Z_2 Z_3 Z_4 Z_5 Z_6
"""

from __future__ import annotations

from lindblad_ttn.qec.stabilizers import PauliString, StabilizerCode


class ColorCode488(StabilizerCode):
    """Distance-3 4.8.8 colour code (= Steane).

    7 data qubits, 6 ancillas, distance 3, transversal Clifford group.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_data = 7
        self.n_ancilla = 6
        self.distance = 3
        self.n_logical = 1
        n = 7
        x_stabs = [
            {0: "X", 2: "X", 4: "X", 6: "X"},
            {1: "X", 2: "X", 5: "X", 6: "X"},
            {3: "X", 4: "X", 5: "X", 6: "X"},
        ]
        z_stabs = [
            {0: "Z", 2: "Z", 4: "Z", 6: "Z"},
            {1: "Z", 2: "Z", 5: "Z", 6: "Z"},
            {3: "Z", 4: "Z", 5: "Z", 6: "Z"},
        ]
        for s in x_stabs:
            self._generators.append(PauliString(n, s))
        for s in z_stabs:
            self._generators.append(PauliString(n, s))
        self._logical_X = [PauliString(n, {i: "X" for i in range(n)})]
        self._logical_Z = [PauliString(n, {i: "Z" for i in range(n)})]
