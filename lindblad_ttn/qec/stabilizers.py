# coding: utf-8
"""Stabilizer formalism (M8).

A stabilizer code on N qubits is defined by a list of commuting Pauli
strings (the *stabilizer generators*) plus the logical Pauli operators.
Syndrome extraction measures each generator, returning a binary outcome
that drives the decoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Pauli strings as sparse representations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PauliString:
    """Tensor product of single-qubit Paulis on a subset of N qubits.

    Stored as a sparse dict ``support[qubit_index] ∈ {'X', 'Y', 'Z'}``.
    Identity factors are NOT stored.

    Examples
    --------
    >>> P = PauliString(n=4, support={0: 'X', 2: 'Z'})
    >>> str(P)
    'X_0 Z_2'
    """

    n: int
    support: dict[int, str] = field(default_factory=dict)
    sign: int = 1   # ±1

    def commutes(self, other: "PauliString") -> bool:
        """True iff the two Pauli strings commute."""
        anti = 0
        for q in set(self.support) & set(other.support):
            if self.support[q] != other.support[q]:
                anti += 1
        return anti % 2 == 0

    def weight(self) -> int:
        return len(self.support)

    def as_dense(self) -> np.ndarray:
        """Materialise as a ``(2^n, 2^n)`` matrix.  Use sparingly — exponential."""
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        ops = {"X": X, "Y": Y, "Z": Z}
        mats = [ops.get(self.support.get(q, "I"), I) for q in range(self.n)]
        result = mats[0]
        for M in mats[1:]:
            result = np.kron(result, M)
        return float(self.sign) * result

    def __str__(self) -> str:
        if not self.support:
            return f"+I" if self.sign > 0 else "−I"
        parts = [f"{self.support[q]}_{q}" for q in sorted(self.support)]
        return f"{'+' if self.sign > 0 else '−'}{' '.join(parts)}"


# ---------------------------------------------------------------------------
# Generic stabilizer-code base class
# ---------------------------------------------------------------------------

class StabilizerCode:
    """Base class for ``[[n, k, d]]`` stabilizer codes.

    Subclasses populate ``self._generators``, ``self._logical_X``,
    ``self._logical_Z`` in their constructor.

    Attributes
    ----------
    n_data : int
        Number of physical data qubits.
    n_ancilla : int
        Number of ancilla qubits used during a syndrome round.
    distance : int
        Code distance ``d`` (minimum weight of a non-trivial logical).
    n_logical : int
        Number of logical qubits ``k`` encoded by the code.
    """

    n_data: int
    n_ancilla: int
    distance: int
    n_logical: int

    def __init__(self) -> None:
        self._generators: list[PauliString] = []
        self._logical_X: list[PauliString] = []
        self._logical_Z: list[PauliString] = []

    def generators(self) -> list[PauliString]:
        return list(self._generators)

    def logical(self, label: str, k: int = 0) -> PauliString:
        """Return the ``k``-th logical Pauli ``label ∈ {'X', 'Z'}``."""
        if label.upper() == "X":
            return self._logical_X[k]
        if label.upper() == "Z":
            return self._logical_Z[k]
        raise ValueError(f"Unknown logical label: {label!r}")

    def verify_commutation(self) -> None:
        """Assert that all stabilizers pairwise commute."""
        for i, gi in enumerate(self._generators):
            for j, gj in enumerate(self._generators[i + 1:], i + 1):
                assert gi.commutes(gj), \
                    f"Generators {i} and {j} do not commute:\n  {gi}\n  {gj}"

    def logical_pauli_dense(self, label: str, k: int = 0) -> np.ndarray:
        return self.logical(label, k).as_dense()

    # ------------------------------------------------------------------
    # Syndrome circuit scaffolding (returns abstract gate list).
    # ------------------------------------------------------------------

    def syndrome_circuit(self) -> list[tuple[str, tuple[int, ...]]]:
        """Return an abstract list of gates implementing one syndrome round.

        Each gate is ``(name, qubits)`` with ``name ∈ {'H', 'CNOT', 'MZ', 'RESET'}``.
        Concrete compilation to Hamiltonian pulses lives in the M6 control
        layer; this method gives the abstract circuit Stim can simulate.
        """
        gates: list[tuple[str, tuple[int, ...]]] = []
        n_data = self.n_data
        for k, P in enumerate(self._generators):
            ancilla = n_data + k
            gates.append(("RESET", (ancilla,)))
            # Z-type stabilizer → CNOTs from data to ancilla.
            # X-type stabilizer → Hadamards on ancilla + CNOTs from ancilla to data.
            kinds = set(P.support.values())
            if kinds <= {"Z"}:
                for q in sorted(P.support):
                    gates.append(("CNOT", (q, ancilla)))
                gates.append(("MZ", (ancilla,)))
            elif kinds <= {"X"}:
                gates.append(("H", (ancilla,)))
                for q in sorted(P.support):
                    gates.append(("CNOT", (ancilla, q)))
                gates.append(("H", (ancilla,)))
                gates.append(("MZ", (ancilla,)))
            else:
                # Mixed XYZ — fall back to a generic per-qubit-basis circuit.
                gates.append(("H", (ancilla,)))
                for q in sorted(P.support):
                    op = P.support[q]
                    if op == "X":
                        gates.append(("CNOT", (ancilla, q)))
                    elif op == "Z":
                        gates.append(("CZ", (ancilla, q)))
                    elif op == "Y":
                        gates.append(("CY", (ancilla, q)))
                gates.append(("H", (ancilla,)))
                gates.append(("MZ", (ancilla,)))
        return gates


# ---------------------------------------------------------------------------
# Concrete: N-qubit Z repetition code
# ---------------------------------------------------------------------------

class RepetitionCode(StabilizerCode):
    """N-qubit ``[N, 1, N]`` Z-repetition code.

    Encodes 1 logical qubit, protects against bit-flips.
    Stabilizers: ``Z_i Z_{i+1}`` for ``i = 0, ..., N-2``.
    Logical operators: ``L̄_X = X_0 X_1 ... X_{N-1}``, ``L̄_Z = Z_0``.
    """

    def __init__(self, N: int = 3) -> None:
        super().__init__()
        if N < 2:
            raise ValueError(f"N must be ≥ 2, got {N}.")
        self.n_data = N
        self.n_ancilla = N - 1
        self.distance = N
        self.n_logical = 1
        for i in range(N - 1):
            self._generators.append(PauliString(N, {i: "Z", i + 1: "Z"}))
        self._logical_X = [PauliString(N, {i: "X" for i in range(N)})]
        self._logical_Z = [PauliString(N, {0: "Z"})]
