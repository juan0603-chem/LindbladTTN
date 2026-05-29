# coding: utf-8
"""Validation of QEC toolkit (M8): stabilizer code commutation & basics."""

from __future__ import annotations

import numpy as np
import pytest

from lindblad_ttn.qec import (
    ColorCode488, PauliString, RepetitionCode, SurfaceCode,
)


def test_repetition_code_3():
    """3-qubit repetition code: 2 generators, distance 3, transversal X̄."""
    code = RepetitionCode(N=3)
    code.verify_commutation()
    assert code.n_data == 3
    assert code.distance == 3
    assert len(code.generators()) == 2
    assert code.logical("X").weight() == 3
    assert code.logical("Z").weight() == 1


def test_steane_code_basics():
    """Steane / 4.8.8 colour-code: 6 generators, distance 3."""
    code = ColorCode488()
    code.verify_commutation()
    assert code.n_data == 7
    assert code.distance == 3
    assert len(code.generators()) == 6
    # Logical X̄ and Z̄ both weight-7
    assert code.logical("X").weight() == 7
    assert code.logical("Z").weight() == 7


def test_surface_code_distance_3():
    """Distance-3 rotated surface code: stabilizers all commute."""
    code = SurfaceCode(distance=3)
    code.verify_commutation()
    assert code.n_data == 9
    assert code.distance == 3
    # Logical Z̄ along a column of 3 data qubits
    assert code.logical("Z").weight() == 3
    assert code.logical("X").weight() == 3


def test_pauli_string_commutes():
    """Sanity check of PauliString.commutes()."""
    n = 4
    Z01 = PauliString(n, {0: "Z", 1: "Z"})
    Z12 = PauliString(n, {1: "Z", 2: "Z"})
    X01 = PauliString(n, {0: "X", 1: "X"})
    assert Z01.commutes(Z12)   # share Z on qubit 1 (same)
    assert Z01.commutes(X01)   # overlap on 0 and 1 — 2 anti-commutes → commute


def test_syndrome_circuit_shape():
    """Repetition-code syndrome circuit has reset, CNOTs, and measurement."""
    code = RepetitionCode(N=3)
    gates = code.syndrome_circuit()
    names = [g[0] for g in gates]
    assert "RESET" in names
    assert "CNOT" in names
    assert "MZ" in names
