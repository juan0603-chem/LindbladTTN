# coding: utf-8
"""Quantum-error-correction toolkit (M8).

Stabilizer codes implemented as classes that know:

* The Pauli-string generators for every stabilizer.
* The logical Pauli operators ``L̄_X``, ``L̄_Z``.
* The list of data qubits and ancilla qubits.
* How to compile a single syndrome-extraction round into ``H_terms`` /
  drive sequences (only sketched here — the full compile-to-pulse code
  belongs in M6/M4).

Concrete codes
--------------
* :class:`SurfaceCode` — rotated planar surface code, distance ``d``.
* :class:`RepetitionCode` — N-qubit Z (or X) repetition code (minimal sanity
  check / pedagogical example).
* :class:`ColorCode488` — distance-3 4.8.8 colour code (12-qubit example).

Decoders are provided via :mod:`lindblad_ttn.qec.decoders`.
"""

from lindblad_ttn.qec.stabilizers import (
    PauliString,
    StabilizerCode,
    RepetitionCode,
)
from lindblad_ttn.qec.codes.surface import SurfaceCode
from lindblad_ttn.qec.codes.color import ColorCode488

__all__ = [
    "PauliString",
    "StabilizerCode",
    "RepetitionCode",
    "SurfaceCode",
    "ColorCode488",
]
