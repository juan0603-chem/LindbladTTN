# coding: utf-8
"""LindbladTTN — Lindblad master equation solver using Tree Tensor Networks.

Public API
----------
* :class:`LindbladTTN` — the main solver.
* :class:`LindbladResult` — return type for :meth:`LindbladTTN.run`.
* :mod:`lindblad_ttn.sites` — Site classes (spin-1/2, spin-S, boson).
* :mod:`lindblad_ttn.templates` — pre-built Hamiltonians (transmon, JC, …).
* :mod:`lindblad_ttn.effective` — Schrieffer–Wolff and Magnus tools.
* :mod:`lindblad_ttn.control` — pulse-shape library.
* :mod:`lindblad_ttn.qec` — stabilizer codes and decoders.
"""

from lindblad_ttn.solver import LindbladResult, LindbladTTN

__all__ = ["LindbladTTN", "LindbladResult"]
__version__ = "0.3.0"
