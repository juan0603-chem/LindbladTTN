# coding: utf-8
"""Benchmark and validation suite for M1–M9.

Run with::

    pytest benchmarks/ -v

Each test compares a LindbladTTN simulation to QuTiP (for dynamics) or
to ``scipy.linalg`` (for static spectra), within a documented tolerance.
"""
