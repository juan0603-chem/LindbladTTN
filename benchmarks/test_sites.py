# coding: utf-8
"""Validation of site operators (M2 boson, M3 spin)."""

from __future__ import annotations

import numpy as np
import pytest

from lindblad_ttn.sites import boson_site, spin_half_site, spin_site, stevens_operator


def test_boson_ladder_relations():
    """[a, a†] = 1 on the (truncated) Fock space, modulo the cutoff edge."""
    c = boson_site(10, "c")
    comm = c.a @ c.adag - c.adag @ c.a
    # Truncation breaks the relation at the top Fock state; check the bulk.
    bulk = comm[:9, :9]
    assert np.allclose(bulk, np.eye(9))


def test_boson_number_op():
    """n |n> = n |n> for the truncated Fock basis."""
    c = boson_site(6, "c")
    for n in range(6):
        psi = c.fock_state(n)
        val = float(np.real(psi.conj() @ c.n @ psi))
        assert abs(val - n) < 1e-10


def test_spin_commutator():
    """[Sx, Sy] = i Sz for S = 1/2, 1, 3/2, 2."""
    for S in (0.5, 1.0, 1.5, 2.0):
        s = spin_site(S, "s")
        comm = s.Sx @ s.Sy - s.Sy @ s.Sx
        assert np.allclose(comm, 1j * s.Sz), f"S={S}: [Sx,Sy] ≠ i Sz"


def test_spin_squared_eigenvalues():
    """S² has eigenvalue S(S+1) on every basis state."""
    for S in (0.5, 1.0, 1.5, 2.0, 3.5):
        s = spin_site(S, "s")
        S2 = s.S_squared()
        expected = S * (S + 1)
        eigs = np.linalg.eigvalsh(S2)
        assert np.allclose(eigs, expected), f"S={S}: S² eigvals = {eigs} ≠ {expected}"


def test_stevens_o20_eigenvalues():
    """O_2^0 = 3 S_z² − S(S+1) I is diagonal with eigenvalues 3m² − S(S+1)."""
    for S in (1.0, 2.0, 3.5):
        op = stevens_operator(2, 0, S)
        d = int(round(2 * S + 1))
        m_vals = np.array([S - i for i in range(d)])
        expected = 3 * m_vals ** 2 - S * (S + 1)
        eigs = np.sort(np.linalg.eigvalsh(op))
        expected_sorted = np.sort(expected)
        assert np.allclose(eigs, expected_sorted), \
            f"O_2^0 S={S} eigvals mismatch: {eigs} vs {expected_sorted}"


def test_spin_half_pauli_match():
    """SpinSite(S=1/2) S_x, S_y, S_z match the spin-1/2 Pauli/2."""
    s_half = spin_site(0.5, "q")
    sx_p = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy_p = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    sz_p = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    assert np.allclose(s_half.Sx, sx_p)
    assert np.allclose(s_half.Sy, sy_p)
    assert np.allclose(s_half.Sz, sz_p)
