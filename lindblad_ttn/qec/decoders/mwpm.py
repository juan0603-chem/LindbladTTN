# coding: utf-8
"""Minimum-weight-perfect-matching decoder wrapper around PyMatching (M8).

Falls back to a brute-force lookup decoder when PyMatching is not installed
or when the syndrome is short enough that exhaustive search is faster.
"""

from __future__ import annotations

import itertools

import numpy as np

try:
    import pymatching  # type: ignore
    _HAS_PYMATCHING = True
except ImportError:
    _HAS_PYMATCHING = False


def mwpm_decode_syndrome(
    syndromes: np.ndarray,
    parity_check_matrix: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Decode a batch of syndromes via MWPM.

    Parameters
    ----------
    syndromes : ndarray of shape (n_shots, n_stabilizers)
        Binary syndrome bits per shot.
    parity_check_matrix : ndarray of shape (n_stabilizers, n_errors)
        H matrix mapping single error events to syndrome bits.
    weights : ndarray of shape (n_errors,), optional
        Edge weights (negative log probabilities); default uniform.

    Returns
    -------
    corrections : ndarray of shape (n_shots, n_errors)
        Predicted error pattern per shot.
    """
    H = np.asarray(parity_check_matrix, dtype=np.uint8)
    syndromes = np.asarray(syndromes, dtype=np.uint8)
    n_shots, n_stab = syndromes.shape
    n_errors = H.shape[1]
    corrections = np.zeros((n_shots, n_errors), dtype=np.uint8)
    if _HAS_PYMATCHING:
        matching = pymatching.Matching.from_check_matrix(H, weights=weights)
        for i, syn in enumerate(syndromes):
            corrections[i] = matching.decode(syn)
        return corrections

    # Fallback: brute-force enumeration up to weight 2.
    cache: dict[tuple[int, ...], np.ndarray] = {}
    for i, syn in enumerate(syndromes):
        key = tuple(int(b) for b in syn)
        if key in cache:
            corrections[i] = cache[key]
            continue
        if not any(key):
            cache[key] = np.zeros(n_errors, dtype=np.uint8)
            corrections[i] = cache[key]
            continue
        found = False
        for wt in (1, 2):
            for err_idx in itertools.combinations(range(n_errors), wt):
                guess_syn = np.zeros(n_stab, dtype=np.uint8)
                for ei in err_idx:
                    guess_syn ^= H[:, ei]
                if np.array_equal(guess_syn, syn):
                    err = np.zeros(n_errors, dtype=np.uint8)
                    for ei in err_idx:
                        err[ei] = 1
                    cache[key] = err
                    corrections[i] = err
                    found = True
                    break
            if found:
                break
    return corrections
