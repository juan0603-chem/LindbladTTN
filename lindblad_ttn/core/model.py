# coding: utf-8
"""TTN state: a Frame with tensor valuations.

The :class:`Model` associates a :class:`~lindblad_ttn.core.graph.Node` with a
``torch.Tensor`` and optionally tracks which node is the current
orthogonality (gauge) center.

This mirrors ``tenso/state/puremodel.py`` with one addition: the
``gauge_center`` attribute.
"""

from __future__ import annotations

from math import prod
from typing import Iterable

import numpy as np
import torch

from lindblad_ttn.core.backend import DEVICE, DTYPE, zeros
from lindblad_ttn.core.graph import Frame, Node, Point


# ---------------------------------------------------------------------------
# Triangular index generator (mirror of tenso's triangular)
# ---------------------------------------------------------------------------

def triangular(n_list: list[int]):
    """Yield flat indices in "triangular" (diagonal-first) order.

    Used for canonical initialization of TTN tensors.

    Parameters
    ----------
    n_list : list[int]
        Sizes of each dimension.

    Yields
    ------
    int
        Flat indices into a tensor of shape ``n_list``.

    Examples
    --------
    >>> list(triangular([2, 2]))
    [0, 1, 2, 3]
    """
    length = len(n_list)
    prod_list = [1]
    for n in n_list:
        prod_list.append(prod_list[-1] * n)

    def key(case):
        return sum(n * i for n, i in zip(prod_list, case))

    combinations: dict[int, list[list[int]]] = {0: [[0] * length]}
    for m in range(prod_list[-1]):
        if m not in combinations:
            prev = combinations[m - 1]
            permutation = [
                case[:j] + [case[j] + 1] + case[j + 1:]
                for case in prev
                for j in range(length)
                if case[j] + 1 < n_list[j]
            ]
            combinations[m] = []
            for case in permutation:
                if case not in combinations[m]:
                    combinations[m].append(case)
        for case in combinations[m]:
            yield key(case)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class Model:
    """TTN state: a Frame with tensor valuations.

    Parameters
    ----------
    valuation : dict[Node, torch.Tensor] or iterable of (Node, Tensor) pairs
        Initial tensor assignments.
    gauge_center : Node, optional
        Which node is currently the orthogonality center.

    Notes
    -----
    The topology (Frame) is kept *separate* from the Model — this mirrors
    tenso's design and keeps the two concerns cleanly separated.
    """

    def __init__(
        self,
        valuation: dict[Node, torch.Tensor] | Iterable[tuple[Node, torch.Tensor]],
        gauge_center: Node | None = None,
    ) -> None:
        self._valuation: dict[Node, torch.Tensor] = dict(valuation)
        self.gauge_center: Node | None = gauge_center

    # ------------------------------------------------------------------
    # Container interface
    # ------------------------------------------------------------------

    def __contains__(self, p: Node) -> bool:
        return p in self._valuation

    def __getitem__(self, p: Node) -> torch.Tensor:
        return self._valuation[p]

    def __setitem__(self, p: Node, v: torch.Tensor) -> None:
        self._valuation[p] = v

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> set[Node]:
        """Set of all nodes present in the valuation."""
        return set(self._valuation.keys())

    def shape(self, p: Node) -> list[int]:
        """Return the shape of the tensor at node ``p``."""
        return list(self._valuation[p].shape)

    def order(self, p: Node) -> int:
        """Return the number of legs (ndim) of the tensor at node ``p``."""
        return self._valuation[p].ndim

    def dimension(self, p: Node, i: int) -> int:
        """Return the size of axis ``i`` of the tensor at node ``p``."""
        return self._valuation[p].shape[i]

    # ------------------------------------------------------------------
    # Copies and transforms
    # ------------------------------------------------------------------

    def copy(self) -> "Model":
        """Shallow copy of this Model (tensors are *not* cloned)."""
        return Model(self._valuation, gauge_center=self.gauge_center)

    def conjugate(self) -> "Model":
        """Return a new Model with all tensors complex-conjugated."""
        new_val = {p: a.conj() for p, a in self._valuation.items()}
        return Model(new_val, gauge_center=self.gauge_center)

    def substitute(
        self,
        valuation: dict[Node, torch.Tensor] | Iterable[tuple[Node, torch.Tensor]],
        gauge_center: Node | None = None,
    ) -> "Model":
        """Return a new Model with updated tensor entries.

        Parameters
        ----------
        valuation : dict or iterable
            Overrides for specific nodes.
        gauge_center : Node, optional
            New gauge center.  If not provided, keeps the existing one.

        Returns
        -------
        Model
        """
        new_model = self.copy()
        new_model._valuation.update(dict(valuation))
        if gauge_center is not None:
            new_model.gauge_center = gauge_center
        return new_model

    def update(
        self,
        valuation: dict[Node, torch.Tensor] | Iterable[tuple[Node, torch.Tensor]],
    ) -> None:
        """In-place update of tensor entries."""
        self._valuation.update(dict(valuation))

    def zero_like(self) -> "Model":
        """Return a new Model with all tensors replaced by zeros of the same shape."""
        shapes = {k: list(v.shape) for k, v in self._valuation.items()}
        return zeros_model(shapes)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, filename: str) -> None:
        """Save the Model to a file using ``torch.save``.

        Parameters
        ----------
        filename : str
            Output path.
        """
        named = {p.name: a for p, a in self._valuation.items()}
        torch.save(named, filename)

    @classmethod
    def load(cls, filename: str) -> "Model":
        """Load a Model previously saved with :meth:`save`.

        Parameters
        ----------
        filename : str
            Input path.

        Returns
        -------
        Model
        """
        from lindblad_ttn.core.graph import Node as _Node
        named = torch.load(filename)
        val = {_Node(name=n): a for n, a in named.items()}
        return cls(val)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def zeros_model(shapes: dict[Node, list[int]]) -> Model:
    """Build a Model with all-zero tensors.

    Parameters
    ----------
    shapes : dict[Node, list[int]]
        Desired shape for each node.

    Returns
    -------
    Model
    """
    val = {p: torch.zeros(shape, dtype=DTYPE, device=DEVICE) for p, shape in shapes.items()}
    return Model(val)


def eye_model(frame: Frame, root: Node, shapes: dict[Node, list[int]]) -> Model:
    """Build a Model with identity-like tensor valuations.

    The root tensor is initialized as a flat vector ``[1, 0, 0, ...]``.
    Non-root tensors are initialized so that axis-0 (the parent axis) is the
    identity map and all other indices are canonical (``triangular`` order).

    Parameters
    ----------
    frame : Frame
        The tree topology.
    root : Node
        Root node (no parent axis).
    shapes : dict[Node, list[int]]
        Desired shape for each node.

    Returns
    -------
    Model
        With ``gauge_center`` set to ``root``.
    """
    assert root in frame
    axes = frame.get_node_axes(root)

    valuation: dict[Node, torch.Tensor] = {}
    for p in frame.nodes:
        shape = shapes[p]
        ax = axes[p]
        if ax is None:
            # Root: flat vector, first element = 1
            flat = np.zeros(prod(shape))
            flat[0] = 1.0
            tensor = torch.tensor(flat.reshape(shape), dtype=DTYPE, device=DEVICE)
        else:
            l_dim = shape[ax]
            r_shape = shape[:ax] + shape[ax + 1:]
            mat = np.zeros([l_dim, prod(r_shape)], dtype=complex)
            for v_i, idx in zip(mat, triangular(r_shape)):
                v_i[idx] = 1.0
            arr = mat.reshape([l_dim] + r_shape)
            arr = np.moveaxis(arr, 0, ax)
            tensor = torch.tensor(arr, dtype=DTYPE, device=DEVICE)
        valuation[p] = tensor

    return Model(valuation, gauge_center=root)
