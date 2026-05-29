# coding: utf-8
"""Tree topology for tensor networks.

Provides :class:`Point`, :class:`Node`, :class:`End`, and :class:`Frame` —
the graph data structures used to represent the topology of a TTN.

This module also contains the tree traversal and construction utilities
(mirrored from ``tenso/libs/utils.py`` and ``tenso/state/pureframe.py``).
"""

from __future__ import annotations

from collections import OrderedDict
try:
    from itertools import pairwise
except ImportError:
    # Python < 3.10 compatibility
    def pairwise(iterable):
        it = iter(iterable)
        a = next(it, None)
        for b in it:
            yield a, b
            a = b
from operator import itemgetter
from typing import Callable, Generator, Iterable, Literal, Optional, TypeVar
from weakref import WeakValueDictionary

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Traversal utilities (mirror of tenso/libs/utils.py)
# ---------------------------------------------------------------------------

def iter_round_visitor(
    start: T,
    r: Callable[[T], list[T]],
) -> Generator[T, None, None]:
    """Depth-first round-trip visitor.

    Yields each node on the way *down* and again on the way *back up*,
    giving the sweep order needed for TDVP projector splitting.

    Parameters
    ----------
    start : T
        Root of the traversal.
    r : callable
        Neighbor function: ``r(node) -> list[node]``.

    Yields
    ------
    T
        Nodes in round-trip DFS order (each interior node appears twice).
    """
    stack, visited = [start], set()
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            nexts = [n for n in r(vertex) if n not in visited]
            stack.extend(
                nexts[i // 2] if i % 2 else vertex
                for i in range(2 * len(nexts))
            )
        yield vertex


def iter_visitor(
    start: T,
    r: Callable[[T], list[T]],
    method: Literal["DFS", "BFS"] = "DFS",
) -> Generator[T, None, None]:
    """Iterative DFS or BFS visitor.

    Parameters
    ----------
    start : T
        Root of the traversal.
    r : callable
        Neighbor function.
    method : {'DFS', 'BFS'}
        Traversal order.

    Yields
    ------
    T
        Nodes in the requested order (each visited exactly once).
    """
    stack, visited = [start], set()
    while stack:
        if method == "DFS":
            stack, vertex = stack[:-1], stack[-1]
        else:
            vertex, stack = stack[0], stack[1:]
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(n for n in r(vertex) if n not in visited)
            yield vertex


def depths(start: T, r: Callable[[T], list[T]]) -> dict[T, int]:
    """Compute the depth of each node relative to ``start``.

    Parameters
    ----------
    start : T
        Root node (depth 0).
    r : callable
        Neighbor function.

    Returns
    -------
    dict[T, int]
        Node → depth mapping.
    """
    ans = {start: 0}
    stack, visited = [start], set()
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            nexts = {n: ans[vertex] + 1 for n in r(vertex) if n not in visited}
            stack.extend(nexts.keys())
            ans.update(nexts)
    return ans


def path(start: T, stop: T, r: Callable[[T], list[T]]) -> list[T] | None:
    """Find the unique path between two nodes in a tree.

    Parameters
    ----------
    start : T
        Source node.
    stop : T
        Destination node.
    r : callable
        Neighbor function.

    Returns
    -------
    list[T] or None
        Ordered list of nodes from ``start`` to ``stop``, or ``None`` if no
        path exists.
    """
    stack, visited = [[start]], set()
    while stack:
        current_path = stack.pop()
        vertex = current_path[-1]
        if vertex is stop:
            return current_path
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(current_path + [n] for n in r(vertex) if n not in visited)
    return None


def huffman_tree(
    sources: list[T],
    new_obj: Callable[[], T],
    importances: list[int] | None = None,
    n_ary: int = 2,
) -> tuple[OrderedDict[T, list[T]], T]:
    """Build a balanced n-ary tree using the Huffman algorithm.

    Parameters
    ----------
    sources : list[T]
        Leaf nodes (End objects).
    new_obj : callable
        Factory for new interior nodes (called with no arguments).
    importances : list[int], optional
        Importance weights for the leaves.  Defaults to uniform weights.
    n_ary : int
        Branching factor (2 = binary tree).

    Returns
    -------
    graph : OrderedDict[T, list[T]]
        Adjacency list (parent → children), from root to leaves.
    root : T
        The root node.
    """
    if importances is None:
        importances = [1] * len(sources)

    sequence = list(zip(sources, importances))
    graph: OrderedDict[T, list[T]] = OrderedDict()
    while len(sequence) > 1:
        sequence.sort(key=itemgetter(1))
        branch, sequence = sequence[:n_ary], sequence[n_ary:]
        weight = sum(w for _, w in branch)
        new = new_obj()
        graph[new] = [node for node, _ in branch]
        sequence.insert(0, (new, weight))

    return OrderedDict(reversed(graph.items())), sequence[0][0]


# ---------------------------------------------------------------------------
# Graph classes (mirror of tenso/state/pureframe.py)
# ---------------------------------------------------------------------------

class Point:
    """Abstract base for a vertex in the tensor-network graph.

    Uses a ``WeakValueDictionary`` cache so that ``Node('a')`` called twice
    returns the *same* Python object (as long as it is still alive).

    Parameters
    ----------
    name : str, optional
        Human-readable identifier.  If omitted, a hex id string is used.
    """

    __cache: WeakValueDictionary = WeakValueDictionary()

    def __new__(cls, name: str | None = None) -> "Point":
        if name is None:
            return object.__new__(cls)
        cache_key = (cls.__name__, name)
        obj = cls.__cache.get(cache_key)
        if obj is None:
            obj = object.__new__(cls)
            cls.__cache[cache_key] = obj
        return obj

    def __init__(self, name: str | None = None) -> None:
        self.name: str = str(hex(id(self))) if name is None else str(name)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


class Node(Point):
    """An interior vertex in the tensor-network graph.

    Nodes can have arbitrarily many links and carry a core tensor.
    """

    def __repr__(self) -> str:
        return f"({self.name})"


class End(Point):
    """A leaf vertex in the tensor-network graph.

    Ends have exactly one link and represent physical (open) bonds.
    Operators in the SoP are always defined on Ends.
    """

    def __repr__(self) -> str:
        return f"<{self.name}>"


# ---------------------------------------------------------------------------
# Frame — the graph data structure
# ---------------------------------------------------------------------------

class Frame:
    """Holds the topology of a tensor network as an undirected graph.

    Attributes
    ----------
    _neighbor : dict
        Adjacency list: ``Point → list[Point]``.
    _duality : dict
        Maps ``(Point, axis|None) → (Point, axis|None)`` across each edge.
    _axes : dict
        Maps ``(Point1, Point2) → (axis1|None, axis2|None)``.
    """

    def __init__(self) -> None:
        self._neighbor: dict[Point, list[Point]] = {}
        self._duality: dict[tuple[Point, int | None], tuple[Point, int | None]] = {}
        self._axes: dict[tuple[Point, Point], tuple[int | None, int | None]] = {}

    def __contains__(self, p: Point) -> bool:
        return p in self._neighbor

    def __str__(self) -> str:
        parts = []
        for k, v in self.get_graph().items():
            neighbors = ", ".join(f"{type(p).__name__}('{p.name}')" for p in v)
            parts.append(f"{type(k).__name__}('{k.name}'): [{neighbors}]")
        return "{" + ", ".join(parts) + "}"

    def copy(self) -> "Frame":
        """Return a shallow copy of this Frame."""
        new = Frame()
        new._neighbor = dict(self._neighbor)
        new._duality = dict(self._duality)
        new._axes = dict(self._axes)
        return new

    def add_link(self, p: Point, q: Point) -> None:
        """Add an undirected link between two Points.

        Parameters
        ----------
        p : Point
            First endpoint.
        q : Point
            Second endpoint.

        Notes
        -----
        :class:`End` nodes can only have one link (axis is ``None``).
        :class:`Node` nodes accumulate links at successive integer axes.
        Calling this twice on the same pair raises an assertion error.
        """
        is_p_node = isinstance(p, Node)
        is_q_node = isinstance(q, Node)

        if p not in self._neighbor:
            self._neighbor[p] = []
        else:
            assert is_p_node, f"{p!r} is an End and already has a link."
        if q not in self._neighbor:
            self._neighbor[q] = []
        else:
            assert is_q_node, f"{q!r} is an End and already has a link."

        i = len(self._neighbor[p]) if is_p_node else None
        j = len(self._neighbor[q]) if is_q_node else None

        self._axes[(p, q)] = (i, j)
        self._axes[(q, p)] = (j, i)
        self._duality[(p, i)] = (q, j)
        self._duality[(q, j)] = (p, i)
        self._neighbor[p].append(q)
        self._neighbor[q].append(p)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def points(self) -> set[Point]:
        """All Points (Nodes and Ends) in the frame."""
        return set(self._neighbor.keys())

    @property
    def nodes(self) -> set[Node]:
        """All :class:`Node` objects in the frame."""
        return {p for p in self._neighbor if isinstance(p, Node)}

    @property
    def ends(self) -> set[End]:
        """All :class:`End` objects in the frame."""
        return {p for p in self._neighbor if isinstance(p, End)}

    def degree(self, p: Node) -> int:
        """Return the number of links attached to ``p``."""
        return len(self._neighbor[p])

    def dual(self, p: Point, i: int | None) -> tuple[Point, int | None]:
        """Return the dual (neighbour, axis) across the edge ``(p, axis=i)``."""
        return self._duality[p, i]

    def axes(self, p: Point, q: Point) -> tuple[int | None, int | None]:
        """Return the ``(axis_p, axis_q)`` pair for the edge between ``p`` and ``q``."""
        return self._axes[p, q]

    def near_points(self, key: Point) -> list[Point]:
        """Return all neighbors of ``key`` (both Nodes and Ends)."""
        return list(self._neighbor[key])

    def near_nodes(self, key: Node) -> list[Node]:
        """Return all :class:`Node` neighbors of ``key``."""
        return [n for n in self._neighbor[key] if isinstance(n, Node)]

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def node_visitor(
        self, start: Node, method: Literal["DFS", "BFS"] = "DFS"
    ) -> list[Node]:
        """Return all Node objects reachable from ``start`` via DFS or BFS.

        Parameters
        ----------
        start : Node
        method : {'DFS', 'BFS'}

        Returns
        -------
        list[Node]
        """
        return list(iter_visitor(start, self.near_nodes, method=method))

    def point_visitor(
        self, start: Point, method: Literal["DFS", "BFS"] = "DFS"
    ) -> list[Point]:
        """Return all Points reachable from ``start`` via DFS or BFS."""
        return list(iter_visitor(start, self.near_points, method=method))

    def node_link_visitor(
        self, start: Node
    ) -> list[tuple[Node, int, Node, int]]:
        """Round-trip DFS over Node–Node links, returning ``(p, i, q, j)`` tuples.

        Used by TDVP to schedule the sweep order.

        Parameters
        ----------
        start : Node
            Root of the sweep.

        Returns
        -------
        list[tuple[Node, int, Node, int]]
            Each tuple gives ``(src_node, src_axis, dst_node, dst_axis)``.
        """
        paired = list(pairwise(iter_round_visitor(start, self.near_nodes)))
        result = []
        for p, q in paired:
            i, j = self._axes[p, q]
            result.append((p, i, q, j))
        return result

    def point_link_visitor(
        self, start: Point
    ) -> list[tuple[Point, int | None, Point, int | None]]:
        """Round-trip DFS over all links (including Ends)."""
        paired = list(pairwise(iter_round_visitor(start, self.near_points)))
        result = []
        for p, q in paired:
            i, j = self._axes[p, q]
            result.append((p, i, q, j))
        return result

    def get_node_depths(self, start: Node) -> dict[Node, int]:
        """Compute depth of each Node relative to ``start`` (depth 0)."""
        return depths(start, self.near_nodes)

    def get_node_axes(self, start: Point) -> dict[Node, int | None]:
        """For each Node, return the axis index that points toward ``start``.

        Parameters
        ----------
        start : Point
            The reference point (typically the root).

        Returns
        -------
        dict[Node, int | None]
            ``None`` for ``start`` itself (no parent axis).
        """
        ans: dict[Point, int | None] = {start: None}
        for p, _, q, j in self.point_link_visitor(start):
            if p in ans and q not in ans:
                ans[q] = j
        return {k: v for k, v in ans.items() if isinstance(k, Node)}

    def get_graph(self) -> dict[Node, list[Point]]:
        """Return the adjacency list restricted to Node keys."""
        return {k: list(v) for k, v in self._neighbor.items() if isinstance(k, Node)}

    def path(self, src: Node, dst: Node) -> list[Node]:
        """Return the unique Node path from ``src`` to ``dst``.

        Parameters
        ----------
        src : Node
        dst : Node

        Returns
        -------
        list[Node]
        """
        result = path(src, dst, self.near_nodes)
        if result is None:
            raise ValueError(f"No path from {src!r} to {dst!r}.")
        return result

    def construct_from_graph(self, graph: dict[Node, list[Point]]) -> None:
        """Populate the frame from a pre-built adjacency dict.

        Parameters
        ----------
        graph : dict[Node, list[Point]]
            For each Node, its list of neighbors (Nodes or Ends).
        """
        assert not self._neighbor, "Frame already populated."
        added: set[tuple[Point, Point]] = set()
        for n, children in graph.items():
            for child in children:
                key = (min(id(n), id(child)), max(id(n), id(child)))
                if key not in added:
                    self.add_link(n, child)
                    added.add(key)
