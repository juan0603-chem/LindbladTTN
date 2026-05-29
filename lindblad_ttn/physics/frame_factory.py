# coding: utf-8
"""TTN topologies for N-site Liouville space (heterogeneous dimensions).

Each site contributes one :class:`~lindblad_ttn.core.graph.End` node carrying
a local Liouville dimension ``d_s²`` (square of the physical dimension).  DOF
names are user-supplied (e.g. ``'q0'``, ``'c0'``, ``'I_V'``) or default to the
legacy ``'q0'``, ``'q1'``, ... pattern.

Three topologies are provided:

* :meth:`train` — linear MPS chain.
* :meth:`balanced_tree` — Huffman-balanced binary (or n-ary) tree.
* :meth:`cavity_centered` (M9) — bosonic sites at the tree centre, spin sites
  at the leaves.  Matches the natural Hamiltonian structure of cavity-mediated
  systems and minimises bond-dimension growth for dispersive couplings.
* :meth:`custom` — user-specified adjacency.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

from lindblad_ttn.core.graph import End, Frame, Node, huffman_tree


class LindbladFrameFactory:
    """Build TTN topologies for an N-site Liouville-space simulation.

    Parameters
    ----------
    n_sites : int, optional
        Number of sites (legacy interface).  When provided alone, DOF names
        default to ``q0``…``q{n-1}``.
    dof_names : list[str], optional
        Explicit DOF names — required when site dimensions differ or for
        readable hybrid topologies.  Takes precedence over ``n_sites``.

    Attributes
    ----------
    ends : dict[str, End]
        Mapping from DOF name to the corresponding :class:`End`.
    """

    def __init__(
        self,
        n_sites: int | None = None,
        dof_names: Sequence[str] | None = None,
    ) -> None:
        if dof_names is not None:
            self.dof_names = list(dof_names)
        elif n_sites is not None:
            if n_sites < 1:
                raise ValueError(f"n_sites must be ≥ 1, got {n_sites}.")
            self.dof_names = [f"q{i}" for i in range(n_sites)]
        else:
            raise ValueError("Pass either n_sites or dof_names.")

        self.n_sites = len(self.dof_names)
        self.ends: dict[str, End] = {name: End(name=name) for name in self.dof_names}

    # ------------------------------------------------------------------
    # Linear tensor train
    # ------------------------------------------------------------------

    def train(self) -> tuple[Frame, Node, dict[str, End]]:
        """Build a linear tensor-train (MPS) topology.

        Returns
        -------
        frame : Frame
        root : Node
        dof_to_end : dict[str, End]
        """
        frame = Frame()
        n = self.n_sites
        names = self.dof_names

        root = Node(name="root")
        frame.add_link(root, self.ends[names[0]])

        prev = root
        for i in range(1, n):
            node = Node(name=f"n{i}")
            frame.add_link(prev, node)
            frame.add_link(node, self.ends[names[i]])
            prev = node

        return frame, root, dict(self.ends)

    # ------------------------------------------------------------------
    # Balanced n-ary tree
    # ------------------------------------------------------------------

    def balanced_tree(self, n_ary: int = 2) -> tuple[Frame, Node, dict[str, End]]:
        """Build a balanced n-ary tree using the Huffman algorithm.

        Parameters
        ----------
        n_ary : int
            Branching factor (default: 2 for binary tree).

        Returns
        -------
        frame : Frame
        root : Node
        dof_to_end : dict[str, End]
        """
        sources = [self.ends[name] for name in self.dof_names]

        if len(sources) == 1:
            frame = Frame()
            root = Node(name="root")
            frame.add_link(root, sources[0])
            return frame, root, dict(self.ends)

        counter = {"n": 0}

        def new_node() -> Node:
            name = f"hnode{counter['n']}"
            counter["n"] += 1
            return Node(name=name)

        graph, huffman_root = huffman_tree(sources, new_node, n_ary=n_ary)

        frame = Frame()
        for parent, children in graph.items():
            for child in children:
                frame.add_link(parent, child)

        if isinstance(huffman_root, Node):
            root = huffman_root
        else:
            root = new_node()
            frame.add_link(root, huffman_root)

        return frame, root, dict(self.ends)

    # ------------------------------------------------------------------
    # Cavity-centred (M9)
    # ------------------------------------------------------------------

    def cavity_centered(
        self,
        cavity_dofs: Sequence[str],
        spin_dofs: Sequence[str] | None = None,
    ) -> tuple[Frame, Node, dict[str, End]]:
        """Star-like topology with bosonic / cavity sites at the centre.

        Cavity DOFs sit on an interior spine; spin / qubit DOFs branch off the
        nearest cavity node.  When there is only one cavity DOF, the structure
        is a pure star.

        Parameters
        ----------
        cavity_dofs : sequence of str
            DOF names corresponding to bosonic / cavity modes (interior).
        spin_dofs : sequence of str, optional
            DOF names for the spin / qubit leaves.  Defaults to every DOF not
            listed in ``cavity_dofs``.

        Returns
        -------
        frame : Frame
        root : Node
        dof_to_end : dict[str, End]
        """
        cavities = list(cavity_dofs)
        if spin_dofs is None:
            spin_dofs = [n for n in self.dof_names if n not in cavities]
        spins = list(spin_dofs)

        if not cavities:
            raise ValueError("cavity_centered requires at least one cavity DOF.")
        for name in cavities + spins:
            if name not in self.ends:
                raise ValueError(f"Unknown DOF name: {name!r}.")

        frame = Frame()

        # Build a spine of cavity Nodes c0 — c1 — c2 — … with the cavity End
        # hanging off each cavity Node.  Spins are then distributed by index.
        cavity_nodes: list[Node] = []
        for i, cname in enumerate(cavities):
            node = Node(name=f"cnode_{cname}")
            cavity_nodes.append(node)
            if i > 0:
                frame.add_link(cavity_nodes[i - 1], node)
            frame.add_link(node, self.ends[cname])

        # Distribute spins evenly across cavity nodes (round-robin); each spin
        # gets its own interior Node so the cavity Node retains its spine
        # connections cleanly.
        for j, sname in enumerate(spins):
            owner = cavity_nodes[j % len(cavity_nodes)]
            leaf_node = Node(name=f"snode_{sname}")
            frame.add_link(owner, leaf_node)
            frame.add_link(leaf_node, self.ends[sname])

        root = cavity_nodes[0]
        return frame, root, dict(self.ends)

    # ------------------------------------------------------------------
    # Custom topology
    # ------------------------------------------------------------------

    def custom(
        self, adjacency: dict[str, list[str]], root_name: str = "root"
    ) -> tuple[Frame, Node, dict[str, End]]:
        """Build a user-specified topology from DOF-name adjacency lists.

        Parameters
        ----------
        adjacency : dict[str, list[str]]
            Maps each interior node name to a list of neighbour names.
            Neighbour names can be DOF names or other interior node names.
        root_name : str
            Name of the root interior node.

        Returns
        -------
        frame : Frame
        root : Node
        dof_to_end : dict[str, End]
        """
        node_map: dict[str, Node] = {}
        all_names = set(adjacency.keys())
        for children in adjacency.values():
            all_names.update(children)
        for name in all_names:
            if name not in self.ends:
                node_map[name] = Node(name=name)

        frame = Frame()
        added: set[frozenset] = set()
        for parent_name, children_names in adjacency.items():
            parent = node_map[parent_name]
            for child_name in children_names:
                child = self.ends.get(child_name) or node_map[child_name]
                key = frozenset([id(parent), id(child)])
                if key not in added:
                    frame.add_link(parent, child)
                    added.add(key)

        root = node_map[root_name]
        return frame, root, dict(self.ends)
