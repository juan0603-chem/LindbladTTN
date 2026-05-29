# TENSO Architectural Analysis

Reference codebase: `tenso/src/tenso/` — read-only blueprint.  
LindbladTTN must **never** import from `tenso`.

---

## 1. Data Structures

### Point / Node / End (`tenso/state/pureframe.py`)

```
Point  ←  abstract base, unique name, WeakValueDictionary cache in __new__
  Node   ←  interior vertex, multiple links allowed
  End    ←  leaf (physical DOF), exactly one link
```

- `Point.__new__` uses a **class-level `WeakValueDictionary`** keyed by `(cls.__name__, name)`.
  Calling `Node('a')` twice returns the *same* object.
- `Node` repr: `(name)`, `End` repr: `<name>`.

### Frame (`tenso/state/pureframe.py`)

Stores the tree topology in three parallel dicts:

| Attribute | Type | Meaning |
|-----------|------|---------|
| `_neighbor` | `dict[Point, list[Point]]` | Adjacency list |
| `_duality` | `dict[(Point, int\|None), (Point, int\|None)]` | Maps each (point, axis-index) to its dual across the edge |
| `_axes` | `dict[(Point1, Point2), (int\|None, int\|None)]` | Axis indices on each side of an edge |

`add_link(p, q)`:
- If `p` is a `Node`, its axis index `i = len(current neighbors of p)` (0-indexed, increments).
- If `p` is an `End`, axis is `None` (Ends have no axis dimension — they are contracted out).
- Both `_duality` and `_axes` are filled symmetrically.

Key traversal methods:
- `node_link_visitor(start)` → `list[(p, i, q, j)]` in round-trip DFS order — used for sweep scheduling.
- `get_node_axes(start)` → `dict[Node, int|None]` — for each node, which axis-index points toward `start` (None for start itself = root).
- `node_visitor(start, method='DFS'|'BFS')` — simple traversal without edge info.

### Model (`tenso/state/puremodel.py`)

```python
class Model:
    _valuation: dict[Node, OptArray]   # Node → PyTorch tensor
```

- **No Frame reference** — topology is always kept separate.
- **No gauge tracking** — gauge center is implicit; the propagator tracks it.
- Functional update style: `substitute(valuation)` returns a *new* Model.
- `eye_model(frame, root, shapes)` builds an identity-like initial state using the `triangular` index generator.
- `triangular(n_list)` generates multi-index positions in "triangular" order for canonical initialization.

### Tensor Leg Ordering

Axis `i` of Node `p`'s tensor corresponds to the neighbor registered at position `i` in `frame._neighbor[p]`.  
`frame._axes[(p, q)] = (i, j)` means: axis `i` of `p`'s tensor ↔ axis `j` of `q`'s tensor.  
`End` nodes do **not** contribute an axis to their parent Node's tensor — they are open physical bonds that the SoP operators act on.

---

## 2. Operators — Sum of Products (SoP)

### SparseSPO (`tenso/operator/sparse.py`)

A Hamiltonian/Liouvillian in sum-of-products form:

```
H = Σ_k  O_{e1}^(k) ⊗ O_{e2}^(k) ⊗ ...
```

where each `O_{e}^(k)` is a local matrix on `End` node `e`.

- `op_list`: `list[dict[End, OptArray]]` — time-independent terms.
- `f_list`: callable returning `list[(coeff, dict[End, OptArray])]` at time `t` — time-dependent terms.
- Operators are **always defined on End nodes** (leaves), never on interior Nodes.
- The propagator queries `get_ti_terms()` / `get_td_terms(t)` to get the list of (coeff, op_dict) pairs.

### Environment Contraction (`opt_multitransform`)

`opt_multitransform(op_dict: dict[int, OptArray], tensor) → tensor`

Applies a sequence of matrix multiplications, one per axis:
```python
for ax, mat in op_dict.items():
    tensor = opt_transform(mat, tensor, 1, ax)
```

This is the core primitive for applying local operators to multi-index tensors.

The "mean field" at node `p`, axis `i` is:
```
MF[p, i] = Tr_{all except axis i}(|ψ⟩⟨ψ| ⊗ O_dict)
```
computed bottom-up from leaf Ends toward the root.

---

## 3. Propagation — TDVP

### SparsePropagator (`tenso/operator/sparse.py`)

Three evolution modes:

#### VMF — Variable Mean-Field
1. **Vectorize** all Node tensors into a single flat complex vector.
2. **Complex ODE trick**: split `y` into `(y.real, y.imag)` as a tuple; torchdiffeq works on real numbers, so the RHS function must reconstruct complex from the tuple.
3. **RHS function** `rhs(t, y_tuple)`:
   - Reshape vector → per-node tensors.
   - For each term in SoP: compute mean fields bottom-up.
   - Accumulate derivative: `dA_s/dt += H_eff_s @ A_s`.
   - Vectorize derivatives and return as tuple.
4. Call `torchdiffeq.odeint` or home-made integrator.

EOM for non-root node `s` (TDVP projected gradient):
```
dU_s/dt = (I - U_s U_s†) H_eff_s U_s
```

EOM for root node `0`:
```
dA_0/dt = H_eff_root @ A_0
```

#### PS1 — Projector Splitting (1-site, fixed rank)

Uses `frame.node_link_visitor(root)` which yields `(p, i, q, j)` pairs in round-trip order.

**Forward sweep** (leaf → root, half-step `+dt/2`):
1. For each `(p, i, q, j)`:
   - Compute `H_eff` at `p` using bottom-up environment.
   - Integrate: `A_p ← exp(H_eff_p * dt/2) @ A_p`.
   - `_one_site_split(A_p, axis=i)` → `(p_tensor, edge_array)` where SVD is done at axis `i` and `S·Vᴴ` becomes `edge_array`.
   - `A_q ← _one_site_merge(A_q, axis=j, edge_array)` — absorb `edge_array` into neighbor.

**Backward sweep** (root → leaf, half-step `−dt/2`): same in reverse order.

Result: 2nd-order Strang splitting.

#### PS2 — Projector Splitting (2-site, adaptive rank)

Same structure as PS1 but merges two adjacent tensors before propagation and splits with SVD truncation, allowing bond dimension to grow or shrink.

### `_one_site_split(array, axis)` (key primitive)

```python
shape = list(array.shape)
l_shape = shape[:i] + shape[i+1:]
mat_p = array.moveaxis(i, -1).reshape((-1, dim))
u, s, vh = opt_svd(mat_p)
edge_array = s.to(opt_dtype)[:, None] * vh   # singular values absorbed into Vᴴ
p_tensor = u.reshape(l_shape + [-1]).moveaxis(-1, i)
return p_tensor, edge_array
```

### `_one_site_merge(array, j, from_)` (key primitive)

```python
return opt_transform(from_, array, 1, j)   # contracts from_[1,:] with array[j,:]
```

### Regularization (C*-adjointness)

For ill-conditioned bonds, the VMF EOM needs regularization.
The `vmf_atol` parameter controls the minimum singular value threshold used in pseudo-inverse computations:
```python
torch.linalg.pinv(S_matrix, atol=vmf_atol)
```

---

## 4. HEOM Physical Layer (`tenso/heom/`)

### Structure

```
sys_ket_end  : End — system ket index (dimension = sys_dim)
sys_bra_end  : End — system bra index (dimension = sys_dim)
bath_ends    : list[End] — one per bath mode (dimension = bath_dim_k)
```

The HEOM Liouvillian is assembled in SoP form with operators on these Ends.

### Liouvillian Assembly (`Hierachy.lvn_list`)

```python
# Von Neumann: H|ket⟩⟨bra|
{sys_ket_end: -i*H}   # acts on ket
{sys_bra_end: +i*H†}  # acts on bra

# Lindblad dissipator for (gamma, L):
{sys_ket_end: -0.5*gamma*L†L}  # Lamb shift ket
{sys_bra_end: -0.5*gamma*L†L}  # Lamb shift bra
{sys_ket_end: gamma*L, sys_bra_end: gamma*L†}  # jump
```

### Density Matrix Extraction (`terminate`)

```python
def terminate(tensor, term_dict: dict[int, OptArray]):
    # Contracts term_dict vectors into tensor axes via opt_einsum
    # Used to trace out bath DOFs and extract reduced density matrix
```

### FrameFactory Topologies

- `naive()`: Star topology — all bath Ends connect directly to root Node.
- `tree()`: Huffman balanced binary tree — bath Ends are leaves.
- `train()`: Linear chain — nodes connected sequentially.

Built using `huffman_tree(sources, new_obj, n_ary=2)` from `utils.py`.

---

## 5. Backend (`tenso/libs/backend.py`)

| Symbol | Value |
|--------|-------|
| `opt_dtype` | `torch.complex128` (CPU) or `torch.complex64` (GPU) |
| `opt_device` | `'cpu'` by default; `'cuda'` if available and `FORCE_CPU=False` |
| `_opt.set_grad_enabled(False)` | Autograd disabled globally |

### SVD

```python
def opt_svd(a):
    if not ON_DEVICE_EIGEN_SOLVER:
        a = a.cpu()   # SVD done on CPU even for GPU tensors
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    return u.to(device), s.to(device), vh.to(device)
```

### ODE Integration (complex trick)

torchdiffeq requires real-valued tensors. Complex ODEs are handled by:
```python
# Pack: complex tensor → (real, imag) tuple
# Unpack inside rhs: complex = real + 1j * imag
# Return as (re_dot, im_dot) tuple
```
This is done transparently inside `opt_odeint`.

### Key Contraction Primitives

```python
opt_transform(op, tensor, op_ax, tensor_ax):
    dotted = tensordot(tensor, op, axes=([tensor_ax], [op_ax]))
    return dotted.movedim(-1, tensor_ax)

opt_multitransform(op_dict: dict[int, OptArray], tensor):
    for ax, mat in op_dict.items():
        tensor = opt_transform(mat, tensor, 1, ax)
    return tensor
```

---

## 6. Key Conventions to Mirror

| Convention | Details |
|-----------|---------|
| Default dtype | `torch.complex128` |
| Autograd | Disabled globally (`torch.set_grad_enabled(False)`) |
| SVD on CPU | Even for GPU tensors, SVD falls back to CPU |
| Node axis ordering | Axis `i` = position `i` in `frame._neighbor[p]` list (insertion order) |
| Operator on Ends | SoP terms are always `dict[End, matrix]`, never `dict[Node, matrix]` |
| Gauge center | Tracked externally by propagator (not in Model) |
| Initial state | `eye_model` for identity-like, `zeros_model` for zero |
| Triangular ordering | Used for canonical multi-index initialization |
| Round-trip visitor | `iter_round_visitor` gives (p, i, q, j) pairs for TDVP sweep order |
| `_one_site_split` | SVD at bond axis, `S·Vᴴ` absorbed into `edge_array`, `U` stays as isometry |
| Complex ODE | Split into `(re, im)` tuple for torchdiffeq compatibility |

---

## 7. Design Decisions to Adopt

- **Frame/Node/End topology**: Clean separation of topology (Frame) from data (Model).
- **Functional Model updates**: `substitute()` returns new Model, enables easy rollback.
- **SoP as list of dicts**: Simple, extensible, easy to combine with `+` and `*`.
- **torchdiffeq for ODE**: Proven adaptive integrators (dopri5 default).
- **`(re, im)` tuple trick**: Essential for complex-valued torchdiffeq.
- **`opt_multitransform` pattern**: Efficient multi-axis contraction.
- **Huffman tree builder**: Balanced n-ary topology from leaves.
- **`_one_site_split` / `_one_site_merge`**: Core PS1 primitives, clean and reusable.

---

## 8. Design Decisions to Improve Upon

| tenso pattern | LindbladTTN improvement |
|--------------|------------------------|
| `gauge_center` not in Model | Add `gauge_center: Node \| None` field to `Model` |
| No result dataclass | Use `@dataclass LindbladResult` |
| No Pauli decomposition | Add `pauli_decompose()` utility in `physics/liouvillian.py` |
| Mixed Lindblad/HEOM logic | Clean separate `physics/` layer for Lindblad only |
| Sparse operator on tenso Ends | Use DOF name strings (`'q0'`…) mapped to Ends for cleaner API |
| Backend globals only | Expose `set_device()` / `set_dtype()` functions |
| No docstrings on many internals | NumPy-style docstrings on every public symbol |
| No user-facing solver class | `LindbladTTN` class that is the only user-facing entry point |
| No test suite | Comprehensive `tests/` with analytical validation |
