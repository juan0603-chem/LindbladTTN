# LindbladTTN

Lindblad master equation solver using Tree Tensor Networks (TTN) and TDVP.

## Installation

Requires Python ≥ 3.9 and pip ≥ 21.3.

**From GitHub (no clone):**

```bash
pip install "git+https://github.com/juan0603-chem/LindbladTTN.git"
```

**From a local clone (editable / development):**

```bash
git clone https://github.com/juan0603-chem/LindbladTTN.git
cd lindblad-ttn
pip install -e .
```

**Optional extras:**

```bash
pip install -e ".[dev]"       # pytest + coverage (run the test suite)
pip install -e ".[qec]"       # PyMatching (surface/color-code decoding)
pip install -e ".[examples]"  # qutip (parity-check reference in examples/)
pip install -e ".[all]"       # everything above
```

## Quick Start

```python
import numpy as np
from lindblad_ttn import LindbladTTN

# Pauli matrices
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
sm = np.array([[0, 0], [1, 0]], dtype=complex)  # sigma_minus

# Single qubit decay
omega = 1.0
gamma = 0.1
H0 = 0.5 * omega * sz
L_ops = [(gamma, sm)]

solver = LindbladTTN(
    H0=H0,
    f=None,
    V=None,
    L_ops=L_ops,
    n_sites=1,
    bond_dim=4,
    topology='train',
    device='cpu',
    strategy='ps1',
)

rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)

result = solver.run(
    rho0=rho0,
    t_span=(0.0, 10.0),
    dt=0.01,
    observables=[sz],
    save_every=10,
    verbose=True,
)

print("Final norm:", result.norm[-1])
print("<sz>(t):", result.expect[0])
```

## Driven System

```python
solver = LindbladTTN(
    H0=0.5 * omega * sz,
    f=lambda t: np.cos(omega * t),   # drive envelope
    V=sx,                             # drive operator
    L_ops=[(gamma, sm)],
    n_sites=1,
    bond_dim=8,
)
```

## Multi-Qubit System

```python
import numpy as np
from lindblad_ttn import LindbladTTN

n = 4  # qubits
I2 = np.eye(2)
sx = np.array([[0,1],[1,0]], dtype=complex)
sm = np.array([[0,0],[1,0]], dtype=complex)

def kron_list(ops):
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

# Local decay on each qubit
L_ops = []
for i in range(n):
    ops = [I2]*n; ops[i] = sm
    L_ops.append((0.05, kron_list(ops)))

solver = LindbladTTN(
    H0=None,
    f=None,
    V=None,
    L_ops=L_ops,
    n_sites=n,
    bond_dim=16,
    topology='tree',
)
```

## Architecture

LindbladTTN is architecturally inspired by [tenso](https://github.com/vINyLogY/tenso) but
written entirely from scratch. It uses the same backend:
- **torch** — tensor operations, GPU support
- **torchdiffeq** — adaptive ODE integration (dopri5)
- **numpy / scipy** — input handling, operator construction
- **tqdm** — progress bars

### Project Layout

```
lindblad_ttn/
├── core/           # TTN infrastructure: graph, model, SoP, gauge, backend
├── propagation/    # TDVP engine: integrators, effective Hamiltonian, sweeps
├── physics/        # Lindblad layer: Liouville space, Liouvillian SoP, frame factory
├── solver.py       # LindbladTTN — the only class the user touches
└── time_dependent.py  # H(t) = H0 + f(t)*V combination
```
