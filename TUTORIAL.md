# LindbladTTN — Usage Tutorial

A hands-on guide to simulating open quantum systems with tree tensor networks.
Every snippet is self-contained and runnable after `pip install -e .`.

**Contents**

1. [Install & 60-second example](#1-install--60-second-example)
2. [Core concepts & conventions](#2-core-concepts--conventions)
3. [Single qubit: decay & driving (legacy API)](#3-single-qubit-decay--driving-legacy-api)
4. [Multiple qubits: topology & bond dimension](#4-multiple-qubits-topology--bond-dimension)
5. [Heterogeneous sites: spins, bosons, higher spins](#5-heterogeneous-sites-spins-bosons-higher-spins)
6. [Hamiltonian templates](#6-hamiltonian-templates)
7. [Time-dependent control & pulses](#7-time-dependent-control--pulses)
8. [Analysis tools: SW, dispersive χ, Magnus, steady state, levels](#8-analysis-tools)
9. [QEC toolkit](#9-qec-toolkit)
10. [Choosing parameters & gotchas](#10-choosing-parameters--gotchas)

---

## 1. Install & 60-second example

```bash
pip install -e .          # from a clone; needs Python ≥ 3.9, pip ≥ 21.3
```

A single qubit decaying from |0⟩ under T₁ relaxation:

```python
import numpy as np
from lindblad_ttn import LindbladTTN

sz = np.array([[1, 0], [0, -1]], dtype=complex)
sm = np.array([[0, 0], [1, 0]], dtype=complex)   # |1><0|, lowers the excited state |0>

solver = LindbladTTN(
    H0=0.5 * sz,           # H = ½ ω σ_z, with ω = 1
    L_ops=[(0.1, sm)],     # one dissipator: rate γ = 0.1, jump operator σ_-
    n_sites=1, bond_dim=4,
    strategy="ps1",
)

rho0 = np.array([[1, 0], [0, 0]], dtype=complex)   # |0><0|
res = solver.run(rho0=rho0, t_span=(0.0, 10.0), dt=0.01,
                 observables=[sz], save_every=10)

print("final <sz> =", res.expect[0][-1].real)      # → relaxes toward -1
print("final Tr(rho) =", res.norm[-1])             # → stays 1.0
```

`res.expect[0]` is the ⟨σ_z⟩ trajectory; `res.times` are the matching times.

---

## 2. Core concepts & conventions

**What it solves.** The Lindblad master equation

$$\dot\rho = -i[H(t),\rho] + \sum_k \gamma_k\Big(L_k\rho L_k^\dagger - \tfrac12\{L_k^\dagger L_k,\rho\}\Big)$$

The density matrix `ρ` is vectorised into Liouville space and represented as a
**tree tensor network**; time evolution uses **TDVP**, so cost grows like
`O(N · D²)` in the bond dimension `D` instead of `O(16^N)` for a dense
Liouvillian.

**Two API surfaces** (mutually exclusive — pick one per solver):

| | Legacy (qubits only) | Heterogeneous (M1+) |
|---|---|---|
| Sites | `n_sites=` (all spin-½) | `sites=[...]` (mixed dims) |
| Hamiltonian | `H0=` full `(2ᴺ,2ᴺ)` matrix | `H_terms=[(c, {name: op})]` |
| Dissipators | `L_ops=[(γ, L)]` full matrices | `L_terms=[(γ, {name: op})]` |
| Drives | `drives=[(f, V)]` | `drives_nd=[(f, V_terms)]` |

**Operator conventions** (shared by both APIs):

- `sz` has eigenvalues `(+1, −1)` for `(|0⟩, |1⟩)`.
- `sm = |1⟩⟨0|` **lowers the excited state** `|0⟩ → |1⟩` (so `|0⟩` is the
  high-energy state under `H = ½ω σ_z`). `sp = sm†`.
- Template frequencies are **angular**: a "5 GHz qubit" is `omega_q = 2*np.pi*5e9`.
  Templates do **not** multiply by 2π for you.

**`solver.run(...)` returns a `LindbladResult`** with:

| field | meaning |
|---|---|
| `res.times` | saved time points (every `save_every` steps) |
| `res.expect[i]` | complex ndarray: ⟨observable *i*⟩ vs time |
| `res.rho_final` | final density matrix, shape `(D, D)`, `D = ∏ dᵢ` |
| `res.norm` | `Tr(ρ)` vs time (sanity check — should stay 1) |
| `res.bond_dims` | max bond dim vs time |
| `res.save_txt(path)` | dump a human-readable report |

**Observables** are passed as full dense `(D, D)` numpy matrices in the
**Kronecker order of your site/qubit list**. For `sites=[q, c]` with dims
`[2, N]`, "⟨σ_z⟩ on the qubit" is `np.kron(q.sz, np.eye(N))`.

`rho0` is automatically normalised to `Tr = 1`. You may also pass a state
vector (1-D array); it is converted to `|ψ⟩⟨ψ|`.

> **Integrator choice.** Use `strategy="ps1"` (projector-splitting, the
> default) for production. `strategy="vmf"` is available but its tangent-space
> projector is only exact for `bond_dim ≤ local_dim²`; on larger bonds it
> tracks diagonal observables but mishandles some coherences.

---

## 3. Single qubit: decay & driving (legacy API)

### Rabi driving with dephasing

```python
import numpy as np
from lindblad_ttn import LindbladTTN

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
sm = np.array([[0, 0], [1, 0]], dtype=complex)

omega   = 1.0      # qubit splitting
Omega_R = 0.3      # Rabi frequency (drive strength)
gamma   = 0.05     # T1 rate
gamma_phi = 0.02   # pure dephasing rate

solver = LindbladTTN(
    H0=0.5 * omega * sz,
    f=lambda t: Omega_R * np.cos(omega * t),   # resonant drive envelope
    V=sx,                                       # ... applied to σ_x
    L_ops=[(gamma, sm), (gamma_phi, sz)],       # T1 + Tφ
    n_sites=1, bond_dim=4, strategy="ps1",
)

rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
res = solver.run(rho0=rho0, t_span=(0, 30), dt=0.01,
                 observables=[sx, sy, sz], save_every=20)

# res.expect[0], [1], [2] are <sx>, <sy>, <sz>
```

**Multiple independent drives** — pass `drives=[(f1, V1), (f2, V2), ...]`
instead of a single `f, V`. The total Hamiltonian becomes
`H(t) = H₀ + Σᵢ fᵢ(t)·Vᵢ`.

---

## 4. Multiple qubits: topology & bond dimension

The legacy API builds `N` qubits; operators are full `(2ᴺ, 2ᴺ)` matrices in
Kronecker order `q0 ⊗ q1 ⊗ …`.

```python
import numpy as np
from functools import reduce
from lindblad_ttn import LindbladTTN

I2 = np.eye(2, dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sm = np.array([[0, 0], [1, 0]], dtype=complex)
kron = lambda *o: reduce(np.kron, o)

# 3-qubit chain with nearest-neighbour ZZ coupling + local decay
J = 0.5
H0 = J * (kron(sz, sz, I2) + kron(I2, sz, sz))
L_ops = [(0.02, kron(sm, I2, I2)),
         (0.02, kron(I2, sm, I2)),
         (0.02, kron(I2, I2, sm))]

solver = LindbladTTN(
    H0=H0, L_ops=L_ops,
    n_sites=3, bond_dim=16,
    topology="tree",        # 'train' (chain) | 'tree' (balanced) | 'cavity_centered'
    strategy="ps1",
)

rho0 = np.zeros((8, 8), dtype=complex); rho0[0, 0] = 1.0   # |000>
res = solver.run(rho0=rho0, t_span=(0, 20), dt=0.01,
                 observables=[kron(sz, I2, I2)], save_every=20)
print("max bond dim reached:", res.bond_dims.max())
```

**Topology cheat-sheet**

- `"train"` — a 1-D chain (MPS-like). Best when interactions are local.
- `"tree"` — a balanced binary tree. Lower bond dims for all-to-all coupling.
- `"cavity_centered"` — puts bosonic mode(s) at the centre; pass
  `cavity_dofs=[...]`. Best when one cavity mediates everything.

**`bond_dim`** caps entanglement. Start at 16–32; raise it until observables
stop changing. Watch `res.bond_dims` — if it pins at your cap the whole run,
you may be truncating.

---

## 5. Heterogeneous sites: spins, bosons, higher spins

The `sites=` API unlocks bosonic modes and higher spins. Each *site* carries a
name, a dimension, and named operators. Build the Hamiltonian and dissipators
as **term lists** where each term maps site-names → local operators (missing
sites are identity).

### Jaynes–Cummings: qubit + cavity

```python
import numpy as np
from lindblad_ttn import LindbladTTN
from lindblad_ttn.sites import spin_half_site, boson_site

q = spin_half_site("q")        # dim 2: q.sx, q.sy, q.sz, q.sp, q.sm
c = boson_site(8, "c")         # Fock-8 cavity: c.a, c.adag, c.n, c.x, c.p

omega_q, omega_c, g = 1.0, 1.0, 0.05   # resonant, weak coupling

H_terms = [
    (0.5 * omega_q, {q.name: q.sz}),                   # ½ ω_q σ_z
    (omega_c,       {c.name: c.n}),                    # ω_c a†a
    (g,             {q.name: q.sp, c.name: c.a}),      # g σ_+ a
    (g,             {q.name: q.sm, c.name: c.adag}),   # g σ_- a†
]
L_terms = [
    (0.01, {c.name: c.a}),     # cavity photon loss κ
    (1e-3, {q.name: q.sm}),    # qubit T1
]

solver = LindbladTTN(sites=[q, c], H_terms=H_terms, L_terms=L_terms,
                     bond_dim=16, topology="train", strategy="ps1")

# Initial state |e> ⊗ |0>  (qubit excited = |0> in our convention, cavity vacuum)
import numpy as np
rho_q = np.array([[1, 0], [0, 0]], dtype=complex)
rho_c = np.zeros((8, 8), dtype=complex); rho_c[0, 0] = 1.0
rho0 = np.kron(rho_q, rho_c)

# Observables in Kronecker order [q, c]:
n_cav = np.kron(np.eye(2), c.n)
sz_q  = np.kron(q.sz, np.eye(8))
res = solver.run(rho0=rho0, t_span=(0, 200), dt=0.05,
                 observables=[sz_q, n_cav], save_every=20)
# Vacuum Rabi oscillations: excitation sloshes between qubit and cavity.
```

### Site reference

| Constructor | dim | operators / helpers |
|---|---|---|
| `spin_half_site("q")` | 2 | `sx sy sz sp sm proj0 proj1` |
| `boson_site(N, "c")` | N | `a adag n x p`; `.kerr(K)`, `.coherent_state(α)`, `.fock_state(n)` |
| `spin_site(S, "m")` | 2S+1 | `Sx Sy Sz Sp Sm`; `.stevens(k,q)`, `.Sz_squared()`, `.S_squared()` |

### Higher spin + crystal field (Stevens operators)

```python
from lindblad_ttn.sites import spin_site

m = spin_site(3.5, "Dy")          # S = 7/2 → dim 8
# Zero-field splitting D·Sz² + crystal-field O_4^0:
H_terms = [
    (0.2, {m.name: m.Sz_squared()}),
    (1e-3, {m.name: m.stevens(4, 0)}),
]
```

---

## 6. Hamiltonian templates

Templates return a ready-to-use `(sites, H_terms, L_terms)` triple. Frequencies
are **angular**.

```python
import numpy as np
from lindblad_ttn import LindbladTTN
from lindblad_ttn.templates import jaynes_cummings, transmon, vanadyl_spin, merge

# One-liner JC with built-in dissipation:
sites, H_terms, L_terms = jaynes_cummings(
    omega_q=2*np.pi*5.0, omega_c=2*np.pi*5.0, g=2*np.pi*0.1,
    N_cut=8, gamma_q=1e-3, kappa=5e-3,
)
solver = LindbladTTN(sites=sites, H_terms=H_terms, L_terms=L_terms,
                     bond_dim=16, topology="train")
```

**Available templates**

| Template | Signature (key args) |
|---|---|
| `transmon` | `(omega_q, alpha, N_cut=4, T1=, Tphi=)` — Duffing oscillator |
| `fluxonium` | `(E_C, E_J, E_L, phi_ext=π, N_cut=30, T1=)` |
| `bare_lc` | `(omega_c, N_cut=8, kappa=)` |
| `vanadyl_spin` | `(omega_q, A_iso=, A_dip=, T1=, T2=, include_nucleus=False)` |
| `lanthanide_smm` | `(g_J, omega, J, B20=, B40=, B44=)` — TbPc₂ / Dy-SIM |
| `jaynes_cummings` | `(omega_q, omega_c, g, N_cut=8, gamma_q=, kappa=)` |
| `tavis_cummings` | `(omega_qs, omega_c, gs, N_cut=8, gamma_qs=, kappa=)` — N spins + 1 cavity |
| `dispersive_readout` | `(omega_q, omega_c, chi, N_cut=6, gamma_q=, kappa=)` |

**Combine systems with `merge`** (site names must be unique):

```python
from lindblad_ttn.templates import vanadyl_spin, bare_lc, merge

spin = vanadyl_spin(omega_q=1.0, T1=1e4, T2=5e3, name_e="vo")
bus  = bare_lc(omega_c=0.95, N_cut=6, kappa=5e-3, name="cav")
sites, H_terms, L_terms = merge(spin, bus)
# add your own coupling term by hand:
vo, cav = sites[0], sites[1]
H_terms += [
    (0.02, {vo.name: vo.sp, cav.name: cav.a}),
    (0.02, {vo.name: vo.sm, cav.name: cav.adag}),
]
```

---

## 7. Time-dependent control & pulses

The pulse library returns plain `f(t)` callables you plug into `drives=`
(legacy) or `drives_nd=` (heterogeneous).

```python
from lindblad_ttn.control import gaussian, drag, square_rise, integrate_pulse
```

| Pulse | Returns | Use |
|---|---|---|
| `gaussian(amp, t0, sigma)` | `f(t)` | single-quadrature rotation |
| `drag(amp, t0, sigma, anharm, beta=1.0)` | `(f_I, f_Q)` | leakage-suppressed transmon gate |
| `square_rise(amp, t_start, t_end, t_rise)` | `f(t)` | flat-top with smooth edges |
| `constant_pulse`, `cosine_drive`, `sequence` | `f(t)` | CW / composite waveforms |

**Calibrating a rotation angle.** For `H(t) = f(t)·V` with Hermitian `V`, the
propagator is `U = exp(−i V ∫f dt)`. To realise `exp(−i(θ/2)V)` you need
`∫f dt = θ/2`. Use `integrate_pulse` to check:

```python
import numpy as np
from lindblad_ttn.control import gaussian, integrate_pulse

sigma, t0 = 0.3, 1.5
# Gaussian area = amp·σ·√(2π); for a π/2 rotation set area = π/4:
amp = (np.pi/4) / (sigma*np.sqrt(2*np.pi))
f = gaussian(amp, t0, sigma)
print("pulse area =", integrate_pulse(f, 0, 3))   # ≈ π/4 ≈ 0.785
```

**Heterogeneous pulse example** — a Gaussian σ_x drive on a qubit:

```python
import numpy as np
from lindblad_ttn import LindbladTTN
from lindblad_ttn.sites import spin_half_site
from lindblad_ttn.control import gaussian

q = spin_half_site("q")
sigma, t0 = 0.3, 1.5
amp = (np.pi/2) / (sigma*np.sqrt(2*np.pi))   # π rotation (area = π/2)
f = gaussian(amp, t0, sigma)

solver = LindbladTTN(
    sites=[q],
    H_terms=[],                                       # no static H (rotating frame)
    drives_nd=[(f, [(1.0+0j, {q.name: q.sx})])],      # f(t) · (1·σ_x)
    L_terms=[],
    bond_dim=4, topology="train", strategy="ps1",
)
rho0 = np.array([[1, 0], [0, 0]], dtype=complex)       # |0>
res = solver.run(rho0=rho0, t_span=(0, 3), dt=0.005,
                 observables=[q.sz], save_every=10)
print("final <sz> =", res.expect[0][-1].real)         # ≈ +1 → flipped to |1>
```

> A worked 2-qubit pulse circuit (Hadamard + cross-resonance, validated against
> QuTiP) lives in `examples/qutip_circuit.py`. A full hybrid spin+cavity+transmon
> demo is in `examples/hybrid_vanadyl_transmon.py`.

---

## 8. Analysis tools

These are **dense** helpers for parameterising and sanity-checking a TTN
simulation — they operate on small numpy matrices, not on the TTN itself.

### Dispersive shift χ from a Jaynes–Cummings Hamiltonian

```python
import numpy as np
from lindblad_ttn.effective import dispersive_shift

# Build the dense JC Hamiltonian in basis |i_q> ⊗ |n_c> (index = i_q*N + n_c):
N = 6
a = np.diag(np.sqrt(np.arange(1, N)), 1)            # annihilation
adag = a.conj().T
n = adag @ a
sz = np.array([[1, 0], [0, -1]], dtype=complex)
sp = np.array([[0, 1], [0, 0]], dtype=complex)
sm = sp.conj().T

wq, wc, g = 5.0, 4.0, 0.05       # dispersive regime: |g/Δ| = 0.05 ≪ 1
H = (0.5*wq*np.kron(sz, np.eye(N)) + wc*np.kron(np.eye(2), n)
     + g*(np.kron(sp, a) + np.kron(sm, adag)))

chi = dispersive_shift(H, qubit_dim=2, cavity_dim=N)
print("chi  =", chi)                 # → 0.002482
print("g²/Δ =", g**2 / (wq - wc))    # → 0.0025  (leading-order estimate; agrees to <1%)
```

### Schrieffer–Wolff & Magnus

```python
import numpy as np
from lindblad_ttn.effective import schrieffer_wolff, magnus_average

# SW: block-diagonalise H0+V to 4th order
H_eff = schrieffer_wolff(H0, V, order=4)

# Magnus: time-averaged (Floquet) Hamiltonian over one period
H_avg = magnus_average(lambda t: H0 + np.cos(2*np.pi*t)*V, period=1.0, order=2)
```

### Steady state & energy levels

```python
import numpy as np
from lindblad_ttn.propagation.steady import steady_state_dense, energy_levels_dense
from lindblad_ttn.sites import spin_half_site

q = spin_half_site("q")
# Driven, damped qubit → unique steady state:
H_terms = [(0.5, {q.name: q.sz}), (0.1, {q.name: q.sx})]
L_terms = [(0.05, {q.name: q.sm})]
rho_ss = steady_state_dense(site_dims=[2], dof_names=["q"],
                            H_terms=H_terms, L_terms=L_terms)
print("steady-state populations:", np.diag(rho_ss).real)

# Lowest k levels of a dense Hamiltonian:
H_dense = 0.5*q.sz + 0.1*q.sx
eigs, vecs = energy_levels_dense(H_dense, k=2)
print("levels:", eigs)
```

---

## 9. QEC toolkit

Stabilizer codes know their generators, logical operators, and an abstract
syndrome circuit.

```python
from lindblad_ttn.qec import RepetitionCode, SurfaceCode, ColorCode488

code = SurfaceCode(distance=3)
print("data qubits:", code.n_data, " distance:", code.distance)

for g in code.generators():
    print(g)                         # e.g. "X_0 X_1 X_3 X_4"

logical_Z = code.logical("Z")        # PauliString
print("logical Z:", logical_Z, " weight:", logical_Z.weight())

# Abstract syndrome-extraction circuit (gate, qubits):
for gate in code.syndrome_circuit()[:6]:
    print(gate)                      # ('RESET', (9,)), ('CNOT', (0, 9)), ...
```

`PauliString` supports `.commutes(other)`, `.weight()`, `.as_dense()` (small
codes only — exponential), and `str()`. `RepetitionCode(N)` and `ColorCode488()`
share the same interface. MWPM decoding is available via
`lindblad_ttn.qec.decoders` (requires the `qec` extra: `pip install -e ".[qec]"`).

---

## 10. Choosing parameters & gotchas

**Integrator (`strategy`)**
- `"ps1"` — default, production-grade, trace-preserving. Fixed bond dimension.
- `"vmf"` — ODE-based; only use for `bond_dim ≤ local_dim²` (e.g. one qubit).

**`bond_dim`** — the entanglement cap. Converge it: rerun with a larger value
and check observables are stable. If `res.bond_dims` sits at the cap, increase.

**`dt`** — the step. Halve it and confirm trajectories don't move. Pulses need
several steps per `sigma`; oscillations need several steps per period.

**`N_cut` (boson cutoff)** — must exceed the largest photon number reached.
Check `⟨n⟩` stays well below `N_cut−1`; if `⟨a†a⟩` approaches the cutoff, raise it.

**Frequencies are angular** in templates (multiply your Hz by 2π).

**Known limitations (v0.3.0)**
- TDVP is **fixed-rank** (PS1): a rank-1 product initial state on a large tree
  can't grow entanglement freely, which can show up as slow trace drift over
  very long evolutions. Mitigate with a higher `bond_dim`, a shorter horizon,
  or a slightly mixed initial state.
- `topology="cavity_centered"` with multiple children per interior node falls
  back to a train-style initialiser; it's most useful for many spins around one
  cavity. See `CHANGELOG.md`.
- `vmf` mishandles some off-diagonal coherences when `bond_dim > local_dim²`.

**Sanity checks to keep in every run**
- `res.norm` should stay `1.0` (trace preservation).
- `res.expect[i]` of a Hermitian observable should be real (tiny imaginary part).
- Compare a small case against `examples/qutip_*.py` (QuTiP parity ≈ 1e-5).

---

*See also:* `README.md` for the quick start, `CHANGELOG.md` for release notes,
the `examples/` folder for runnable scripts, and `docs/roadmap_site/` for the
physics background (molecular spins, superconducting elements, coupling regimes,
effective Hamiltonians, and QEC codes).
