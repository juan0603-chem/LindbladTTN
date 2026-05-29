"""Convert each example script in `examples/` into a Jupyter notebook.

Writes one .ipynb per source script under docs/notebooks/. Notebooks are
written without executed outputs; running them in a Jupyter session will
populate the figure cells.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
NOTEBOOKS_DIR = ROOT / "docs" / "notebooks"

# Each entry: (source_script, output_notebook_stem, title, description)
JOBS: list[tuple[str, str, str, str]] = [
    (
        "qutip_simple.py",
        "single_qubit_decay",
        "Single-qubit decay",
        (
            "Spontaneous emission of a two-level atom. We solve the Lindblad "
            "master equation for $H = \\tfrac{\\omega}{2}\\sigma_z$ with a "
            "$L = \\sqrt{\\gamma}\\,\\sigma_-$ jump operator and validate "
            "the LindbladTTN result against `qutip.mesolve`. The two solvers "
            "should agree to ~1e-7."
        ),
    ),
    (
        "qutip_driven.py",
        "driven_qubit",
        "Driven qubit (Rabi + decay)",
        (
            "An on-resonance coherent drive $\\Omega\\cos(\\omega_d t)\\,\\sigma_x$ "
            "on top of a T1 decay channel. The population executes Rabi "
            "oscillations while the envelope damps toward the steady state. "
            "Demonstrates the `f, V` single-drive shortcut."
        ),
    ),
    (
        "two_qubit_bell_state.py",
        "two_qubit_bell_state",
        "Bell-state generation (Rx + cross-resonance)",
        (
            "Two-qubit Bell state preparation via a 2-ns $R_x(\\pi/2)$ rotation "
            "on qubit 0 followed by a 25-ns cross-resonance ZX gate. Includes "
            "realistic T1 = 100 µs and T2 = 50 µs decoherence. Tracks "
            "populations, coherences, Wootters concurrence, and trace "
            "preservation throughout the gate sequence."
        ),
    ),
    (
        "five_qubit_cat_state.py",
        "five_qubit_cat_state",
        "Five-qubit cat state",
        (
            "Prepare and evolve a GHZ-style cat state "
            "$(|00000\\rangle + |11111\\rangle)/\\sqrt{2}$ under local "
            "$\\sigma_-$ decay on every site. A natural fit for the "
            "`topology='tree'` balanced-binary-tree TTN."
        ),
    ),
    (
        "five_qubit_cascade_gaussian_drives.py",
        "five_qubit_cascade",
        "Cascade of Gaussian drives on five qubits",
        (
            "Drive each qubit in turn with a Gaussian pulse, using the "
            "multi-channel `drives=[(f_i, V_i), ...]` API. Each drive is a "
            "separate SoP term composed at evaluation time by "
            "`TimeDependentSoP`."
        ),
    ),
    (
        "cat_state_benchmark.py",
        "scaling_benchmark",
        "Scaling benchmark",
        (
            "Wall-clock time and maximum bond dimension as a function of "
            "system size for $N = 2, 4, 6, 8$ qubits at fixed bond "
            "dimension. Includes a CPU vs GPU comparison when CUDA is "
            "available."
        ),
    ),
]


def split_cells(source: str) -> list[str]:
    """Split a Python script into cells along blank-line + comment-block boundaries.

    A new cell starts at any `# ---` or `# ===` comment line that begins a
    new section, or after every ~30 lines as a fallback.
    """
    lines = source.splitlines()
    cells: list[list[str]] = [[]]

    def is_section_header(line: str, prev: str) -> bool:
        # Section markers used in the example scripts
        if re.match(r"^#\s*[-=]{3,}", line):
            return True
        return False

    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        prev = lines[i - 1] if i > 0 else ""
        if is_section_header(line, prev) and cells[-1]:
            cells.append([])
        cells[-1].append(line)
        i += 1

    return ["\n".join(c).rstrip() + "\n" for c in cells if any(l.strip() for l in c)]


def make_cell(source: list[str], cell_type: str = "code") -> dict:
    text = source if isinstance(source, str) else "".join(source)
    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": text.splitlines(keepends=True),
        }
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def build_notebook(source: str, title: str, description: str) -> dict:
    intro_md = (
        f"# {title}\n"
        f"\n"
        f"{description}\n"
        f"\n"
        f"> Auto-generated from the matching script under `examples/`. "
        f"Run the cells in order to reproduce the figures.\n"
    )

    cells: list[dict] = [make_cell(intro_md, cell_type="markdown")]
    for chunk in split_cells(source):
        cells.append(make_cell(chunk, cell_type="code"))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    for script_name, stem, title, desc in JOBS:
        src_path = EXAMPLES / script_name
        if not src_path.exists():
            print(f"  skip — {src_path} not found")
            continue
        source = src_path.read_text(encoding="utf-8")
        nb = build_notebook(source, title, desc)
        out_path = NOTEBOOKS_DIR / f"{stem}.ipynb"
        out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"  wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
