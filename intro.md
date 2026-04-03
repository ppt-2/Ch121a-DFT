# Ch121a — Quantum Chemistry & DFT in Practice: From Fundamentals to Research

Welcome to the interactive textbook for **Ch121a: Quantum Chemistry & Density Functional Theory in Practice** at Caltech. This Jupyter Book provides hands-on computational chemistry tutorials that bridge theoretical foundations with modern research workflows.

:::{note}
This book is released under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html). You are free to use, modify, and redistribute all materials provided you preserve the same license.
:::

---

## Purpose

This course takes you from the fundamentals of quantum chemistry — wave functions, the Schrödinger equation, basis sets — all the way to practical density functional theory (DFT) calculations used in current research. By the end, you will be able to:

- Set up and run DFT calculations with **PySCF** (open-source) and production codes **ORCA** and **Jaguar**
- Optimize molecular geometries and compute vibrational spectra
- Analyze electronic structure: charges, bond orders, frontier orbitals, and NBO/NPA
- Study transition metal complexes, open-shell systems, and spin states
- Model excited states with TD-DFT
- Incorporate solvation effects with implicit solvent models

All notebooks are self-contained and include worked examples on real molecules (H₂O, benzene, ferrocene, Fe(CO)₅, Mn-porphyrin, and more).

---

## Software Prerequisites

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| [PySCF](https://pyscf.org) | ≥ 2.5.0 | Open-source quantum chemistry engine | Apache-2.0 |
| [ASE](https://wiki.fysik.dtu.dk/ase/) | ≥ 3.23.0 | Atomic simulation environment, I/O | LGPL-2.1 |
| [py3Dmol](https://3dmol.csb.pitt.edu) | ≥ 2.1.0 | Interactive 3-D molecular visualization | BSD-3 |
| [nglview](https://nglviewer.org) | ≥ 3.1.0 | Jupyter widget for molecular structures | MIT |
| [cclib](https://cclib.github.io) | ≥ 1.8.1 | Parse ORCA/Jaguar/Gaussian output files | BSD-3 |
| [Jupyter Book](https://jupyterbook.org) | ≥ 1.0.0 | Build this interactive book | BSD-3 |

---

## Chapter Overview

| # | Title | Key Concepts | Molecules | Est. Runtime |
|---|-------|-------------|-----------|-------------|
| 00 | Setup & Tools | Environment setup, PySCF basics, visualization | H₂O, CH₄ | < 1 min |
| 01 | Molecular Structure & Basis Sets | XYZ format, basis set convergence, def2-SVP vs cc-pVTZ | H₂O, HF, N₂ | 2–5 min |
| 02 | Hartree-Fock Theory | SCF procedure, HF energy, Koopman's theorem, HOMO/LUMO | H₂, H₂O, NH₃ | 2–5 min |
| 03 | DFT Fundamentals | Exchange-correlation functionals, LDA/GGA/hybrid, dispersion | benzene dimer | 5–10 min |
| 04 | Geometry Optimization | Gradient-based optimization, convergence criteria, PES scans | H₂O, ethanol | 5–10 min |
| 05 | Vibrational Analysis & Thermochemistry | Normal modes, IR spectra, ZPE, ΔG at 298 K | H₂O, CO₂, CH₄ | 5–10 min |
| 06 | Molecular Properties | Dipole moment, polarizability, NMR shielding | H₂O, benzene | 5–10 min |
| 07 | Reaction Energetics | Reaction enthalpies, isodesmic reactions, bond dissociation | H₂ + F₂ → 2 HF | 5–10 min |
| 08 | Transition Metal Complexes | d-orbital splitting, spin states, Cr(NH₃)₆³⁺, ferrocene | ferrocene, Fe(CO)₅ | 10–20 min |
| 09 | Population Analysis & Bonding | Mulliken, NPA charges, Wiberg bond orders, NBO | H₂O, NH₃, BH₃ | 5–10 min |
| 10 | Excited States & TD-DFT | Linear-response TDDFT, UV/Vis spectra, charge-transfer | benzene, Mn-porphyrin | 10–20 min |
| 11 | Solvation Effects | PCM, COSMO, aqueous vs. organic solvents, pKa shifts | acetic acid, H₂O | 5–10 min |
| 12 | ORCA: Capabilities & Practical Guide | ORCA input syntax, compound methods, RIJCOSX, DLPNO | Fe(CO)₅, ferrocene | reference |
| 13 | Jaguar: Capabilities & Practical Guide | Jaguar input/output, Maestro integration, LMP2, GVB | transition metals | reference |

---

## How to Run Locally

### Option A — pip (recommended for most users)

```bash
# 1. Clone the repository
git clone https://github.com/ppt-2/Ch121a-DFT.git
cd Ch121a-DFT

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Launch JupyterLab
jupyter lab
```

### Option B — conda / mamba

```bash
# 1. Clone the repository
git clone https://github.com/ppt-2/Ch121a-DFT.git
cd Ch121a-DFT

# 2. Create the conda environment
conda env create -f environment.yml
conda activate ch121a-dft

# 3. Launch JupyterLab
jupyter lab
```

### Building the Jupyter Book

```bash
jupyter-book build .
# Open _build/html/index.html in your browser
```

---

## How to Run on Google Colab

Each notebook has a **"Open in Colab"** badge at the top. Click it to open in Google Colab. The first cell in each notebook installs the required packages automatically:

```python
# Auto-install block (runs only on Colab)
import sys
if "google.colab" in sys.modules:
    !pip install -q pyscf ase py3Dmol nglview cclib
```

No local installation is needed — all computation runs in the Colab cloud environment.

---

## Research vs. Teaching Philosophy

This book is deliberately dual-purpose:

**For students** — each notebook starts with brief theory, walks through worked examples step-by-step, and includes exercises with model answers. All calculations use small, fast molecules so that notebooks complete in minutes on a laptop.

**For researchers** — production-quality input file templates for ORCA and Jaguar are provided in Chapters 12–13. Realistic output files in `data/sample_outputs/` demonstrate how to parse results programmatically with `cclib`. The molecule library in `data/molecules/` covers representative organic, inorganic, and organometallic systems.

The philosophy: understand the method deeply enough to know when to trust (and when to distrust) the numbers your software produces.

---

## Acknowledgments

This course material builds on the open-source scientific Python ecosystem. Key software acknowledgments:

- **PySCF**: Q. Sun *et al.*, *WIREs Comput. Mol. Sci.* **2018**, *8*, e1340; Q. Sun *et al.*, *J. Chem. Phys.* **2020**, *153*, 024109.
- **ORCA**: F. Neese, *WIREs Comput. Mol. Sci.* **2012**, *2*, 73–78; F. Neese *et al.*, *WIREs Comput. Mol. Sci.* **2022**, *12*, e1606.
- **Jaguar**: Schrödinger, LLC, *Jaguar*, version 11, New York, NY, 2023. See also: A. D. Bochevarov *et al.*, *Int. J. Quantum Chem.* **2013**, *113*, 2110–2142.
- **ASE**: A. H. Larsen *et al.*, *J. Phys.: Condens. Matter* **2017**, *29*, 273002.
- **cclib**: N. M. O'Boyle *et al.*, *J. Comput. Chem.* **2008**, *29*, 839–845.

Course development supported by the Division of Chemistry and Chemical Engineering, California Institute of Technology.
