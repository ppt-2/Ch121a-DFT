[![Jupyter Book](https://jupyterbook.org/en/stable/_images/badge.svg)](https://ppt-2.github.io/Ch121a-DFT)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

# Ch121a — Module 3: Periodic DFT — From Theory to Practice

_Course Instructor: Prof. William A. Goddard III | Caltech_

An interactive Jupyter Book for **Module 3** of Ch121a. This module takes you from the conceptual foundations of periodic (solid-state) DFT all the way to running real calculations with VASP and Quantum ESPRESSO, visualizing results, and using machine-learned force fields.

---

## Notebooks

| # | Title | Description |
|---|-------|-------------|
| 01 | From Finite to Periodic DFT | Supercells, PBC, Bloch's theorem, Brillouin zone, band structure |
| 02 | Plane-waves, Pseudopotentials & k-points | ENCUT, PAW, Monkhorst-Pack, smearing, DFT+U, vdW |
| 03 | Codes, Workflow & Visualization | VASP input files, Quantum ESPRESSO, VESTA, vaspkit, pymatgen |
| 04 | Example Calculations | Si DOS, TiO₂ band gap, graphene bands, Co-Pc+CO spin state |
| 05 | Ab-initio MD & Machine-learned Force-fields | AIMD, BOMD, MLFFs, MACE, universal potentials |
| 06 | Applications of Plane-wave DFT | Catalysis, semiconductors, light–matter interactions, batteries |

---

## Software Prerequisites

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| [VASP](https://www.vasp.at) | ≥ 6.3 | Periodic DFT engine (licensed) | Commercial |
| [Quantum ESPRESSO](https://www.quantum-espresso.org) | ≥ 7.2 | Open-source periodic DFT | GPL |
| [VESTA](https://jp-minerals.org/vesta/en/) | ≥ 3.5 | Crystal structure visualization | Free academic |
| [vaspkit](https://vaspkit.com) | ≥ 1.3 | VASP pre/post-processing toolkit | Free |
| [pymatgen](https://pymatgen.org) | ≥ 2024.1 | Materials informatics, MP API | MIT |
| [ASE](https://wiki.fysik.dtu.dk/ase/) | ≥ 3.23 | Atomic simulation environment | LGPL-2.1 |
| [numpy](https://numpy.org) | ≥ 1.24 | Numerical computing | BSD |
| [matplotlib](https://matplotlib.org) | ≥ 3.7 | Plotting | PSF |

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/ppt-2/Ch121a-DFT.git
cd Ch121a-DFT/Module3_Periodic-DFT

# 2. Install Python dependencies
pip install numpy matplotlib pymatgen ase

# 3. Launch JupyterLab
jupyter lab
```

The runnable Python cells in each notebook use only **numpy** and **matplotlib** (always available) or **pymatgen/ASE** wrapped in `try/except` blocks so they degrade gracefully if those packages are absent.

## How to Run on Google Colab

Each notebook has an **"Open in Colab"** badge at the top. Click it to open directly in Google Colab — no local installation needed.

---

## Sample Input Files

VASP and Quantum ESPRESSO input files for all example calculations are provided in `tmp/`:

```
tmp/
├── Si_dos/vasp/          INCAR_scf, INCAR_dos, KPOINTS_scf, KPOINTS_dos, POSCAR
│   └── qe/               si_scf.in, si_nscf.in, si_dos.in
├── TiO2_bandgap/vasp/    INCAR_scf_gga, INCAR_scf_dftu, KPOINTS, POSCAR
│   └── qe/               tio2_scf.in
├── graphene_bands/vasp/  INCAR_scf, INCAR_bands, KPOINTS_scf, KPOINTS_bands, POSCAR
│   └── qe/               graphene_scf.in, graphene_bands.in
└── CoPc_CO_spin/vasp/    INCAR_highspin, INCAR_lowspin, KPOINTS, POSCAR_CoPc_CO
```
