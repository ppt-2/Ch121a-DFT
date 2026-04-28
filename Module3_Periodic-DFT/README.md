[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# Ch121a — Module 3: Periodic DFT

**Course**: Ch121a — Quantum Chemistry & DFT in Practice, Caltech  
**Co-instructor**: Dr. Prabhat Prakash

This module covers periodic (solid-state) DFT from conceptual foundations through practical workflows, example calculations, and machine-learned force fields.

---

## Structure

```
Module3_Periodic-DFT/
├── intro.md                       # Jupyter Book landing page
├── _toc.yml                       # Jupyter Book table of contents
├── README.md                      # This file
├── notebooks/
│   ├── 01_finite_to_periodic_dft.ipynb
│   ├── 02_planewaves_pseudopotentials_kpoints.ipynb
│   ├── 03a_codes_workflow_visualization.ipynb
│   ├── 03b_Bader_NEB_ImDimer.ipynb
│   ├── 04a_example_calculations_basic.ipynb
|   ├── 04b_example_calculations_advanced.ipynb
│   ├── 05_aimd_and_mlff.ipynb
│   └── 06_applications_of_pwdft.ipynb
└── tmp/
    ├── sample/
    |   └── Sample structures, scripts
    ├── Si_dos/
    │   ├── vasp/    INCAR_scf, INCAR_dos, KPOINTS_scf, KPOINTS_dos, POSCAR
    │   └── qe/      si_scf.in, si_nscf.in, si_dos.in
```

---

## Notebook Summaries

| # | Notebook | Topics |
|---|----------|--------|
| 01 | From Finite to Periodic DFT | Supercells, PBC, Bloch's theorem, reciprocal lattice, Brillouin zone, band structure vs. MO diagram, applicability |
| 02 | Plane-waves, Pseudopotentials & k-points | ENCUT convergence, PAW vs. USPP vs. NCpp, Monkhorst-Pack grids, smearing methods, DFT+U, vdW corrections |
| 03a | Codes, Workflow & Visualization | VASP input files (INCAR/POSCAR/POTCAR/KPOINTS), Quantum ESPRESSO pw.x, VESTA, vaspkit, pymatgen/Materials Project API |
| 03b | Bader, NEB and Improved dimer calculations |
| 04a | Example Calculations | Si DOS, band gap |
| 04b | Example Calculations | CO vs. H adsorption on Cu-100 surface |
| 05 | Ab-initio MD & MLFFs | BOMD with VASP, thermostat settings, MLFF architectures, VASP ML-FF, MACE-MP-0 universal potential |
| 06 | Applications of Plane-wave DFT | Catalysis, semiconductors, light-matter interactions, batteries |

---

## Software

| Software | Version | Purpose | License |
|----------|---------|---------|---------|
| [VASP](https://www.vasp.at) | ≥ 6.3 | Periodic DFT engine | Commercial |
| [Quantum ESPRESSO](https://www.quantum-espresso.org) | ≥ 7.2 | Open-source periodic DFT | GPL |
| [VESTA](https://jp-minerals.org/vesta/en/) | ≥ 3.5 | Crystal structure visualization | Free academic |
| [vaspkit](https://vaspkit.com) | ≥ 1.3 | VASP pre/post-processing | Free |
| [pymatgen](https://pymatgen.org) | ≥ 2024.1 | Materials informatics, MP API | MIT |
| [ASE](https://wiki.fysik.dtu.dk/ase/) | ≥ 3.23 | Atomic simulation environment | LGPL-2.1 |

---

## Sample Inputs

All VASP and Quantum ESPRESSO input files used in Notebook 04 are provided under `tmp/`. See `intro.md` for the full directory tree.

---

## License

GNU General Public License v3.0. See [LICENSE](../LICENSE).
