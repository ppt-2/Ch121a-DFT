# Ch121a — Quantum Chemistry & DFT in Practice: From Fundamentals to Research

Welcome to the interactive notebook for **Ch121a: Quantum Chemistry & Density Functional Theory in Practice** at Caltech. This Jupyter Book provides hands-on computational chemistry tutorials (very basic, in computatioanl cost) that bridge theoretical foundations with modern research workflows.

:::{note}
This book is released under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html). You are free to use, modify, and redistribute all materials provided you preserve the same license.
:::

---

## Purpose

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

## How to Run Locally with pip (recommended for most users)

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
**On HPC, it should run well on Interactive webpage, where one can launch a Jupyter lab environment very easily. One can also create a Miniconda env that can be loaded as preferred kernel for these notebooks. 


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


## Acknowledgments

This course material builds on the open-source scientific Python ecosystem. Key software acknowledgments:

- **PySCF**: Q. Sun *et al.*, *WIREs Comput. Mol. Sci.* **2018**, *8*, e1340; Q. Sun *et al.*, *J. Chem. Phys.* **2020**, *153*, 024109.
- **ORCA**: F. Neese, *WIREs Comput. Mol. Sci.* **2012**, *2*, 73–78; F. Neese *et al.*, *WIREs Comput. Mol. Sci.* **2022**, *12*, e1606.
- **Jaguar**: Schrödinger, LLC, *Jaguar*, version 11, New York, NY, 2023. See also: A. D. Bochevarov *et al.*, *Int. J. Quantum Chem.* **2013**, *113*, 2110–2142.
- **ASE**: A. H. Larsen *et al.*, *J. Phys.: Condens. Matter* **2017**, *29*, 273002.
- **cclib**: N. M. O'Boyle *et al.*, *J. Comput. Chem.* **2008**, *29*, 839–845.
