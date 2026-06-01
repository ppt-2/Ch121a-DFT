[![Jupyter Book](https://jupyterbook.org/en/stable/_images/badge.svg)](https://ppt-2.github.io/Ch121a-DFT)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

# Ch121a: Atomic-Level Simulations of Materials and Molecules
_Course Instructor: Dr. Prabhat Prakash, Prof. William A. Goddard III_

## Anouncements for SP 2026

| 0 | New scripts are added to [Module1_DFT/data](Module1_DFT/data) useful for some initial ORCA code workflows (xtb, fast-DFT, potential energy scans). 

| 1 | For some of the proposed projects, practical workflow is now added here:
[Module1_DFT/ Notebook 5](https://github.com/ppt-2/Ch121a-DFT/blob/main/Module1_DFT/notebooks_v2/05_practical_calculations.ipynb).**  

| 2 | Sometimes, github does not render the notebooks properly. You can read the .md (markdown files) version of all notebooks kept in '[notebook-all-readable](notebook-all-readable)' folder.** 

| 3 | Important instructions relevant to this class, hpc login, vi editor, linux commands, initial scripts are kept in '[Initialization](Initialization)' directory** 

---

## Module Overview

| # | Title |
|---|-------|
| 01 | Quantum Chemistry (theory) and DFT for finite structures |
| 02 | Molecular Dynamics |
| 03 | DFT in condensed phase, solid-state electronic structure, MLIPs |
| -- | All notebooks in readable format are kept in 'notebook-all-readable' |
---

## Installation

### pip

```bash
git clone https://github.com/ppt-2/Ch121a-DFT.git
cd Ch121a-DFT
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

### conda / mamba

```bash
git clone https://github.com/ppt-2/Ch121a-DFT.git
cd Ch121a-DFT
conda env create -f environment.yml
conda activate ch121a-dft
jupyter lab
```

---

## Quick Start

```python
# Run a DFT calculation on water in 5 lines
from pyscf import gto, dft

mol = gto.M(atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469",
            basis="def2-SVP", charge=0, spin=0)
mf = dft.RKS(mol, xc="b3lyp").run()
print(f"B3LYP/def2-SVP energy: {mf.e_tot:.6f} Hartree")
```

---

## Software

| Software | Version | Role | License |
|----------|---------|------|---------|
| [PySCF](https://pyscf.org) | ≥ 2.5.0 | Open-source QC engine | Apache-2.0 |
| [ASE](https://wiki.fysik.dtu.dk/ase/) | ≥ 3.23.0 | Atomic simulation / I/O | LGPL-2.1 |
| [ORCA](https://orcaforum.kofo.mpg.de) | 6.x | Production DFT/WFT code | Free academic |
| [Jaguar](https://www.schrodinger.com/jaguar) | 11.x | Commercial DFT, LMP2 | Commercial |
| [cclib](https://cclib.github.io) | ≥ 1.8.1 | Parse QC output files | BSD-3 |
| [py3Dmol](https://3dmol.csb.pitt.edu) | ≥ 2.1.0 | 3-D molecular viewer | BSD-3 |
| [VASP](https://vasp.at/wiki/The_VASP_Manual) | ≥ 6.3.2 | Periodic DFT | Commercial |

---

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/ppt-2/Ch121a-DFT).

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Commit your changes and open a pull request against `main`

Please follow existing notebook style and ensure all cells execute cleanly (`jupyter nbconvert --to notebook --execute`).

---

## Citation

If you use this material in your research or teaching, please cite:

```bibtex
@misc{ch121a-dft,
  author       = {Prakash, Prabhat; Goddard III, William A.},
  title        = {{Ch121a - Quantum Chemistry \& DFT in Practice}},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/ppt-2/Ch121a-DFT}
}
```
---

## License

This project is licensed under the **GNU General Public License v3.0**.
See [LICENSE](LICENSE) for details.
