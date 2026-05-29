# 06-pre — Building a Water Box with PACKMOL & Visualizing with OVITO

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ppt-2/Ch121a-DFT/blob/main/Module%202%20-%20Molecular%20dynamics/notebooks/06_pre_packmol_ovito.ipynb)

**Ch121a — Quantum Mechanics and its Applications** | Module 2: Molecular Dynamics

---

Two essential tools:

1. **PACKMOL** — pack molecules into a simulation box at a target density
2. **OVITO** — visualize and analyze the resulting structure and trajectories

## 🎯 Learning Objectives

- Write a single-molecule PDB/XYZ file for a water molecule
- Compute the box length needed for 216 water molecules at liquid density
- Write and run a PACKMOL input file to produce a packed water box
- Load and render molecular structures in OVITO
- Compute the radial distribution function (RDF) and coordination number with OVITO's Python API


```python
pip install packmol
```

    Collecting packmol
      Downloading packmol-21.2.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.5 kB)
    Downloading packmol-21.2.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (1.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.2/1.2 MB[0m [31m20.6 MB/s[0m  [33m0:00:00[0m
    [?25hInstalling collected packages: packmol
    Successfully installed packmol-21.2.1
    Note: you may need to restart the kernel to use updated packages.


---
## Part 1 — PACKMOL

### 1.1 What is PACKMOL?

[PACKMOL](https://m3g.github.io/packmol/) (PAcking Optimization for Molecular dynamics simulations) places a specified number of molecules inside a defined geometric region — sphere, box, cylinder, *etc.* — without overlapping atomic radii. It minimises a penalty function of pairwise distances to reach a configuration suitable as an MD starting point.

| Feature | Details |
|---------|--------|
| Input/output formats | PDB, XYZ, mol2, tinker |
| Geometry constraints | cube, sphere, cylinder, shell, plane |
| Mixed systems | multiple species, solvation, membranes |
| Availability | free, open source (MIT); conda / pip / binary |

### 1.2 Installation

```bash
# Option A — conda (recommended)
conda install -c conda-forge packmol

# Option B — build from source (Linux/macOS)
git clone https://github.com/m3g/packmol.git && cd packmol && make
```

Verify with `packmol --version` (or `packmol < /dev/null 2>&1 | head -2`).

### 1.3 Box-Size Calculation

Liquid water at 298 K and 1 atm has a density of $\rho \approx 0.997\,\text{g cm}^{-3}$.

For $N = 216$ molecules (molar mass $M = 18.015\,\text{g mol}^{-1}$):

$$L = \left(\frac{N\,M}{\rho\,N_A}\right)^{1/3}$$


```python
# =============================================================================
# Ch121a: Molecular Dynamics — Notebook 06-pre: PACKMOL + OVITO
# License: GPL-3.0 (https://www.gnu.org/licenses/gpl-3.0.en.html)
# =============================================================================
import numpy as np
import os, subprocess
from pathlib import Path

# --- Box-size calculation ---
N          = 216          # number of water molecules
M_water    = 18.015       # g/mol
rho        = 0.997        # g/cm^3
N_A        = 6.02214076e23

V_cm3 = (N * M_water) / (rho * N_A)          # cm^3
V_A3  = V_cm3 * 1e24                          # Angstrom^3  (1 cm = 1e8 A)
L_A   = V_A3 ** (1/3)

print(f'N molecules : {N}')
print(f'Box volume  : {V_A3:.2f}  Å³')
print(f'Box length  : {L_A:.3f}  Å  ({L_A/10:.3f} nm)')
```

    N molecules : 216
    Box volume  : 6481.00  Å³
    Box length  : 18.644  Å  (1.864 nm)


### 1.4 Single-Water Molecule File

PACKMOL reads molecule templates from PDB or XYZ files.  
Below we write a PDB file for one SPC/E water molecule (O at origin, H₁ and H₂ placed according to the experimental geometry).

| Atom | x (Å) | y (Å) | z (Å) |
|------|-------|-------|-------|
| O    | 0.000 | 0.000 | 0.000 |
| H₁   | 0.957 | 0.000 | 0.000 |
| H₂   |−0.240 | 0.927 | 0.000 |


```python
WORK_DIR = Path('packmol_water')
WORK_DIR.mkdir(exist_ok=True)

# --- single water molecule (SPC/E geometry) ---
water_pdb = """\
ATOM      1  OW  WAT A   1       0.000   0.000   0.000  1.00  0.00           O
ATOM      2  HW1 WAT A   1       0.957   0.000   0.000  1.00  0.00           H
ATOM      3  HW2 WAT A   1      -0.240   0.927   0.000  1.00  0.00           H
END
"""

(WORK_DIR / 'water.pdb').write_text(water_pdb)
print('water.pdb written:')
print(water_pdb)
```

    water.pdb written:
    ATOM      1  OW  WAT A   1       0.000   0.000   0.000  1.00  0.00           O
    ATOM      2  HW1 WAT A   1       0.957   0.000   0.000  1.00  0.00           H
    ATOM      3  HW2 WAT A   1      -0.240   0.927   0.000  1.00  0.00           H
    END
    


### 1.5 PACKMOL Input File

A PACKMOL input file has three parts:

```
tolerance  <min_dist>   # minimum inter-atomic distance in Å
output     <file>       # output filename
filetype   <fmt>        # pdb, xyz, …

structure  <template>   # start of a molecule block
  number   <N>          # how many copies to place
  inside   box  xmin ymin zmin  xmax ymax zmax
end structure
```

We add a small **2 Å margin** on each face so molecules are not placed right at the periodic boundary.


```python
margin = 2.0          # Å
lo     = margin
hi     = L_A - margin

packmol_inp = f"""\
tolerance 2.0
output    {WORK_DIR}/water_box.pdb
filetype  pdb

structure {WORK_DIR}/water.pdb
  number   {N}
  inside   box  {lo:.3f} {lo:.3f} {lo:.3f}  {hi:.3f} {hi:.3f} {hi:.3f}
end structure
"""

inp_path = WORK_DIR / 'pack_water.inp'
inp_path.write_text(packmol_inp)
print(packmol_inp)
```

    tolerance 2.0
    output    packmol_water/water_box.pdb
    filetype  pdb
    
    structure packmol_water/water.pdb
      number   216
      inside   box  2.000 2.000 2.000  16.644 16.644 16.644
    end structure
    


### 1.6 Running PACKMOL


```python
result = subprocess.run(
    ['packmol'],
    input=inp_path.read_text(),
    capture_output=True, text=True
)

# Print the last few lines of PACKMOL's output
for line in result.stdout.splitlines()[-20:]:
    print(line)

if result.returncode != 0:
    print('\n--- STDERR ---')
    print(result.stderr)
else:
    out_file = WORK_DIR / 'water_box.pdb'
    n_atoms = sum(1 for ln in out_file.read_text().splitlines() if ln.startswith('ATOM'))
    print(f'\n✓ Output: {out_file}  ({n_atoms} atoms = {n_atoms//3} molecules)')
```

### 1.7 Quick Sanity Check — Reading the PDB

Parse the output PDB with plain Python and confirm atom counts and approximate density.


```python
import re

coords = {'O': [], 'H': []}
pdb_lines = (WORK_DIR / 'water_box.pdb').read_text().splitlines()

for ln in pdb_lines:
    if not ln.startswith('ATOM'):
        continue
    element = ln[76:78].strip()
    x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
    if element in coords:
        coords[element].append([x, y, z])

O_xyz = np.array(coords['O'])
H_xyz = np.array(coords['H'])

n_mol     = len(O_xyz)
mass_g    = n_mol * M_water / N_A
vol_cm3   = (L_A * 1e-8) ** 3
rho_calc  = mass_g / vol_cm3

print(f'O atoms   : {len(O_xyz)}   (= {n_mol} molecules)')
print(f'H atoms   : {len(H_xyz)}')
print(f'Density   : {rho_calc:.3f} g/cm³  (target: {rho} g/cm³)')
print(f'O centroid: {O_xyz.mean(axis=0).round(2)} Å')
```

### 1.8 Matplotlib Preview

A quick 3-D scatter plot of oxygen positions to confirm uniform packing.


```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(5, 5))
ax  = fig.add_subplot(111, projection='3d')

ax.scatter(O_xyz[:, 0], O_xyz[:, 1], O_xyz[:, 2],
           c='royalblue', s=20, alpha=0.6, label='O')

ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)'); ax.set_zlabel('z (Å)')
ax.set_title(f'PACKMOL water box — {n_mol} molecules')
ax.legend()
plt.tight_layout()
plt.show()
```

---
## Part 2 — OVITO

### 2.1 What is OVITO?

[OVITO](https://www.ovito.org/) (Open Visualization Tool) is a scientific visualization and analysis program for atomistic simulation data. It provides:

- A **GUI** for interactive exploration of trajectories
- A **Python API** (`ovito` package) for scripted, reproducible analysis

| Feature | Details |
|---------|--------|
| Formats | LAMMPS dump, GROMACS XTC, PDB, XYZ, CIF, … |
| Modifiers | RDF, CNA, Voronoi, polyhedral template matching, … |
| Output | PNG/TIFF images, MP4 movies, CSV data |
| License | Basic free; Pro commercial |

### 2.2 Installation

```bash
pip install ovito          # Python API (ovito-basic)
```

The GUI version can be downloaded from [https://www.ovito.org/](https://www.ovito.org/).
Basic version is free. 


```python
# Install if not already present (uncomment in Colab)
# !pip install -q ovito

import ovito
print(f'OVITO version: {ovito.version_string}')
```

### 2.3 Loading the Water Box

The OVITO Python API is built around three classes:

| Class | Role |
|-------|------|
| `Pipeline` | chain of data source + modifiers |
| `DataCollection` | snapshot of particles, bonds, cell, … |
| `Modifier` | transforms data (RDF, CNA, slice, …) |


```python
from ovito.io import import_file

pipeline = import_file(str(WORK_DIR / 'water_box.pdb'))
data     = pipeline.compute()          # evaluate pipeline at frame 0

particles = data.particles
print(f'Particles   : {particles.count}')
print(f'Particle types: {set(particles.particle_types.types[t.id].name for t in particles.particle_types.types)}')
cell = data.cell
print(f'Simulation cell:\n{cell.matrix}')
```

### 2.4 Rendering a Static Image

OVITO's `Viewport` class renders frames off-screen — no display required.


```python
from ovito.vis import Viewport, TachyonRenderer
from IPython.display import Image

# Attach the pipeline to the scene so the viewport can see it
pipeline.add_to_scene()

vp = Viewport(type=Viewport.Type.Ortho)
vp.zoom_all()               # auto-fit camera to bounding box

render_path = str(WORK_DIR / 'water_box.png')
vp.render_image(
    size=(600, 600),
    filename=render_path,
    background=(1, 1, 1),   # white background
    renderer=TachyonRenderer()
)

pipeline.remove_from_scene()
Image(render_path)
```

### 2.5 Computing the Radial Distribution Function (RDF)

The O–O RDF $g_{OO}(r)$ characterises the liquid structure and is the primary validation observable for water models.

$$g(r) = \frac{V}{N^2} \left\langle \sum_{i \neq j} \delta(r - r_{ij}) \right\rangle \cdot \frac{1}{4\pi r^2 \Delta r}$$

The first peak of $g_{OO}$ at ~2.8 Å corresponds to the first hydrogen-bond shell.


```python
from ovito.modifiers import CoordinationAnalysisModifier

cutoff    = 8.0     # Å — maximum r for RDF
n_bins    = 200

pipeline.modifiers.append(
    CoordinationAnalysisModifier(
        cutoff=cutoff,
        number_of_bins=n_bins,
        partial=True        # compute partial RDFs by type
    )
)

data_rdf = pipeline.compute()

# Extract RDF table
rdf_table = data_rdf.tables['coordination-rdf']
r         = rdf_table.axis_x                        # bin centres (Å)
rdf_vals  = rdf_table.y                             # shape (n_bins, n_partials)

print(f'RDF table shape : {rdf_vals.shape}')
print(f'Partial labels  : {[c.name for c in rdf_table.y.component_names]}')
```


```python
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'figure.dpi': 120, 'font.size': 11,
                     'axes.labelsize': 12, 'legend.fontsize': 10})

fig, ax = plt.subplots(figsize=(6, 4))

component_names = [c.name for c in rdf_table.y.component_names]
colors = ['royalblue', 'tomato', 'seagreen']

for idx, name in enumerate(component_names):
    ax.plot(r, rdf_vals[:, idx], lw=1.5,
            label=f'g$_{{\\rm {name.replace("-","-")}}}$(r)',
            color=colors[idx % len(colors)])

ax.axhline(1, color='k', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel(r'$r$ (Å)')
ax.set_ylabel(r'$g(r)$')
ax.set_title('Partial RDFs — PACKMOL water box (216 molecules)')
ax.legend()
ax.set_xlim(0, cutoff)
plt.tight_layout()
plt.show()
```

### 2.6 Coordination Number

Integrating the O–O RDF to the first minimum (~3.5 Å) gives the **coordination number** — the mean number of nearest-neighbour oxygens.


```python
# Number density of oxygens
n_oxygen   = len(O_xyz)
V_A3_cell  = np.linalg.det(data.cell.matrix[:, :3])   # actual cell volume from OVITO
rho_O      = n_oxygen / V_A3_cell                      # atoms/Å^3

# Find the O-O partial (usually index 0 for a single-component system)
oo_label = [n for n in component_names if n.upper() in ('OW-OW', 'O-O', '1-1')]
oo_idx   = component_names.index(oo_label[0]) if oo_label else 0

g_oo = rdf_vals[:, oo_idx]
dr   = r[1] - r[0]

# Integrate up to first minimum (~3.5 Å)
r_cut  = 3.5
mask   = r <= r_cut
coord  = 4 * np.pi * rho_O * np.trapz(g_oo[mask] * r[mask]**2, r[mask])

print(f'O–O partial index : {oo_idx} ({component_names[oo_idx]})')
print(f'Number density ρ_O: {rho_O:.5f} Å⁻³')
print(f'Coordination number (r < {r_cut} Å): {coord:.2f}')
print('(liquid water at 298 K: ~4.4 nearest neighbours)')
```

> **Note:** Because the PACKMOL output is a *random packed* configuration rather than an equilibrated liquid, the RDF will not yet show the sharp first peak characteristic of equilibrated liquid water.  Run the structure through an MD equilibration (see **Notebook 06**) and re-analyze to obtain the correct $g_{OO}(r)$.

### 2.7 Brief OVITO GUI Workflow

The same steps above can be performed interactively in the OVITO desktop application:

1. **File → Open** — load `water_box.pdb`
2. **Add modifier → Coordination analysis** — set cutoff to 8 Å, enable *Compute partial RDFs*
3. Inspect the plotted $g(r)$ in the *Data inspector* panel
4. **Rendering → Render active viewport** — choose *Tachyon* renderer for publication-quality images
5. **File → Export file** — export to LAMMPS data or XYZ for the next simulation step

A typical OVITO viewport for a water box looks like this (atom colours: O = red, H = white):

```
┌─────────────────────────────┐
│  ·  ·  · ○  ·  ○  ·  ·  · │
│  ○  ·  ·  ·  ·  ·  ○  ·  ○│
│  ·  ○  ·  ·  ○  ·  ·  ·  ·│
│  ·  ·  ·  ○  ·  ·  ○  ·  ·│
└─────────────────────────────┘
       216 H₂O  @ 0.997 g/cm³
```

---
## Summary

| Step | Tool | Output |
|------|------|--------|
| Define box size | Python (NumPy) | $L \approx 18.6$ Å for 216 H₂O |
| Create template | text editor / Python | `water.pdb` (3 atoms) |
| Pack molecules | **PACKMOL** | `water_box.pdb` (648 atoms, 216 molecules) |
| Visualise | **OVITO** GUI or Python API | rendered PNG, trajectory movies |
| Compute RDF | **OVITO** `CoordinationAnalysisModifier` | $g_{OO}(r)$, coordination number |

---

### Further Reading

- Martínez *et al.*, *J. Comput. Chem.* **30**, 2157 (2009) — original PACKMOL paper
- Stukowski, *Modelling Simul. Mater. Sci. Eng.* **18**, 015012 (2010) — OVITO paper
- [PACKMOL documentation](https://m3g.github.io/packmol/)
- [OVITO Python API reference](https://www.ovito.org/docs/current/python/)
