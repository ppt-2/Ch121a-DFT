**Ch121a | Module 3: Periodic DFT**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ppt-2/Ch121a-DFT/blob/main/Module3_Periodic-DFT/notebooks/04b_adsorption_energy.ipynb)

# Notebook 4b: Advanced Example: CO vs. H Adsorption Energy on Cu(100)

---

## Objectives

- A periodic slab model of a Cu(100) surface
- Adsorbate+slab and clean-slab pwDFT calculations
- Compute adsorption energies of CO and H on Cu(100)
- Understand the hollow, bridge, and top adsorption sites
- Compare CO vs. H adsorption energies and interpret the results
- Compute free energy corrections (ZPE + thermal) via VASP frequency calculations

## 1. Adsorption Energy:

The adsorption energy is defined as:

$$E_{\rm ads} = E_{\rm slab+ads} - E_{\rm slab} - E_{\rm ads(gas)}$$

where:
- $E_{\rm slab+ads}$ — total energy of the slab with the adsorbate
- $E_{\rm slab}$ — total energy of the clean slab (same geometry, same number of layers)
- $E_{\rm ads(gas)}$ — total energy of the isolated adsorbate molecule in vacuum

A **negative** $E_{\rm ads}$ means exothermic (favorable) adsorption.

| Adsorbate | Typical $E_{\rm ads}$ on Cu(100) (PBE) | Notes |
|-----------|----------------------------------------|-------|
| CO | −0.5 to −0.8 eV | Top site preferred; PBE slightly underestimates exp. |
| H | −2.4 to −2.7 eV | Hollow site preferred; strong chemisorption |

> **Note**: These are PBE values. vdW corrections (IVDW=12) improve CO adsorption energies; H adsorption is less sensitive to dispersion.


```python
## A minimal pymatgen script to generate slabs, needs bulk structure 
## Make sure you have pymatgen, if not, use pip install for local env
## Also copy this script as a .py file to save it seperately. 

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.core.surface import SlabGenerator

# Load the bulk structure from a POSCAR file
bulk_structure = Structure.from_file("../tmp/sample/Cu_mp-30_conventional_standard.cif")

# Define the miller index for the desired plane
miller_index = (1, 0, 0)  # Example: (1, 0, 0) plane

# Create a SlabGenerator object
slab_gen = SlabGenerator(bulk_structure, miller_index, min_slab_size=10, min_vacuum_size=15)

# Generate the slab
slabs = slab_gen.get_slabs()

# Save the first slab to a POSCAR file
slab = slabs[0]
Poscar(slab).write_file("../tmp/Cu_CO_H/Cu_100_conv.vasp")

print("Slab structure generated")
```

    Slab structure generated



```python
from ase.io import read
import py3Dmol
import numpy as np

atoms = read("/resnick/groups/wag/pp-4-ch121a/test-Cu_CO/CONTCAR")
atoms = atoms.repeat((1,1,1))
symbols = atoms.get_chemical_symbols()
positions = atoms.get_positions()

xyz_str = f"{len(symbols)}\nPOSCAR\n"
for s, (x, y, z) in zip(symbols, positions):
    xyz_str += f"{s} {x:.6f} {y:.6f} {z:.6f}\n"

cell = atoms.get_cell()      # 3x3 lattice vectors
a, b, c = cell[0], cell[1], cell[2]

origin = np.zeros(3)

corners = [
    origin,
    a,
    b,
    c,
    a + b,
    a + c,
    b + c,
    a + b + c
]

edges = [
    (0,1), (0,2), (0,3),
    (1,4), (1,5),
    (2,4), (2,6),
    (3,5), (3,6),
    (4,7), (5,7), (6,7)
]
view = py3Dmol.view(width=800, height=500)

# Add coordinates
view.addModel(xyz_str, "xyz")
view.setStyle({
    'sphere': {'scale': 0.5,
        'shininess': 80,
        'specular': 'white'
},
    'stick': {'radius': 0.15}
})
for i, j in edges:
    p1 = corners[i]
    p2 = corners[j]
    view.addLine({
        'start': {'x': float(p1[0]), 'y': float(p1[1]), 'z': float(p1[2])},
        'end':   {'x': float(p2[0]), 'y': float(p2[1]), 'z': float(p2[2])},
        'color': 'black',
        'radius': 0.08
    })

element_colors = {
    "H":  "#FFFFFF",   # white
    "O":  "#FF0D0D",   # red
    "Si": "#B0B7C6",   # steel (metallic gray-blue)
    "C":  "#808080",   # gray
    "Ti": "#FF77AA",   # pink
    "Mo": "#4F7942"
}
# Apply element-specific colors
for elem, color in element_colors.items():
    view.setStyle(
        {'elem': elem},
        {
            'sphere': {'scale': 0.3, 'color': color},
            'stick':  {'radius': 0.15, 'color': color}
        }
    )

view.setBackgroundColor('white')
view.setProjection("orthographic")
view.zoomTo()
view.show()
```


<div id="3dmolviewer_17777649540187144"  style="position: relative; width: 800px; height: 500px;">
        <p id="3dmolwarning_17777649540187144" style="background-color:#ffcccc;color:black">3Dmol.js failed to load for some reason.  Please check your browser console for error messages.<br></p>
        </div>
<script>

var loadScriptAsync = function(uri){
  return new Promise((resolve, reject) => {
    //this is to ignore the existence of requirejs amd
    var savedexports, savedmodule;
    if (typeof exports !== 'undefined') savedexports = exports;
    else exports = {}
    if (typeof module !== 'undefined') savedmodule = module;
    else module = {}

    var tag = document.createElement('script');
    tag.src = uri;
    tag.async = true;
    tag.onload = () => {
        exports = savedexports;
        module = savedmodule;
        resolve();
    };
  var firstScriptTag = document.getElementsByTagName('script')[0];
  firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
});
};

if(typeof $3Dmolpromise === 'undefined') {
$3Dmolpromise = null;
  $3Dmolpromise = loadScriptAsync('https://cdn.jsdelivr.net/npm/3dmol@2.5.4/build/3Dmol-min.js');
}

var viewer_17777649540187144 = null;
var warn = document.getElementById("3dmolwarning_17777649540187144");
if(warn) {
    warn.parentNode.removeChild(warn);
}
$3Dmolpromise.then(function() {
viewer_17777649540187144 = $3Dmol.createViewer(document.getElementById("3dmolviewer_17777649540187144"),{backgroundColor:"white"});
viewer_17777649540187144.zoomTo();
	viewer_17777649540187144.addModel("74\nPOSCAR\nC 4.597062 4.480307 15.035429\nO 4.631082 4.462444 16.208609\nCu 0.000000 0.000000 9.958470\nCu 1.810631 1.810631 9.958470\nCu 0.000000 1.810631 8.147839\nCu 1.810631 0.000000 8.147839\nCu 0.005499 0.007026 13.486375\nCu 1.821297 1.820377 13.505066\nCu 0.003976 1.811101 11.728095\nCu 1.809158 0.004697 11.725990\nCu 0.000000 3.621262 9.958470\nCu 1.810631 5.431893 9.958470\nCu 0.000000 5.431893 8.147839\nCu 1.810631 3.621262 8.147839\nCu 0.004066 3.633234 13.496424\nCu 1.798449 5.446475 13.480544\nCu 10.853408 5.432835 11.722823\nCu 1.824728 3.621767 11.741773\nCu 0.000000 7.242524 9.958470\nCu 1.810631 9.053155 9.958470\nCu 0.000000 9.053155 8.147839\nCu 1.810631 7.242524 8.147839\nCu 10.857944 7.237042 13.496071\nCu 1.816792 9.060145 13.496538\nCu 0.004183 9.050977 11.718469\nCu 1.806737 7.246811 11.725764\nCu 3.621262 0.000000 9.958470\nCu 5.431893 1.810631 9.958470\nCu 3.621262 1.810631 8.147839\nCu 5.431893 0.000000 8.147839\nCu 3.632501 0.005064 13.495718\nCu 5.446061 1.797005 13.479612\nCu 3.621484 1.825051 11.740150\nCu 5.433092 10.853036 11.722777\nCu 3.621262 3.621262 9.958470\nCu 5.431893 5.431893 9.958470\nCu 3.621262 5.431893 8.147839\nCu 5.431893 3.621262 8.147839\nCu 3.612827 3.612060 13.562223\nCu 5.449215 5.449794 13.550697\nCu 3.619566 5.434529 11.695558\nCu 5.435198 3.618869 11.690678\nCu 3.621262 7.242524 9.958470\nCu 5.431893 9.053155 9.958470\nCu 3.621262 9.053155 8.147839\nCu 5.431893 7.242524 8.147839\nCu 3.607599 7.255985 13.479344\nCu 5.421302 9.049742 13.495406\nCu 3.620121 9.064822 11.722818\nCu 5.432964 7.232326 11.738804\nCu 7.242524 0.000000 9.958470\nCu 9.053155 1.810631 9.958470\nCu 7.242524 1.810631 8.147839\nCu 9.053155 0.000000 8.147839\nCu 7.236525 10.859319 13.497193\nCu 9.058289 1.817798 13.497650\nCu 7.246248 1.807313 11.726570\nCu 9.051459 0.003752 11.717746\nCu 7.242524 3.621262 9.958470\nCu 9.053155 5.431893 9.958470\nCu 7.242524 5.431893 8.147839\nCu 9.053155 3.621262 8.147839\nCu 7.256695 3.608656 13.478303\nCu 9.048754 5.422025 13.494634\nCu 7.231702 5.433443 11.737228\nCu 9.064969 3.620036 11.722792\nCu 7.242524 7.242524 9.958470\nCu 9.053155 9.053155 9.958470\nCu 7.242524 9.053155 8.147839\nCu 9.053155 7.242524 8.147839\nCu 7.238653 7.237743 13.501866\nCu 9.050484 9.051628 13.485554\nCu 7.244001 9.051859 11.727229\nCu 9.051229 7.245795 11.725157\n","xyz");
	viewer_17777649540187144.setStyle({"sphere": {"scale": 0.5, "shininess": 80, "specular": "white"}, "stick": {"radius": 0.15}});
	viewer_17777649540187144.addLine({"start": {"x": 0.0, "y": 0.0, "z": 0.0}, "end": {"x": 10.863786, "y": 0.0, "z": 0.0}, "color": "black", "radius": 0.08});
	viewer_17777649540187144.addLine({"start": {"x": 0.0, "y": 0.0, "z": 0.0}, "end": {"x": 0.0, "y": 10.863786, "z": 0.0}, "color": "black", "radius": 0.08});
	viewer_17777649540187144.addLine({"start": {"x": 0.0, "y": 0.0, "z": 0.0}, "end": {"x": 0.0, "y": 0.0, "z": 21.727572}, "color": "black", "radius": 0.08});
	viewer_17777649540187144.addLine({"start": {"x": 10.863786, "y": 0.0, "z": 0.0}, "end": {"x": 10.863786, "y": 10.863786, "z": 0.0}, "color": "black", "radius": 0.08});
	viewer_17777649540187144.addLine({"start": {"x": 10.863786, "y": 0.0, "z": 0.0}, "end": {"x": 10.863786, "y": 0.0, "z": 21.727572}, "color": "black", "radius": 0.08});
	viewer_17777649540187144.addLine({"start": {"x": 0.0, "y": 10.863786, "z": 0.0}, "end": {"x": 10.863786, "y": 10.863786, "z": 0.0}, "color": "black", "radius": 0.08});
	viewer_17777649540187144.addLine({"start": {"x": 0.0, "y": 10.863786, "z": 0.0}, "end": {"x": 0.0, "y": 10.863786, "z": 21.727572}, "color": "black", "radius": 0.08});
	viewer_17777649540187144.addLine({"start": {"x": 0.0, "y": 0.0, "z": 21.727572}, "end": {"x": 10.863786, "y": 0.0, "z": 21.727572}, "color": "black", "radius": 0.08});
	viewer_17777649540187144.addLine({"start": {"x": 0.0, "y": 0.0, "z": 21.727572}, "end": {"x": 0.0, "y": 10.863786, "z": 21.727572}, "color": "black", "radius": 0.08});
	viewer_17777649540187144.addLine({"start": {"x": 10.863786, "y": 10.863786, "z": 0.0}, "end": {"x": 10.863786, "y": 10.863786, "z": 21.727572}, "color": "black", "radius": 0.08});
	viewer_17777649540187144.addLine({"start": {"x": 10.863786, "y": 0.0, "z": 21.727572}, "end": {"x": 10.863786, "y": 10.863786, "z": 21.727572}, "color": "black", "radius": 0.08});
	viewer_17777649540187144.addLine({"start": {"x": 0.0, "y": 10.863786, "z": 21.727572}, "end": {"x": 10.863786, "y": 10.863786, "z": 21.727572}, "color": "black", "radius": 0.08});
	viewer_17777649540187144.setStyle({"elem": "H"},{"sphere": {"scale": 0.3, "color": "#FFFFFF"}, "stick": {"radius": 0.15, "color": "#FFFFFF"}});
	viewer_17777649540187144.setStyle({"elem": "O"},{"sphere": {"scale": 0.3, "color": "#FF0D0D"}, "stick": {"radius": 0.15, "color": "#FF0D0D"}});
	viewer_17777649540187144.setStyle({"elem": "Si"},{"sphere": {"scale": 0.3, "color": "#B0B7C6"}, "stick": {"radius": 0.15, "color": "#B0B7C6"}});
	viewer_17777649540187144.setStyle({"elem": "C"},{"sphere": {"scale": 0.3, "color": "#808080"}, "stick": {"radius": 0.15, "color": "#808080"}});
	viewer_17777649540187144.setStyle({"elem": "Ti"},{"sphere": {"scale": 0.3, "color": "#FF77AA"}, "stick": {"radius": 0.15, "color": "#FF77AA"}});
	viewer_17777649540187144.setStyle({"elem": "Mo"},{"sphere": {"scale": 0.3, "color": "#4F7942"}, "stick": {"radius": 0.15, "color": "#4F7942"}});
	viewer_17777649540187144.setBackgroundColor("white");
	viewer_17777649540187144.setProjection("orthographic");
	viewer_17777649540187144.zoomTo();
viewer_17777649540187144.render();
});
</script>


## 2. Slab Model: Cu(100)

We use a **4-layer Cu(100) slab** with a (2×2) lateral supercell (4 Cu atoms per layer, 16 total) and ~15 Å of vacuum to decouple periodic images.

### Cu(100) POSCAR (clean slab, 4 layers, 2×2)

```
Cu(100) 2x2 4-layer slab
1.0
   7.234   0.000   0.000
   0.000   7.234   0.000
   0.000   0.000  20.000
Cu
16
Selective dynamics
Direct
  0.000  0.000  0.000  F F F
  0.500  0.000  0.000  F F F
  0.000  0.500  0.000  F F F
  0.500  0.500  0.000  F F F
  0.250  0.250  0.091  F F F
  0.750  0.250  0.091  F F F
  0.250  0.750  0.091  F F F
  0.750  0.750  0.091  F F F
  0.000  0.000  0.182  T T T
  0.500  0.000  0.182  T T T
  0.000  0.500  0.182  T T T
  0.500  0.500  0.182  T T T
  0.250  0.250  0.273  T T T
  0.750  0.250  0.273  T T T
  0.250  0.750  0.273  T T T
  0.750  0.750  0.273  T T T
```

**Selective dynamics**: bottom two layers are frozen (`F F F`); top two layers are relaxed (`T T T`) to mimic a semi-infinite bulk while keeping computational cost manageable.

The lattice constant for Cu is 3.617 Å, so the 2×2 cell is 2 × 3.617 Å ≈ 7.234 Å.

### Adsorption Sites on Cu(100)

```
    ·─────·─────·
    │  H  │  H  │   ← hollow sites (center of 4 Cu)
    ·──B──·──B──·   ← bridge sites (midpoint between 2 Cu)
    │  H  │  H  │
    ·─────·─────·
    T = top site (directly above a Cu atom)
```

- **Top site**: CO prefers this on Cu(100)
- **Hollow site**: H prefers this on Cu(100)

## 3. VASP Input Files

### 3.1 INCAR generic

```
# INCAR — Cu(100) clean slab geometry optimization
SYSTEM  = Cu100 clean slab
ISTART  = 0
ICHARG  = 2
ENCUT   = 400        # Change to 1.2x Enmax for good quality, 1.5x for cell relaxation
PREC    = Accurate
EDIFF   = 1E-5       # Change to 1e-6 or 1e-7 for more precise electronic structure calculations
EDIFFG  = -0.02      
NSW     = 100        # Increase if does not converge
IBRION  = 2          # conjugate gradient
ISIF    = 2          # relax atoms only, fix cell shape, use 3 for complete cell relaxation
POTIM   = 0.3        # Lower for trickier geoms, 
ISMEAR  = 1          # Methfessel-Paxton for metals, use 0 for semiconductors, surfaces
SIGMA   = 0.2        # smearing width (eV)
ALGO    = Fast       # Change to normal if open shell caculcations (with ISPIN = 2 ) see a fluctuation during convergence
NELM    = 100
LWAVE   = .TRUE.     # Save only when useful
LCHARG  = .TRUE.     # Save only when useful
LAECHG  = .TRUE.    ## Use when doing Bader charges

ISPIN   = 2 # Important for open shell, metals
LORBIT  = 11 # orbital resolved spin
MAGMOM  = 72*0.1 # NumberofatomsA*value in BM.
```

### 3.2 KPOINTS

```
Automatic
0
Gamma
  4  4  1
  0  0  0
```
A 4×4×1 Gamma-centered mesh is standard for a (2×2) surface cell. For cell vectors ax, ay, az ∈ **a**, Use kpoints = 20/**a** for initial optimization, 25/**a** for surface energies, 30/**a** for high quality electronic structure. Nn c-direction, use 1 for vacuum cases.

### 3.3 INCAR — Slab + CO (top site)

Append a CO molecule at the top site of the Cu(100) surface. Place C at ~1.9 Å above a surface Cu atom, O at ~3.1 Å (C–O bond ≈ 1.16 Å, CO stands upright).


### 3.4 INCAR — Slab + H (hollow site)

Place H at the hollow site (~1.0 Å above the surface plane).


### 3.5 Gas-phase reference energies

Calculate isolated CO (spin-paired) and H₂ molecule in a large box.

For H, we use ½ E(H₂), where H₂ is calculated in the same large box with ISMEAR=0.

## 4. Adsorption Energy Calculation

After all DFT runs finish, extract the final total energies from OSZICAR or OUTCAR and compute:

$$E_{\rm ads}(\rm CO) = E_{\rm Cu100+CO} - E_{\rm Cu100} - E_{\rm CO(gas)}$$

$$E_{\rm ads}(\rm H) = E_{\rm Cu100+H} - E_{\rm Cu100} - \tfrac{1}{2}E_{\rm H_2(gas)}$$

The code below reads the energies from OUTCAR files (stored in `../tmp/Cu100/`) and computes the adsorption energies.

## 5. Free Energy Correction via Frequency Calculation

DFT total energies represent the **electronic energy at 0 K**. To model catalytic processes at finite temperature and pressure we need the **Gibbs free energy**:

$$\Delta G_{\rm ads}(T) = \Delta E_{\rm ads} + \Delta G_{\rm vib}(T)$$

where the vibrational free energy correction is:

$$\Delta G_{\rm vib}(T) = \Delta {\rm ZPE} + \Delta H_{\rm vib}(T) - T\,\Delta S_{\rm vib}(T)$$

### 5.1 VASP Frequency Calculation Setup

After geometry optimization, run a **finite-difference Hessian** calculation using `IBRION = 5`:

```
# INCAR — frequency calculation for CO/Cu(100)
SYSTEM  = CO on Cu100 frequency
ISTART  = 1         # read WAVECAR from previous geometry optimization
ICHARG  = 0
ENCUT   = 400
PREC    = Accurate
EDIFF   = 1E-7      # tighter convergence needed for accurate forces
NSW     = 1
IBRION  = 5         # finite-difference Hessian (use IBRION=6 with symmetry)
NFREE   = 2         # forward + backward displacement per atom per direction
POTIM   = 0.015     # displacement step in Å (0.01–0.02 Å is typical)
ISMEAR  = 1
SIGMA   = 0.2
LWAVE   = .FALSE.
LCHARG  = .FALSE.
```

> **Key point**: In the POSCAR, keep the Cu substrate atoms **frozen** (`F F F` selective dynamics) and free only the adsorbate atoms (`T T T`). VASP then displaces only the free atoms, reducing the number of single-point force evaluations from $6N$ (full slab) to $6 N_{\rm ads}$ (adsorbate only).

After the run, vibrational frequencies appear in `OUTCAR`. Extract them with:

```bash
grep 'THz' OUTCAR
```

Typical output for CO adsorbed at the top site on Cu(100):

```
 f  =   58.98 THz  370.36 2PiTHz  1967.1 cm-1   243.8 meV
 f  =   10.55 THz   66.27 2PiTHz   351.8 cm-1    43.6 meV
 f  =   10.52 THz   66.09 2PiTHz   350.7 cm-1    43.5 meV
 f  =    1.58 THz    9.91 2PiTHz    52.5 cm-1     6.5 meV
 f  =    1.53 THz    9.63 2PiTHz    51.1 cm-1     6.3 meV
 f  =    1.46 THz    9.19 2PiTHz    48.6 cm-1     6.0 meV
```

The highest-frequency mode (~1967 cm⁻¹) is the **C–O stretch**; lower modes are frustrated translations and rotations (hindered motion of the adsorbate on the surface).

> **Imaginary frequencies** (`f/i` in OUTCAR) indicate a saddle point. A small imaginary frequency (< 50 cm⁻¹) is usually numerical noise; larger values mean the geometry has not fully relaxed.

### 5.2 Harmonic Oscillator Thermodynamic Corrections

For surface adsorbates, translational and rotational degrees of freedom are replaced by the frustrated modes captured in the frequency calculation. Under the **harmonic oscillator (HO) approximation**:

$$\text{ZPE} = \sum_i \frac{h\nu_i}{2}$$

$$H_{\rm vib}(T) = \sum_i \frac{h\nu_i}{e^{h\nu_i/k_BT} - 1}$$

$$S_{\rm vib}(T) = k_B \sum_i \left[\frac{h\nu_i/k_BT}{e^{h\nu_i/k_BT} - 1} - \ln\!\left(1 - e^{-h\nu_i/k_BT}\right)\right]$$

$$G_{\rm vib}(T) = \text{ZPE} + H_{\rm vib}(T) - T\,S_{\rm vib}(T)$$

The code cell below shows how to parse frequencies from OUTCAR and evaluate these corrections.


```python
import numpy as np

# Physical constants (SI)
h  = 6.62607015e-34   # Planck constant  (J·s)
kB = 1.380649e-23     # Boltzmann constant (J/K)
c  = 2.99792458e10    # Speed of light     (cm/s)
eV = 1.602176634e-19  # 1 eV in joules


def parse_frequencies(outcar_path):
    """
    Parse real vibrational frequencies (cm^-1) from a VASP OUTCAR.
    Imaginary frequencies (lines containing 'f/i') are skipped.
    """
    freqs = []
    with open(outcar_path, "r") as fh:
        for line in fh:
            if "THz" in line and "cm-1" in line:
                if "f/i" in line:
                    continue  # skip imaginary modes
                parts = line.split()
                # VASP format: f  =  X THz  Y 2PiTHz  Z cm-1  W meV
                idx = parts.index("cm-1")
                freqs.append(float(parts[idx - 1]))
    return freqs


def compute_zpe(freqs_cm1):
    """Zero-point energy in eV: ZPE = sum_i (h*nu_i / 2)."""
    return sum(0.5 * h * (nu * c) for nu in freqs_cm1) / eV


def compute_vib_enthalpy(freqs_cm1, T):
    """Thermal vibrational enthalpy correction (eV) beyond ZPE."""
    H = 0.0
    for nu in freqs_cm1:
        x = h * nu * c / (kB * T)
        H += kB * T * x / (np.exp(x) - 1)
    return H / eV


def compute_vib_entropy(freqs_cm1, T):
    """Vibrational entropy contribution T*S_vib in eV."""
    S = 0.0
    for nu in freqs_cm1:
        x = h * nu * c / (kB * T)
        S += kB * (x / (np.exp(x) - 1) - np.log(1 - np.exp(-x)))
    return S * T / eV  # T*S in eV


def free_energy_correction(freqs_cm1, T=298.15):
    """
    Compute ZPE, H_vib, T*S_vib, and G_vib = ZPE + H_vib - T*S_vib (all in eV).
    """
    zpe   = compute_zpe(freqs_cm1)
    Hvib  = compute_vib_enthalpy(freqs_cm1, T)
    TSvib = compute_vib_entropy(freqs_cm1, T)
    return zpe, Hvib, TSvib, zpe + Hvib - TSvib


# ---------------------------------------------------------------------------
# Example: CO adsorbed on Cu(100) top site
# In practice, replace the lists below with:
#   freqs_CO_ads = parse_frequencies("path/to/CO_ads/OUTCAR")
#   freqs_CO_gas = parse_frequencies("path/to/CO_gas/OUTCAR")
# ---------------------------------------------------------------------------

# 6 modes for CO@top-site: C-O stretch + 2 frustrated rot. + 3 frustrated trans.
freqs_CO_ads = [1967.1, 351.8, 350.7, 52.5, 51.1, 48.6]  # cm^-1

# 1 vibrational mode for gas-phase CO
# (translational/rotational contributions excluded here;
#  for a rigorous ΔG include ideal-gas partition function for the molecule)
freqs_CO_gas = [2143.0]  # cm^-1  (experimental C-O stretch)

T = 298.15  # K (25 °C)

zpe_ads, Hvib_ads, TSvib_ads, Gvib_ads = free_energy_correction(freqs_CO_ads, T)
zpe_gas, Hvib_gas, TSvib_gas, Gvib_gas = free_energy_correction(freqs_CO_gas, T)

header = f"{'':28s} {'ZPE':>8s} {'H_vib':>10s} {'TS_vib':>10s} {'G_vib':>10s}  (eV)"
sep    = "-" * len(header)
print(f"Thermodynamic corrections at T = {T:.1f} K\n")
print(header)
print(sep)
print(f"{'CO (ads, top site)':28s} {zpe_ads:8.4f} {Hvib_ads:10.4f} {TSvib_ads:10.4f} {Gvib_ads:10.4f}")
print(f"{'CO (gas, vib only)':28s} {zpe_gas:8.4f} {Hvib_gas:10.4f} {TSvib_gas:10.4f} {Gvib_gas:10.4f}")
print(sep)

delta_Gvib = Gvib_ads - Gvib_gas
print(f"\nΔG_vib  (ads − gas)  = {delta_Gvib:+.4f} eV")

# Combine with DFT electronic adsorption energy
E_ads_CO   = -0.65   # eV  (example PBE value for CO on Cu(100))
deltaG_ads = E_ads_CO + delta_Gvib

print(f"\nE_ads  (DFT, 0 K)     = {E_ads_CO:+.2f} eV")
print(f"ΔG_ads (T = {T:.0f} K) = {deltaG_ads:+.4f} eV")
print(f"Free energy correction: {delta_Gvib:+.4f} eV  ({100*abs(delta_Gvib/E_ads_CO):.1f}% of |E_ads|)")
```

## 6. Discussion

### Why does CO bind at the top site?

CO binds at the **top (atop) site** on most close-packed metal surfaces including Cu(100). The C lone pair donates into the Cu 3d/4s states (Blyholder model: 5σ donation and 2π* back-donation). The overlap is maximized at the top site where a single Cu atom interacts directly with C.

> **PBE artifact**: PBE predicts CO preferring the hollow site on some metals (e.g., Pt(111)), which is the wrong answer. This 'CO adsorption site puzzle' was a famous DFT failure resolved by using the RPBE functional or hybrid functionals. On Cu(100), PBE gives the correct top-site preference.

### Why does H bind at the hollow site?

Hydrogen adsorbs as an atom (H₂ dissociates at the surface). The H 1s orbital interacts with multiple Cu atoms simultaneously, maximizing the bonding overlap in the **hollow** site. The adsorption energy is much larger in magnitude than CO because H forms a true chemisorptive bond.

### Practical implications

These adsorption energies are inputs to microkinetic models for catalytic reactions such as:
- CO₂ reduction → CO + O* → further reduction
- Water-gas shift reaction: CO + H₂O → CO₂ + H₂
- Fischer-Tropsch synthesis

## Further Reading

- Hammer, B.; Hansen, L. B.; Nørskov, J. K. *Phys. Rev. B* **59**, 7413 (1999) — RPBE and CO adsorption site puzzle
- Blyholder, G. *J. Phys. Chem.* **68**, 2772 (1964) — Blyholder model for CO–metal bonding
- Nørskov, J. K. et al. *J. Catal.* **209**, 275 (2002) — universality of adsorption energies
- [VASP Wiki: Surface calculations](https://www.vasp.at/wiki/Surface_calculations)
- [ASE surface tutorial](https://wiki.fysik.dtu.dk/ase/tutorials/surface.html)

---
*Ch121a | Caltech | Module 3 — Notebook 4b of 6*
