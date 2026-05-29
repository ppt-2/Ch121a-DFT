# 12 -- ORCA: Capabilities & Practical Guide

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ppt-2/Ch121a-DFT/blob/main/notebooks/12_orca_tutorial.ipynb)

## Learning Objectives
- Write valid ORCA input files for single-point, optimization, and frequency jobs
- Understand the `!` keyword line and `%` block syntax
- Use the RIJCOSX approximation to accelerate hybrid-DFT calculations
- Set up DLPNO-CCSD(T) for benchmark-quality energetics
- Parse ORCA output files with Python and cclib
- Choose appropriate methods, basis sets, and auxiliary basis sets
- Recognize common ORCA error messages and their solutions

## 1. ORCA Overview

**ORCA** (developed by Frank Neese, MPI Mulheim) is a free-for-academic production
quantum chemistry code. Download from https://orcaforum.kofo.mpg.de

| Area | Methods |
|------|--------|
| DFT | All major XC functionals; hybrid, double-hybrid |
| Post-HF | MP2, CCSD(T), DLPNO-CCSD(T) |
| Multireference | CASSCF, NEVPT2, MRCI |
| Spectroscopy | EPR, Mossbauer, X-ray, NMR |
| Relativistic | ZORA, DKH, CPCM solvation |

## 2. Input File Syntax

An ORCA input file (`job.inp`) consists of:
1. **`!` keyword line** -- method, basis set, and job-type keywords
2. **`%` blocks** -- detailed options (optional)
3. **`* xyz` coordinate block** -- molecular geometry

### 2.1 Minimal Single-Point

```
! B3LYP def2-SVP TightSCF

* xyz 0 1
  O   0.000   0.000   0.117
  H   0.000   0.757  -0.469
  H   0.000  -0.757  -0.469
*
```

### 2.2 Geometry Optimization + Frequencies

```
! B3LYP def2-TZVP TightSCF Opt Freq

%maxcore 4000
%pal nprocs 8 end

* xyz 0 1
  O   0.000   0.000   0.117
  H   0.000   0.757  -0.469
  H   0.000  -0.757  -0.469
*
```

### 2.3 RIJCOSX for Fast Hybrid DFT

The **RIJCOSX** approximation (RI-J for Coulomb + COSX for exchange) reduces
the cost of hybrid functionals from O(N^4) to ~O(N^3):

```
! B3LYP def2-TZVP def2/J RIJCOSX TightSCF Opt
```

Always pair with a matching auxiliary basis: `def2/J` for Coulomb fitting;
`def2-TZVP/C` for correlation (MP2/CC).

## 3. DLPNO-CCSD(T): Gold-Standard Energetics

**DLPNO-CCSD(T)** (Domain-based Local Pair Natural Orbital) achieves near-canonical
CCSD(T) accuracy at a fraction of the cost (O(N^3) vs O(N^7)):

```
! DLPNO-CCSD(T) def2-TZVP def2-TZVP/C TightSCF TightPNO

%maxcore 8000
%pal nprocs 16 end

* xyz 0 1
  ...
*
```

**Typical accuracy**: Delta-H errors < 1 kcal/mol vs canonical CCSD(T).

**TightPNO thresholds** (recommended for benchmarks):
- `TCutPNO = 1e-7` (default Normal = 3.33e-7)
- `TCutPairs = 1e-5`
- `TCutMKN = 1e-3`


```python
# =============================================================================
# Ch121a: Quantum Chemistry & DFT -- Notebook 12: ORCA Tutorial
# License: GPL-3.0 (https://www.gnu.org/licenses/gpl-3.0.en.html)
# =============================================================================

# ------------------------------------------------------------------
# Writing ORCA Input Files with Python
# ------------------------------------------------------------------

def write_orca_input(filename, method, basis, charge, mult, atoms,
                     extra_keywords='', extra_blocks=''):
    '''Write a minimal ORCA .inp file.'''
    coord_block = '\n'.join(
        f'  {sym:2s}  {x:10.5f}  {y:10.5f}  {z:10.5f}'
        for sym, x, y, z in atoms
    )
    content = f'! {method} {basis} TightSCF {extra_keywords}\n'
    if extra_blocks:
        content += extra_blocks + '\n'
    content += f'\n* xyz {charge} {mult}\n{coord_block}\n*\n'
    with open(filename, 'w') as f:
        f.write(content)
    return content

water_atoms = [
    ('O',  0.000,  0.000,  0.117),
    ('H',  0.000,  0.757, -0.469),
    ('H',  0.000, -0.757, -0.469),
]

inp_sp = write_orca_input(
    '/tmp/water_sp.inp',
    method='B3LYP', basis='def2-TZVP',
    charge=0, mult=1, atoms=water_atoms,
    extra_keywords='def2/J RIJCOSX'
)
print('=== Water Single-Point (B3LYP/def2-TZVP/RIJCOSX) ===')
print(inp_sp)

# Example: Fe(CO)5 optimization
feco5_atoms = [
    ('Fe',  0.000,  0.000,  0.000),
    ('C',   0.000,  0.000,  1.807), ('C',  0.000,  0.000, -1.807),
    ('C',   1.807,  0.000,  0.000), ('C', -1.807,  0.000,  0.000),
    ('C',   0.000,  1.807,  0.000),
    ('O',   0.000,  0.000,  2.975), ('O',  0.000,  0.000, -2.975),
    ('O',   2.975,  0.000,  0.000), ('O', -2.975,  0.000,  0.000),
    ('O',   0.000,  2.975,  0.000),
]
inp_opt = write_orca_input(
    '/tmp/feco5_opt.inp',
    method='BP86', basis='def2-TZVP',
    charge=0, mult=1, atoms=feco5_atoms,
    extra_keywords='def2/J RI Opt',
    extra_blocks='%maxcore 4000\n%pal nprocs 4 end'
)
print('=== Fe(CO)5 Optimization (BP86/def2-TZVP/RI) ===')
print(inp_opt)
```

    === Water Single-Point (B3LYP/def2-TZVP/RIJCOSX) ===
    ! B3LYP def2-TZVP TightSCF def2/J RIJCOSX
    
    * xyz 0 1
      O      0.00000     0.00000     0.11700
      H      0.00000     0.75700    -0.46900
      H      0.00000    -0.75700    -0.46900
    *
    
    === Fe(CO)5 Optimization (BP86/def2-TZVP/RI) ===
    ! BP86 def2-TZVP TightSCF def2/J RI Opt
    %maxcore 4000
    %pal nprocs 4 end
    
    * xyz 0 1
      Fe     0.00000     0.00000     0.00000
      C      0.00000     0.00000     1.80700
      C      0.00000     0.00000    -1.80700
      C      1.80700     0.00000     0.00000
      C     -1.80700     0.00000     0.00000
      C      0.00000     1.80700     0.00000
      O      0.00000     0.00000     2.97500
      O      0.00000     0.00000    -2.97500
      O      2.97500     0.00000     0.00000
      O     -2.97500     0.00000     0.00000
      O      0.00000     2.97500     0.00000
    *
    



```python
# =============================================================================
# Minimal Molecular Viewer (py3Dmol)
# =============================================================================

import py3Dmol

def view_structure(filename, style='ballstick', width=500, height=400):
    """
    Minimal, clean py3Dmol viewer for XYZ or PDB files.
    Automatically detects format from file extension.
    """
    ext = filename.split('.')[-1].lower()
    if ext not in ('xyz', 'pdb'):
        raise ValueError("File must be .xyz or .pdb")

    with open(filename, 'r') as f:
        mol_data = f.read()

    view = py3Dmol.view(width=width, height=height)

    # Add model
    view.addModel(mol_data, ext)

    # Style options
    if style == 'ballstick':
        view.setStyle({'stick': {'radius': 0.15},
                       'sphere': {'scale': 0.25}})
    elif style == 'stick':
        view.setStyle({'stick': {'radius': 0.15}})
    elif style == 'sphere':
        view.setStyle({'sphere': {'scale': 0.3}})
    else:
        view.setStyle({})  # raw atoms

    view.setBackgroundColor('white')
    view.zoomTo()
    return view.show()

# Example usage:
view_structure('../data/molecules/fe_co5.xyz')
# view_structure('/tmp/feco5_opt.xyz', style='stick')
```


<div id="3dmolviewer_17755051832835138"  style="position: relative; width: 500px; height: 400px;">
        <p id="3dmolwarning_17755051832835138" style="background-color:#ffcccc;color:black">3Dmol.js failed to load for some reason.  Please check your browser console for error messages.<br></p>
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

var viewer_17755051832835138 = null;
var warn = document.getElementById("3dmolwarning_17755051832835138");
if(warn) {
    warn.parentNode.removeChild(warn);
}
$3Dmolpromise.then(function() {
viewer_17755051832835138 = $3Dmol.createViewer(document.getElementById("3dmolviewer_17755051832835138"),{backgroundColor:"white"});
viewer_17755051832835138.zoomTo();
	viewer_17755051832835138.addModel("11\nIron pentacarbonyl Fe(CO)5 - D3h trigonal bipyramidal\nFe   0.000000   0.000000   0.000000\nC    0.000000   0.000000   1.827000\nC    0.000000   0.000000  -1.827000\nC    1.830000   0.000000   0.000000\nC   -0.915000   1.585000   0.000000\nC   -0.915000  -1.585000   0.000000\nO    0.000000   0.000000   3.006000\nO    0.000000   0.000000  -3.006000\nO    3.006000   0.000000   0.000000\nO   -1.503000   2.603000   0.000000\nO   -1.503000  -2.603000   0.000000\n","xyz");
	viewer_17755051832835138.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}});
	viewer_17755051832835138.setBackgroundColor("white");
	viewer_17755051832835138.zoomTo();
viewer_17755051832835138.render();
});
</script>



```python
# =============================================================================
# Minimal Molecular Viewer (py3Dmol)
# =============================================================================

import py3Dmol

def view_structure(filename, style='ballstick', width=500, height=400):
    """
    Minimal, clean py3Dmol viewer for XYZ or PDB files.
    Automatically detects format from file extension.
    """
    ext = filename.split('.')[-1].lower()
    if ext not in ('xyz', 'pdb'):
        raise ValueError("File must be .xyz or .pdb")

    with open(filename, 'r') as f:
        mol_data = f.read()

    view = py3Dmol.view(width=width, height=height)

    # Add model
    view.addModel(mol_data, ext)

    # Style options
    if style == 'ballstick':
        view.setStyle({'stick': {'radius': 0.15},
                       'sphere': {'scale': 0.25}})
    elif style == 'stick':
        view.setStyle({'stick': {'radius': 0.15}})
    elif style == 'sphere':
        view.setStyle({'sphere': {'scale': 0.3}})
    else:
        view.setStyle({})  # raw atoms

    view.setBackgroundColor('white')
    view.zoomTo()
    return view.show()

# Example usage:
view_structure('../data/molecules/ferrocene.xyz')
# view_structure('/tmp/feco5_opt.xyz', style='stick')
```


<div id="3dmolviewer_1775505232487022"  style="position: relative; width: 500px; height: 400px;">
        <p id="3dmolwarning_1775505232487022" style="background-color:#ffcccc;color:black">3Dmol.js failed to load for some reason.  Please check your browser console for error messages.<br></p>
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

var viewer_1775505232487022 = null;
var warn = document.getElementById("3dmolwarning_1775505232487022");
if(warn) {
    warn.parentNode.removeChild(warn);
}
$3Dmolpromise.then(function() {
viewer_1775505232487022 = $3Dmol.createViewer(document.getElementById("3dmolviewer_1775505232487022"),{backgroundColor:"white"});
viewer_1775505232487022.zoomTo();
	viewer_1775505232487022.addModel("21\nFerrocene Fe(C5H5)2 - eclipsed D5h conformation\nFe   0.000000   0.000000   0.000000\nC    1.220000   0.887000   1.660000\nC   -0.466000   1.434000   1.660000\nC   -1.508000   0.000000   1.660000\nC   -0.466000  -1.434000   1.660000\nC    1.220000  -0.887000   1.660000\nC    1.220000   0.887000  -1.660000\nC   -0.466000   1.434000  -1.660000\nC   -1.508000   0.000000  -1.660000\nC   -0.466000  -1.434000  -1.660000\nC    1.220000  -0.887000  -1.660000\nH    2.168000   1.574000   1.660000\nH   -0.827000   2.545000   1.660000\nH   -2.678000   0.000000   1.660000\nH   -0.827000  -2.545000   1.660000\nH    2.168000  -1.574000   1.660000\nH    2.168000   1.574000  -1.660000\nH   -0.827000   2.545000  -1.660000\nH   -2.678000   0.000000  -1.660000\nH   -0.827000  -2.545000  -1.660000\nH    2.168000  -1.574000  -1.660000\n","xyz");
	viewer_1775505232487022.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}});
	viewer_1775505232487022.setBackgroundColor("white");
	viewer_1775505232487022.zoomTo();
viewer_1775505232487022.render();
});
</script>



```python
# ------------------------------------------------------------------
# Parsing ORCA Output Files
# ------------------------------------------------------------------
import re

mock_orca_out = '''
ORCA SCF CONVERGED AFTER  11 CYCLES
Total Energy       :         -76.437879956 Eh
TOTAL ENERGY          -76.437879956 Hartree
TOTAL ENTHALPY        -76.435432876 Hartree

Mulliken charges:
   0 O   :  -0.708
   1 H   :   0.354
   2 H   :   0.354

VIBRATIONAL FREQUENCIES
   0:       0.00 cm**-1
   3:    1641.82 cm**-1
   4:    3779.56 cm**-1
   5:    3892.41 cm**-1
'''

e_match = re.search(r'TOTAL ENERGY\s+([\-0-9.]+)\s+Hartree', mock_orca_out)
charge_matches = re.findall(r'\s+\d+\s+(\w+)\s+:\s+([\-0-9.]+)', mock_orca_out)
freq_matches = re.findall(r'\s+\d+:\s+([0-9.]+) cm\*\*-1', mock_orca_out)

print('Parsed from mock ORCA .out:')
if e_match:
    print(f'  Total energy = {e_match.group(1)} Hartree')
print('  Mulliken charges:', {sym: float(q) for sym, q in charge_matches})
non_zero = [f for f in freq_matches if float(f) > 10]
print('  Vibrational frequencies (cm-1):', non_zero)

print()
print('--- cclib usage (with a real ORCA .out file) ---')
print('import cclib')
print('data = cclib.io.ccread("water.out")')
print('print(data.scfenergies[-1], "eV")')
print('print(data.atomcharges["mulliken"])')
print('print(data.vibfreqs, "cm-1")')
```

    Parsed from mock ORCA .out:
      Total energy = -76.437879956 Hartree
      Mulliken charges: {'O': -0.708, 'H': 0.354}
      Vibrational frequencies (cm-1): ['1641.82', '3779.56', '3892.41']
    
    --- cclib usage (with a real ORCA .out file) ---
    import cclib
    data = cclib.io.ccread("water.out")
    print(data.scfenergies[-1], "eV")
    print(data.atomcharges["mulliken"])
    print(data.vibfreqs, "cm-1")


## 4. Common ORCA Keywords Reference

| Keyword | Purpose |
|---------|--------|
| `TightSCF` / `VeryTightSCF` | SCF convergence criteria |
| `Opt` | Geometry optimization |
| `Freq` | Analytical or numerical Hessian |
| `NMR` | NMR shielding tensors |
| `RIJCOSX` | RI-J Coulomb + seminumerical exchange (hybrids) |
| `RI` / `RIJONX` | RI approximation for pure DFT |
| `CPCM(Water)` | Conductor-like PCM solvation |
| `def2/J` | Auxiliary Coulomb-fitting basis |
| `def2-TZVP/C` | Auxiliary correlation basis for MP2/CC |
| `TightPNO` | Tight thresholds for DLPNO methods |
| `ZORA` / `DKH` | Scalar relativistic corrections |
| `D3BJ` | Grimme D3 dispersion with Becke-Johnson damping |
| `Grid5` / `Grid7` | Integration grid accuracy |

## Research Connection

ORCA is used across computational chemistry research:

- **Bioinorganic chemistry**: DLPNO-CCSD(T) energetics for iron-sulfur clusters and
  metalloenzyme active sites too large for canonical CC.
- **Catalysis**: BP86/TZVP geometry optimizations followed by DLPNO-CCSD(T) single-points
  is a standard protocol for reaction energy benchmarks.
- **Spectroscopy**: ORCA's EFG, EPR, and Mossbauer modules interpret transition metal spectra.
- **ML potentials**: ORCA generates training data for neural network potentials
  (e.g., wB97X-D3/def2-TZVP for ANI-2x).

## Summary

| Topic | Key Point |
|-------|----------|
| Input syntax | `! method basis [keywords]` + `* xyz charge mult ... *` |
| RIJCOSX | RI-J Coulomb + COSX exchange; ~10x faster hybrids |
| DLPNO-CCSD(T) | O(N^3) near-canonical CCSD(T); < 1 kcal/mol error |
| Auxiliary basis | `def2/J` for RI-J; `def2-TZVP/C` for correlation |
| Output parsing | cclib: `cclib.io.ccread('file.out')` |
| Parallelism | `%pal nprocs N end`; `%maxcore MB_per_core` |
| Grid | `Grid5` default; `Grid7` for final energies |

## Exercises

1. **Write an input**: Write an ORCA input for geometry optimization of ethanol
   (CH3CH2OH) using B3LYP-D3BJ/def2-TZVP with RIJCOSX. Include `%pal` and
   `%maxcore` blocks for 8 cores and 4 GB/core.

2. **DLPNO setup**: Write a DLPNO-CCSD(T)/def2-TZVP single-point input for
   the optimized ethanol geometry. Use TightPNO settings.

3. **cclib parsing**: Given a real ORCA frequency output, write Python code using
   cclib to (a) extract all vibrational frequencies, (b) compute the ZPE, and
   (c) identify imaginary modes.

4. **Basis set convergence**: Design an ORCA input series using def2-SVP, def2-TZVP,
   and def2-QZVP for HF single-points on the HF molecule.

5. **RIJCOSX accuracy**: Explain the RIJCOSX approximation. What property calculations
   might be sensitive to the COSX approximation for the exchange part?

---

# 13 -- Jaguar: Capabilities & Practical Guide

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ppt-2/Ch121a-DFT/blob/main/notebooks/13_jaguar_tutorial.ipynb)

## Learning Objectives

By the end of this notebook, you will be able to:
- Understand the Jaguar `.in` input file format (sections and keywords)
- Set up DFT, LMP2, and GVB-PP calculations in Jaguar
- Use the Maestro GUI workflow for job submission and visualization
- Parse Jaguar `.out` files with Python and cclib
- Understand Jaguar's strengths: LMP2, GVB, transition metal DFT
- Know when to choose Jaguar vs ORCA vs PySCF

## 1. Jaguar Overview

**Jaguar** (Schrodinger, Inc.) is a commercial quantum chemistry code optimized for:

| Capability | Details |
|------------|--------|
| DFT | All major XC functionals; pseudospectral method for speed |
| LMP2 | Local MP2 -- scales O(N^3) vs O(N^5) for canonical MP2 |
| GVB-PP | Generalized Valence Bond (perfect pairing) for static correlation |
| Transition metals | LACVP**, ECP basis sets built-in |
| Solvation | PBF (Poisson-Boltzmann Finite element) solvent model |
| Integration | Schrodinger Suite: Maestro, Glide, Prime, MacroModel |

**License**: Commercial. Academic licenses available through Schrodinger.

## 2. Input File Format

A Jaguar input file (`job.in`) uses named sections delimited by `&section ... &`:

### 2.1 Minimal Single-Point

```
&gen
 basis=6-31g**
 dftname=b3lyp
 igeopt=0
 molchg=0
 multip=1
&

&zmat
 O  0.000  0.000  0.117
 H  0.000  0.757 -0.469
 H  0.000 -0.757 -0.469
&
```

### 2.2 Geometry Optimization

```
&gen
 basis=def2-tzvp
 dftname=b3lyp
 igeopt=1
 maxitg=100
 gconv=5.0e-5
&
```

### 2.3 LMP2 Single-Point

```
&gen
 basis=cc-pvtz
 mp2=yes
 lmp2=yes
 igeopt=0
&
```

### 2.4 Transition Metal (LACVP** ECP)

```
&gen
 basis=lacvp**
 dftname=b3lyp
 igeopt=1
 molchg=0
 multip=1
&

&zmat
 Fe  0.000  0.000  0.000
 C   0.000  0.000  1.807
 O   0.000  0.000  2.975
 ...
&
```


```python
# =============================================================================
# Ch121a: Quantum Chemistry & DFT -- Notebook 13: Jaguar Tutorial
# License: GPL-3.0 (https://www.gnu.org/licenses/gpl-3.0.en.html)
# =============================================================================

# ------------------------------------------------------------------
# Writing Jaguar Input Files with Python
# ------------------------------------------------------------------

def write_jaguar_input(filename, atoms, basis='6-31g**', dftname='b3lyp',
                       igeopt=0, charge=0, mult=1, extra_gen='', extra_sections=''):
    '''Write a Jaguar .in file.'''
    gen = f'basis={basis}\ndftname={dftname}\nigeopt={igeopt}\nmolchg={charge}\nmultip={mult}'
    if extra_gen:
        gen += '\n' + extra_gen
    zmat = '\n'.join(
        f' {sym:2s}  {x:10.5f}  {y:10.5f}  {z:10.5f}'
        for sym, x, y, z in atoms
    )
    content = f'&gen\n{gen}\n&\n\n&zmat\n{zmat}\n&\n'
    if extra_sections:
        content += '\n' + extra_sections
    with open(filename, 'w') as f:
        f.write(content)
    return content

water = [('O',0.000,0.000,0.117),('H',0.000,0.757,-0.469),('H',0.000,-0.757,-0.469)]

print('=== B3LYP/def2-TZVP single-point ===')
print(write_jaguar_input('/tmp/water_sp.in', water, basis='def2-tzvp', dftname='b3lyp'))

print('=== LMP2/cc-pVTZ single-point ===')
print(write_jaguar_input('/tmp/water_lmp2.in', water,
                         basis='cc-pvtz', dftname='none',
                         extra_gen='mp2=yes\nlmp2=yes'))

print('=== B3LYP/6-31G** + PBF solvation (water) ===')
print(write_jaguar_input('/tmp/water_pbf.in', water,
                         extra_gen='isolv=2\nsolvent=water'))
```

    === B3LYP/def2-TZVP single-point ===
    &gen
    basis=def2-tzvp
    dftname=b3lyp
    igeopt=0
    molchg=0
    multip=1
    &
    
    &zmat
     O      0.00000     0.00000     0.11700
     H      0.00000     0.75700    -0.46900
     H      0.00000    -0.75700    -0.46900
    &
    
    === LMP2/cc-pVTZ single-point ===
    &gen
    basis=cc-pvtz
    dftname=none
    igeopt=0
    molchg=0
    multip=1
    mp2=yes
    lmp2=yes
    &
    
    &zmat
     O      0.00000     0.00000     0.11700
     H      0.00000     0.75700    -0.46900
     H      0.00000    -0.75700    -0.46900
    &
    
    === B3LYP/6-31G** + PBF solvation (water) ===
    &gen
    basis=6-31g**
    dftname=b3lyp
    igeopt=0
    molchg=0
    multip=1
    isolv=2
    solvent=water
    &
    
    &zmat
     O      0.00000     0.00000     0.11700
     H      0.00000     0.75700    -0.46900
     H      0.00000    -0.75700    -0.46900
    &
    


## 3. GVB-PP: Generalized Valence Bond

**GVB (Generalized Valence Bond)** describes electron pairs as geminal products
of two natural orbitals, capturing static (near-degeneracy) correlation that DFT misses.

### GVB-PP Wavefunction

$$|\Psi_{\rm GVB}\rangle = \prod_{i=1}^{N/2}
(c_{i1}\phi_{i1} + c_{i2}\phi_{i2})_\alpha
(c_{i1}\phi_{i1} + c_{i2}\phi_{i2})_\beta$$

**When to use GVB**:
- Bond-breaking reactions (where DFT and RHF break down)
- Diradical / biradical character
- Homolytic bond dissociation curves
- Starting orbitals for CASSCF / MRCI

### GVB Input Example

```
&gen
 basis=6-31g**
 igvb=1
 npair=2
 igeopt=0
&
```

`npair`: number of GVB pairs to correlate (start with the bonds of interest).


```python
# ------------------------------------------------------------------
# Parsing Jaguar Output Files with Python / cclib
# ------------------------------------------------------------------
import re

mock_jaguar_out = '''
  Jaguar version 11.3, release 011

  Total energy:        -76.437819  hartrees

  ATOMIC COORDINATES
   Atom       X          Y          Z
     O     0.000000   0.000000   0.117000
     H     0.000000   0.757000  -0.469000
     H     0.000000  -0.757000  -0.469000

  Mulliken charges:
     O    -0.7024
     H     0.3512
     H     0.3512

  Vibrational frequencies:
    Mode 1:   1643.2 cm-1
    Mode 2:   3802.4 cm-1
    Mode 3:   3919.1 cm-1

  ZPE           =   0.021143 hartrees
  Total enthalpy = -76.410843 hartrees
'''

e_match  = re.search(r'Total energy:\s+([\-0-9.]+)\s+hartrees', mock_jaguar_out)
charges  = re.findall(r'(O|H|C|N|F|Cl)\s+([\-0-9.]+)', mock_jaguar_out)
freqs    = re.findall(r'Mode \d+:\s+([0-9.]+) cm-1', mock_jaguar_out)
zpe      = re.search(r'ZPE\s+=\s+([0-9.]+) hartrees', mock_jaguar_out)

HARTREE2KCAL = 627.5095
print('Parsed from mock Jaguar .out:')
if e_match:
    print(f'  Total energy = {e_match.group(1)} Hartree')
print(f'  Mulliken charges: { {s: float(q) for s, q in charges} }')
print(f'  Frequencies: {freqs} cm-1')
if zpe:
    zpe_kcal = float(zpe.group(1)) * HARTREE2KCAL
    print(f'  ZPE = {zpe.group(1)} Hartree = {zpe_kcal:.2f} kcal/mol')

print()
print('--- cclib usage (with a real Jaguar .out file) ---')
print('import cclib')
print('data = cclib.io.ccread("job.out")')
print('print(data.scfenergies)   # SCF energies in eV')
print('print(data.atomcharges)   # Mulliken / Lowdin charges')
print('print(data.vibfreqs)      # Vibrational frequencies')
print('print(data.zpve)          # Zero-point vibrational energy')
```

    Parsed from mock Jaguar .out:
      Total energy = -76.437819 Hartree
      Mulliken charges: {'O': -0.7024, 'H': 0.3512}
      Frequencies: ['1643.2', '3802.4', '3919.1'] cm-1
      ZPE = 0.021143 Hartree = 13.27 kcal/mol
    
    --- cclib usage (with a real Jaguar .out file) ---
    import cclib
    data = cclib.io.ccread("job.out")
    print(data.scfenergies)   # SCF energies in eV
    print(data.atomcharges)   # Mulliken / Lowdin charges
    print(data.vibfreqs)      # Vibrational frequencies
    print(data.zpve)          # Zero-point vibrational energy


## 4. Jaguar vs ORCA vs PySCF

| Feature | PySCF | ORCA | Jaguar |
|---------|-------|------|--------|
| License | Open-source | Free academic | Commercial |
| DFT | Yes | Yes (faster) | Yes (pseudospectral) |
| LMP2 | -- | DLPNO-MP2 | Yes (native) |
| GVB | -- | -- | Yes (native) |
| DLPNO-CCSD(T) | -- | Yes | -- |
| Transition metals | Yes | Yes | Yes (LACVP** built-in) |
| Solvation | ddCOSMO | CPCM | PBF |
| GUI | -- | -- | Maestro |
| Python API | Native | cclib | cclib |

**Use Jaguar when**: LMP2 accuracy is needed for large systems, GVB for
multireference character, or the Maestro GUI workflow is required.

## 5. Maestro GUI Workflow

The **Maestro** GUI (Schrodinger Suite) provides a graphical interface for Jaguar:

1. **Build / Import**: Draw or import structure into Maestro Project Table
2. **Jaguar panel**: `Applications -> Quantum Mechanics -> Jaguar`
3. **Theory tab**: Choose DFT functional, basis set, charge, multiplicity
4. **Properties tab**: Check: energy, charges, frequencies, NMR
5. **Solvation tab**: Choose PBF solvent (water, chloroform, octanol...)
6. **Submit**: Run locally or on cluster via Job Control
7. **Analyze**: View optimized structure, vibrations, MOs in Maestro

**Project Table**: Tracks all jobs, properties, and structures in one place.
Especially useful for series of related calculations (conformer searches, pKa, etc.).

## Research Connection

Jaguar is used in pharmaceutical and materials research workflows:

- **Drug discovery**: Jaguar pKa prediction module combines GVB + PBF solvation;
  used in large-scale virtual screening at pharmaceutical companies.
- **Transition metal catalysis**: LACVP** built-in ECPs simplify setup for Pd, Rh,
  and Ir-catalyzed reactions in natural product synthesis planning.
- **Conformer energetics**: Fast DFT/LMP2 relative energies for OPLS force field
  parameterisation (protein-ligand binding free energy calculations).
- **Materials**: Jaguar embedded in Schrodinger Materials Science Suite for
  electronic structure of organic semiconductors and battery electrolytes.

## Summary

| Topic | Key Point |
|-------|----------|
| Input format | `&gen ... &` + `&zmat ... &` sections |
| DFT keyword | `dftname=b3lyp`, `basis=def2-tzvp` |
| Opt keyword | `igeopt=1` |
| LMP2 | `mp2=yes`, `lmp2=yes`; O(N^3) scaling |
| GVB-PP | `igvb=1`, `npair=N`; static correlation |
| TM basis | `lacvp**` (ECP for 1st/2nd-row transition metals) |
| PBF solvation | `isolv=2`, `solvent=water` |
| cclib parsing | `cclib.io.ccread('job.out')` |
| GUI | Maestro -> Quantum Mechanics -> Jaguar |

## Exercises

1. **Write Jaguar inputs**: Write `.in` files for (a) a B3LYP/6-31G** geometry
   optimization of methanol and (b) an LMP2/cc-pVTZ single-point on the result.

2. **GVB for bond breaking**: Explain why GVB-PP gives a better description of
   the H2 dissociation curve than RHF or DFT at large bond distances.

3. **LMP2 vs DLPNO-MP2**: Compare the LMP2 (Jaguar) and DLPNO-MP2 (ORCA) approaches.
   Both achieve O(N^3) scaling -- what approximations do they each make?

4. **LACVP** basis**: Look up the LACVP** basis set. Which atoms use ECPs and which
   use all-electron basis functions? Why are ECPs useful for heavy transition metals?

5. **Workflow comparison**: You need to compute the binding free energy of a Pd(II)
   complex with a phosphine ligand. Compare how you would set this up in PySCF,
   ORCA, and Jaguar. What are the pros and cons of each?
