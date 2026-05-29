# 05 — Practical DFT Calculations: Cycloadditions, Enzyme Active Sites & Metal-Catalysed Coupling

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ppt-2/Ch121a-DFT/blob/main/Module1_DFT/notebooks_v2/05_practical_calculations.ipynb)

## 🎯 Learning Objectives

- DFT (B3LYP/def2-SVP) to three representative reaction classes
- **frontier molecular orbital (FMO) theory** to rationalise Diels-Alder reactivity
- **QM cluster model** of an enzyme active site
- **oxidative-addition** step in a Pd-catalysed cross-coupling cycle
- Orbital-energy diagrams for organometallic complexes

---
- **Use ORCA or Jaguar for your workflow**
## 1. (4+2) Diels-Alder Cycloaddition

### 1.1 Theory

The Diels-Alder reaction is a pericyclic [4+2] cycloaddition between a **conjugated diene** (4π electrons) and a **dienophile** (2π electrons) to form a cyclohexene ring. It is thermally allowed under the Woodward-Hoffmann rules because the orbital symmetry of the HOMO of one reactant matches that of the LUMO of the other.

**Frontier Molecular Orbital (FMO) theory** (Fukui, 1952) predicts the rate-determining orbital interaction:

$$\Delta E_{\text{stab}} \propto -\frac{|\langle \psi_i | \hat{H} | \psi_j \rangle|^2}{\epsilon_j - \epsilon_i}$$

For a **normal electron-demand** DA reaction (electron-rich diene, electron-poor dienophile):
$$\text{Dominant interaction: } \text{HOMO}_{\text{diene}} \leftrightarrow \text{LUMO}_{\text{dienophile}}$$

The rate increases as the **FMO gap** $\Delta\epsilon = \epsilon_{\text{LUMO,dienophile}} - \epsilon_{\text{HOMO,diene}}$ decreases.

**Diene requirement:** The diene must adopt the **s-cis** conformation (C=C–C=C dihedral ≈ 0°) for the orbitals to overlap with the dienophile.

$$\text{s-cis butadiene} + \text{ethylene} \xrightarrow{[4+2]} \text{cyclohexene}$$

### 1.2 Geometry Notes

| Species | Conformation | Bond lengths used |
|---------|-------------|-------------------|
| 1,3-Butadiene | s-cis (dihedral = 0°) | C=C 1.343 Å, C–C 1.467 Å |
| Ethylene | planar | C=C 1.330 Å |
| Cyclohexene | planar hexagon (approx.) | all ring C–C 1.500 Å; C1=C2 is the **retained** C=C double bond (sp2, 1H each); C3–C6 are sp3 (2H each) |

> **Note:** For quantitative reaction energies all geometries should be optimised at the same level of theory. The planar cyclohexene used here gives a qualitative result; use a geometry-optimised structure for production calculations. The [4+2] cycloaddition forms **cyclohexene** (one C=C double bond remains in the ring), *not* fully-saturated cyclohexane.


```python
%%time
# =============================================================================
# Ch121a: Quantum Chemistry & DFT — Notebook 05: Practical Calculations
# License: GPL-3.0 (https://www.gnu.org/licenses/gpl-3.0.en.html)
# =============================================================================
import numpy as np
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt
from pyscf import gto, dft

HA2EV   = 27.2114
HA2KJ   = 2625.5
HA2KCAL = 627.509

# ------------------------------------------------------------------
# 1a) s-cis 1,3-Butadiene
#     C1=C2-C3=C4, dihedral(C1-C2-C3-C4) = 0 deg (s-cis).
#     C=C: 1.343 Ang, C-C: 1.467 Ang, ring angle at C2/C3: 124.5 deg.
# ------------------------------------------------------------------
bd_geom = '''
C  -0.763   1.105   0.000
C   0.000   0.000   0.000
C   1.467   0.000   0.000
C   2.230   1.105   0.000
H  -0.297   2.086   0.000
H  -1.845   1.018   0.000
H  -0.503  -0.959   0.000
H   1.970  -0.959   0.000
H   3.312   1.018   0.000
H   1.765   2.086   0.000
'''

mol_bd = gto.Mole()
mol_bd.atom = bd_geom
mol_bd.basis = 'def2-SVP'
mol_bd.spin = 0
mol_bd.verbose = 0
mol_bd.build()

mf_bd = dft.RKS(mol_bd)
mf_bd.xc = 'B3LYP'
mf_bd.verbose = 0
mf_bd.kernel()

print(f's-cis Butadiene (C4H6):  E = {mf_bd.e_tot:.6f} Ha,  converged: {mf_bd.converged}')

# ------------------------------------------------------------------
# 1b) Ethylene (dienophile)
#     C=C: 1.330 Ang, H-C=C angle: 121.3 deg
# ------------------------------------------------------------------
et_geom = '''
C  -0.665   0.000   0.000
C   0.665   0.000   0.000
H  -1.230   0.925   0.000
H  -1.230  -0.925   0.000
H   1.230   0.925   0.000
H   1.230  -0.925   0.000
'''

mol_et = gto.Mole()
mol_et.atom = et_geom
mol_et.basis = 'def2-SVP'
mol_et.spin = 0
mol_et.verbose = 0
mol_et.build()

mf_et = dft.RKS(mol_et)
mf_et.xc = 'B3LYP'
mf_et.verbose = 0
mf_et.kernel()

print(f'Ethylene (C2H4):         E = {mf_et.e_tot:.6f} Ha,  converged: {mf_et.converged}')
```


```python
# ------------------------------------------------------------------
# Frontier Molecular Orbital (FMO) Analysis
# ------------------------------------------------------------------
def get_fmo(mf):
    """Return (homo_eV, lumo_eV) for a converged closed-shell SCF object."""
    mo_e = mf.mo_energy * HA2EV
    occ  = mf.mo_occ
    homo_idx = np.where(occ > 0)[0][-1]
    lumo_idx = homo_idx + 1
    return mo_e[homo_idx], mo_e[lumo_idx]

homo_bd, lumo_bd = get_fmo(mf_bd)
homo_et, lumo_et = get_fmo(mf_et)

# Normal electron-demand: HOMO(diene) <--> LUMO(dienophile)
gap_ne = lumo_et - homo_bd
# Inverse electron-demand: LUMO(diene) <--> HOMO(dienophile)
gap_ie = lumo_bd - homo_et

print('Frontier Molecular Orbital Energies  (B3LYP/def2-SVP)')
print('=' * 54)
print(f'  s-cis Butadiene  HOMO: {homo_bd:+.2f} eV')
print(f'  s-cis Butadiene  LUMO: {lumo_bd:+.2f} eV')
print()
print(f'  Ethylene         HOMO: {homo_et:+.2f} eV')
print(f'  Ethylene         LUMO: {lumo_et:+.2f} eV')
print()
print(f'  HOMO(diene)-LUMO(dienophile) gap: {gap_ne:.2f} eV  [normal demand]')
print(f'  LUMO(diene)-HOMO(dienophile) gap: {gap_ie:.2f} eV  [inverse demand]')
print()
print('  The smaller gap dominates reactivity.')
print('  EWG on dienophile lower its LUMO, accelerating normal-demand DA.')
```


```python
# ------------------------------------------------------------------
# FMO Diagram
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))

col_d, col_s, hw = 0.20, 0.80, 0.13   # x-positions, half-width

levels = [
    (homo_bd, f'HOMO\n{homo_bd:.2f} eV', col_d, '#1565C0', 3),
    (lumo_bd, f'LUMO\n{lumo_bd:.2f} eV', col_d, '#90CAF9', 2),
    (homo_et, f'HOMO\n{homo_et:.2f} eV', col_s, '#B71C1C', 3),
    (lumo_et, f'LUMO\n{lumo_et:.2f} eV', col_s, '#EF9A9A', 2),
]
for energy, label, col, color, lw in levels:
    ax.plot([col - hw, col + hw], [energy, energy], color=color, lw=lw)
    ax.text(col + hw + 0.03, energy, label, va='center', fontsize=8)

# Dominant interaction arrow: HOMO(diene) --> LUMO(dienophile)
ax.annotate(
    '',
    xy=(col_s - hw, lumo_et), xytext=(col_d + hw, homo_bd),
    arrowprops=dict(arrowstyle='->', color='green', lw=1.8, linestyle='dashed'),
)
mid_y = (homo_bd + lumo_et) / 2
ax.text(0.50, mid_y, f'\u0394 = {gap_ne:.1f} eV', ha='center', va='bottom',
        fontsize=9, color='green', style='italic')

ax.set_xlim(0, 1)
ax.set_xticks([col_d, col_s])
ax.set_xticklabels(['Butadiene\n(diene)', 'Ethylene\n(dienophile)'], fontsize=10)
ax.set_ylabel('Orbital energy (eV)', fontsize=11)
ax.set_title('FMO Diagram — (4+2) Diels-Alder\nNormal Electron Demand', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```


```python
%%time
# ------------------------------------------------------------------
# 1c) Cyclohexene product — planar hexagonal approximation
#     C1 (1.500,0,0) and C2 (0.750,1.299,0) are the sp2 carbons retaining
#     the C=C double bond from the [4+2] cycloaddition (1H each).
#     C3-C6 are sp3 carbons (2H each); ring angle forced to 120 deg.
#     All ring C-C = 1.500 Ang (uniform, approximation).
#     For production use, replace with a geometry-optimised structure.
# ------------------------------------------------------------------
cx_geom = '''
C   1.500   0.000   0.000
C   0.750   1.299   0.000
C  -0.750   1.299   0.000
C  -1.500   0.000   0.000
C  -0.750  -1.299   0.000
C   0.750  -1.299   0.000
H   2.585   0.000   0.000
H   1.292   2.239   0.000
H  -1.237   2.143   0.487
H  -1.237   2.143  -0.487
H  -2.475   0.000   0.487
H  -2.475   0.000  -0.487
H  -1.237  -2.143   0.487
H  -1.237  -2.143  -0.487
H   1.237  -2.143   0.487
H   1.237  -2.143  -0.487
'''

mol_cx = gto.Mole()
mol_cx.atom = cx_geom
mol_cx.basis = 'def2-SVP'
mol_cx.spin = 0
mol_cx.verbose = 0
mol_cx.build()

mf_cx = dft.RKS(mol_cx)
mf_cx.xc = 'B3LYP'
mf_cx.verbose = 0
mf_cx.kernel()

print(f'Cyclohexene (C6H10):     E = {mf_cx.e_tot:.6f} Ha,  converged: {mf_cx.converged}')
print('(planar ring approximation — optimise for production use)')
print()

# Reaction energy: butadiene + ethylene -> cyclohexene
dE_kj = (mf_cx.e_tot - mf_bd.e_tot - mf_et.e_tot) * HA2KJ
dE_kc = dE_kj / 4.184

print('Diels-Alder Reaction Energy  (B3LYP/def2-SVP, single-point)')
print('=' * 54)
print(f'  E(butadiene):    {mf_bd.e_tot:.6f} Ha')
print(f'  E(ethylene):     {mf_et.e_tot:.6f} Ha')
print(f'  E(cyclohexene):  {mf_cx.e_tot:.6f} Ha')
print()
print(f'  Delta_E(rxn) = {dE_kj:.1f} kJ mol^-1  ({dE_kc:.1f} kcal mol^-1)')
print()
print('  Reference (B3LYP/6-31G*, optimised geometries): ~-168 kJ mol^-1')
print('  The less negative value here reflects the unoptimised ring geometry.')
```

---
## 2. Enzyme Active-Site Modelling with DFT

### 2.1 QM Cluster Approach

Enzymes catalyse reactions by providing a specific electrostatic and hydrogen-bond environment. The **QM cluster** (or QM/MM) strategy represents the active site with a small quantum-mechanical model while omitting the protein bulk.

**Serine protease catalytic triad** (Ser–His–Asp): the serine hydroxyl is activated by a proton relay through histidine. We model the key step:

$$\text{Ser-OH} \cdots \text{His} \xrightarrow{\text{proton transfer}} \text{Ser-O}^- \cdots \text{HisH}^+$$

**Simplified QM model:**
- Serine → **methanol** (CH₃OH)
- Histidine → **imidazole** (C₃H₄N₂)

The interaction energy is:
$$\Delta E_{\text{int}} = E_{\text{complex}} - E_{\text{methanol}} - E_{\text{imidazole}}$$

> **Basis-set superposition error (BSSE):** Monomer energies computed in the full complex basis set are artificially lower because each fragment can use basis functions of the other. For quantitative results, apply the **Boys-Bernardi counterpoise correction**. For this qualitative demonstration we show the uncorrected value; BSSE typically accounts for ~10–30% of a hydrogen-bond interaction energy at the def2-SVP level.

### 2.2 Hydrogen-Bond Geometry

| Parameter | Value used |
|-----------|------------|
| O–H (methanol) | 0.960 Å |
| N···H distance | 1.840 Å |
| O–H···N angle | ≈ 180° (linear H-bond) |


```python
%%time
# =============================================================================
# Section 2: Enzyme Active-Site Modelling
# QM cluster: methanol (Serine) + imidazole (Histidine)
# H-bond: O-H...N, O-H = 0.960 Ang, H...N = 1.840 Ang (O...N = 2.800 Ang)
# =============================================================================

# Methanol (Serine surrogate): O at origin, H along +y for H-bond
met_geom = '''
O   0.000   0.000   0.000
H   0.000   0.960   0.000
C   0.000  -0.455  -1.356
H   0.890  -0.455  -1.950
H  -0.890  -0.455  -1.950
H   0.000   0.435  -2.050
'''

# Imidazole (Histidine surrogate): N lone-pair N at (0, 2.800, 0)
# so H...N distance = 2.800 - 0.960 = 1.840 Ang
imid_geom = '''
N   0.370   4.671   0.000
C   1.050   3.557   0.000
N   0.000   2.800   0.000
C  -1.119   3.597   0.000
C  -0.892   4.924   0.000
H   1.023   5.378   0.000
H   2.129   3.508   0.000
H  -2.120   3.126   0.000
H  -1.252   5.938   0.000
'''

# Combined complex geometry
cplx_geom = met_geom + imid_geom

def run_rks(geom, label):
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = 'def2-SVP'
    mol.spin = 0
    mol.verbose = 0
    mol.build()
    mf = dft.RKS(mol)
    mf.xc = 'B3LYP'
    mf.verbose = 0
    mf.kernel()
    print(f'  {label}: E = {mf.e_tot:.6f} Ha,  converged: {mf.converged}')
    return mf

print('Serine-Histidine QM Cluster  (B3LYP/def2-SVP)')
print('=' * 50)
mf_met  = run_rks(met_geom,  'Methanol (Ser model) ')
mf_imid = run_rks(imid_geom, 'Imidazole (His model)')
mf_cplx = run_rks(cplx_geom, 'MeOH + imidazole     ')

E_int_kj = (mf_cplx.e_tot - mf_met.e_tot - mf_imid.e_tot) * HA2KJ

print()
print(f'  Delta_E(interaction) = {E_int_kj:.1f} kJ mol^-1  (uncorrected for BSSE)')
print()
print('  Typical O-H...N hydrogen-bond strength: -20 to -35 kJ mol^-1.')
print('  This electrostatic stabilisation positions the serine nucleophile')
print('  for attack on the substrate carbonyl in the proteolytic mechanism.')
```

### 2.3 Interpreting the Interaction Energy

The interaction energy can be decomposed (e.g., with energy decomposition analysis, EDA) into:

| Component | Physical origin |
|-----------|----------------|
| Electrostatic | Coulombic attraction between permanent multipoles |
| Induction | Polarisation of each monomer by the other's field |
| Dispersion | Correlation-driven London forces |
| Exchange-repulsion | Pauli repulsion at short range |

For hydrogen bonds, **electrostatics** and **induction** are dominant. DFT with a GGA or hybrid functional (B3LYP) describes these well; dispersion-corrected functionals (B3LYP-D3, ωB97X-D) are recommended when dispersion contributes significantly (e.g., π-stacking).

---
## 3. Transition-Metal-Catalysed Cross-Coupling: Oxidative Addition at Pd

### 3.1 The Pd Catalytic Cycle

Palladium-catalysed cross-coupling reactions (Suzuki, Negishi, Heck, …) share a three-step cycle:

1. **Oxidative addition (OA):** Pd(0)L₂ + R–X → Pd(II)(R)(X)L₂  
   Pd is oxidised from 0 to +2; the R–X bond is cleaved.
2. **Transmetalation (TM):** Pd(II)(R)(X)L₂ + R'–M → Pd(II)(R)(R')L₂ + M–X
3. **Reductive elimination (RE):** Pd(II)(R)(R')L₂ → Pd(0)L₂ + R–R'  
   Pd returns to the 0 state; the product C–C bond forms.

**Oxidative addition** is often rate-determining. Its barrier depends on:
- The electron density at Pd (governed by the ligand L)
- The strength of the R–X bond being broken
- Steric bulk around the metal

### 3.2 Model Reaction

We model OA with **ammonia as a phosphine surrogate** and **Cl₂ as the R–X oxidant**:

$$\text{Pd(NH}_3)_2 + \text{Cl}_2 \rightarrow \text{trans-Pd(NH}_3)_2\text{Cl}_2$$

This avoids the large PPh₃ ligands while retaining the essential electronic change (Pd⁰ → Pd²⁺, d¹⁰ → d⁸).

### 3.3 Effective Core Potentials (ECPs) for Pd

Pd (Z = 46) lies in the 4d series. We use the **def2-ECP** (Stuttgart 28-electron ECP) bundled with the def2-SVP valence basis. In PySCF, setting `mol.ecp = {'Pd': 'def2-SVP'}` automatically loads the ECP for Pd while the light atoms (N, H, Cl) use all-electron def2-SVP.

| Molecule | Pd oxidation state | d-electron count | Geometry |
|----------|--------------------|-----------------|----------|
| Pd(NH₃)₂ | 0 | d¹⁰ | linear (T-shaped w/ lone pair) |
| trans-Pd(NH₃)₂Cl₂ | +2 | d⁸ | square planar |


```python
%%time
# =============================================================================
# Section 3: Pd-Catalysed Cross-Coupling — Oxidative Addition
#
# Reaction: Pd(NH3)2  +  Cl2  -->  trans-Pd(NH3)2Cl2
#
# Basis: def2-SVP for all atoms
# ECP:   def2-SVP ECP for Pd (28 core electrons replaced)
# =============================================================================

# ------------------------------------------------------------------
# Pd(NH3)2 — Pd(0), d10, singlet, linear coordination
# Pd-N: 2.080 Ang, N-H: 1.015 Ang, H-N-H: 106.7 deg
# ------------------------------------------------------------------
pd0_geom = '''
Pd  0.000   0.000   0.000
N   0.000   0.000   2.080
N   0.000   0.000  -2.080
H   0.938   0.000   2.463
H  -0.469   0.813   2.463
H  -0.469  -0.813   2.463
H   0.938   0.000  -2.463
H  -0.469   0.813  -2.463
H  -0.469  -0.813  -2.463
'''

mol_pd0 = gto.Mole()
mol_pd0.atom    = pd0_geom
mol_pd0.basis   = 'def2-SVP'
mol_pd0.ecp     = {'Pd': 'def2-SVP'}
mol_pd0.spin    = 0
mol_pd0.verbose = 0
mol_pd0.build()

mf_pd0 = dft.RKS(mol_pd0)
mf_pd0.xc      = 'B3LYP'
mf_pd0.verbose = 0
mf_pd0.kernel()

print(f'Pd(NH3)2          [Pd(0), d10]:  E = {mf_pd0.e_tot:.6f} Ha,  converged: {mf_pd0.converged}')

# ------------------------------------------------------------------
# Cl2 oxidant
# Cl-Cl: 1.988 Ang
# ------------------------------------------------------------------
cl2_geom = '''
Cl  0.000   0.000   0.000
Cl  0.000   0.000   1.988
'''

mol_cl2 = gto.Mole()
mol_cl2.atom    = cl2_geom
mol_cl2.basis   = 'def2-SVP'
mol_cl2.spin    = 0
mol_cl2.verbose = 0
mol_cl2.build()

mf_cl2 = dft.RKS(mol_cl2)
mf_cl2.xc      = 'B3LYP'
mf_cl2.verbose = 0
mf_cl2.kernel()

print(f'Cl2:                              E = {mf_cl2.e_tot:.6f} Ha,  converged: {mf_cl2.converged}')

# ------------------------------------------------------------------
# trans-Pd(NH3)2Cl2 — Pd(II), d8, singlet, square planar
# Pd-N: 2.020 Ang, Pd-Cl: 2.300 Ang
# N-Pd-N and Cl-Pd-Cl are trans (180 deg)
# ------------------------------------------------------------------
pd2_geom = '''
Pd  0.000   0.000   0.000
N   2.020   0.000   0.000
N  -2.020   0.000   0.000
Cl  0.000   2.300   0.000
Cl  0.000  -2.300   0.000
H   2.403   0.938   0.000
H   2.403  -0.469   0.813
H   2.403  -0.469  -0.813
H  -2.403   0.938   0.000
H  -2.403  -0.469   0.813
H  -2.403  -0.469  -0.813
'''

mol_pd2 = gto.Mole()
mol_pd2.atom    = pd2_geom
mol_pd2.basis   = 'def2-SVP'
mol_pd2.ecp     = {'Pd': 'def2-SVP'}
mol_pd2.spin    = 0
mol_pd2.verbose = 0
mol_pd2.build()

mf_pd2 = dft.RKS(mol_pd2)
mf_pd2.xc      = 'B3LYP'
mf_pd2.verbose = 0
mf_pd2.kernel()

print(f'trans-Pd(NH3)2Cl2 [Pd(II), d8]:  E = {mf_pd2.e_tot:.6f} Ha,  converged: {mf_pd2.converged}')
print()

# Reaction energy for oxidative addition
dE_oa_kj = (mf_pd2.e_tot - mf_pd0.e_tot - mf_cl2.e_tot) * HA2KJ

print('Oxidative Addition  Pd(NH3)2 + Cl2 --> trans-Pd(NH3)2Cl2')
print('=' * 58)
print(f'  Delta_E(OA) = {dE_oa_kj:.1f} kJ mol^-1  (single-point, no ZPVE)')
print()
print('  A negative Delta_E indicates a thermodynamically favourable OA.')
print('  Real ligands (PR3) tune the Pd electron density and OA barrier.')
```


```python
# ------------------------------------------------------------------
# Orbital analysis: near-HOMO region for Pd(0) and Pd(II) complexes
# ------------------------------------------------------------------
def mo_table(mf, label, n_occ=6, n_virt=4):
    mo_e = mf.mo_energy * HA2EV
    occ  = mf.mo_occ
    hi   = np.where(occ > 0)[0][-1]
    homo_e = mo_e[hi]
    lumo_e = mo_e[hi + 1]
    gap    = lumo_e - homo_e
    print(f'{label}')
    print(f'  HOMO = {homo_e:.3f} eV   LUMO = {lumo_e:.3f} eV   gap = {gap:.3f} eV')
    print('  Occupied (last few):')
    for i in range(max(0, hi - n_occ + 1), hi + 1):
        tag = '  <- HOMO' if i == hi else ''
        print(f'    MO {i+1:4d}: {mo_e[i]:+8.3f} eV  occ={int(occ[i])}{tag}')
    print('  Virtual (first few):')
    for i in range(hi + 1, min(hi + 1 + n_virt, len(mo_e))):
        tag = '  <- LUMO' if i == hi + 1 else ''
        print(f'    MO {i+1:4d}: {mo_e[i]:+8.3f} eV  occ={int(occ[i])}{tag}')
    print()
    return hi, mo_e

hi_pd0, e_pd0 = mo_table(mf_pd0, 'Pd(NH3)2  [Pd(0), d10]  (B3LYP/def2-SVP)')
hi_pd2, e_pd2 = mo_table(mf_pd2, 'trans-Pd(NH3)2Cl2  [Pd(II), d8]  (B3LYP/def2-SVP)')
```


```python
# ------------------------------------------------------------------
# Orbital energy diagram for Pd(0) and Pd(II) complexes
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

datasets = [
    (mf_pd0, hi_pd0, e_pd0, 'Pd(NH\u2083)\u2082\n[Pd(0), d\u00b9\u2070]'),
    (mf_pd2, hi_pd2, e_pd2, 'trans-Pd(NH\u2083)\u2082Cl\u2082\n[Pd(II), d\u2078]'),
]

n_occ_show, n_virt_show = 10, 5

for ax, (mf, hi, mo_e, title) in zip(axes, datasets):
    occ_e  = mo_e[max(0, hi - n_occ_show + 1): hi + 1]
    virt_e = mo_e[hi + 1: hi + 1 + n_virt_show]

    for e in occ_e:
        ax.plot([0.25, 0.75], [e, e], 'b-', lw=1.5, alpha=0.6)
    for e in virt_e:
        ax.plot([0.25, 0.75], [e, e], 'r--', lw=1.5, alpha=0.6)

    # Highlight HOMO and LUMO
    ax.plot([0.25, 0.75], [mo_e[hi], mo_e[hi]], 'b-', lw=3.5,
            label=f'HOMO ({mo_e[hi]:.2f} eV)')
    ax.plot([0.25, 0.75], [mo_e[hi + 1], mo_e[hi + 1]], 'r-', lw=3.5,
            label=f'LUMO ({mo_e[hi+1]:.2f} eV)')

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel('Orbital energy (eV)', fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('MO Energies Before/After Oxidative Addition\n(B3LYP/def2-SVP)', fontsize=12)
plt.tight_layout()
plt.show()
```

---
## Summary

| System | Key DFT observable | Chemical insight |
|--------|--------------------|------------------|
| Diels-Alder | FMO gap $\Delta\epsilon$ | Smaller gap → more reactive; EWG on dienophile accelerates |
| Enzyme cluster | $\Delta E_{\text{int}}$ | H-bond stabilises nucleophilic serine for catalysis |
| Pd OA | $\Delta E_{\text{OA}}$ | Thermodynamic driving force for Pd⁰ → Pd²⁺ |

### Next Steps

- **Geometry optimisation** (Notebook 02): all single-point energies here should be repeated with optimised geometries for quantitative results.
- **Transition-state search**: locate the transition state for the DA cycloaddition and compute the activation barrier $\Delta E^\ddagger$.
- **Dispersion correction**: add `-D3` or `-D3BJ` to capture London forces (important for enzyme modelling and organometallic reactions).
- **Larger QM cluster**: extend the enzyme model to include the aspartate residue of the catalytic triad and the oxyanion hole.
- **Transmetalation / reductive elimination**: complete the Pd catalytic cycle by computing the remaining two steps.
