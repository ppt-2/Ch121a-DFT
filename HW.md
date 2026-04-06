# Ch121a — Homework Assignments

> **Course**: Ch121a — Quantum Chemistry & DFT in Practice  
> **Instructor**: Prof. William A. Goddard III  
> **Tools**: PySCF, ORCA, Jaguar, ASE, py3Dmol

---

## HW 1 — Cobalt Phthalocyanine (Co-PC): Geometry and Electronic Structure

**Objective**: Perform DFT geometry optimization of Co-PC and analyze its electronic structure.

### Background

Cobalt phthalocyanine (Co-PC) is a planar macrocyclic complex with a Co²⁺ center coordinated by four isoindole nitrogen atoms. It is an important catalyst for oxygen reduction and CO₂ reduction reactions. Co-PC has an open-shell ground state (Co d⁷, S = ½) making it an excellent test case for spin-polarized DFT.

### Tasks

1. **Geometry Optimization**  
   - Start from the provided `CoPC.xyz` structure (available in `data/molecules/CoPC.xyz`).  
   - Optimize the geometry using B3LYP/def2-SVP with spin multiplicity = 2 (doublet, one unpaired electron).  
   - Compare the optimized Co–N bond length to literature values (~1.93 Å).

2. **Basis Set Convergence**  
   - Repeat the single-point energy at the optimized geometry using def2-SVP, def2-TZVP, and def2-QZVP.  
   - Report how the total energy and Co spin density change with basis set.

3. **Spin-State Analysis**  
   - Compute the doublet (S = ½) and quartet (S = 3/2) spin states.  
   - Report the doublet–quartet splitting (kcal/mol) and identify the ground state.  
   - Visualize the spin density (α − β electron density) using py3Dmol or nglview.

4. **Frontier Orbitals**  
   - Plot the HOMO and LUMO of Co-PC.  
   - Identify the dominant metal d-orbital character of each frontier orbital.

### Deliverables

- Optimized geometry (`.xyz` file) and a table of key bond lengths and angles.  
- Total energies for the three basis sets (convergence plot).  
- Spin-state splitting table.  
- HOMO/LUMO orbital visualizations with orbital energies.

### Suggested Input (PySCF)

```python
from pyscf import gto, dft

mol = gto.M(
    atom=open("data/molecules/CoPC.xyz").read(),
    basis="def2-SVP",
    charge=0,
    spin=1,          # 2S = 1 → doublet
)
mf = dft.UKS(mol)
mf.xc = "b3lyp"
mf.kernel()
```

---

## HW 2 — Co-PC-CO Intermediate: CO Adsorption and Desorption Energy

**Objective**: Model CO binding to Co-PC and compute the CO desorption energy.

### Background

Co-PC can catalyze CO₂ reduction to CO. Once CO is produced, it must desorb from the Co center for the catalyst to turn over. The Co-PC-CO intermediate (CO bound axially to Co) can be a resting state that limits turnover frequency. Computing the desorption energy  
ΔE_des = E(Co-PC) + E(CO) − E(Co-PC-CO)  
gives insight into catalyst poisoning and turnover.

### Tasks

1. **Build the Co-PC-CO Geometry**  
   - Starting from the optimized Co-PC geometry (HW 1), place a CO molecule ~2.0 Å above the Co center along the axial direction (z-axis).  
   - Use ASE or manual coordinate editing to construct the initial guess.

2. **Geometry Optimization of Co-PC-CO**  
   - Optimize Co-PC-CO at B3LYP/def2-SVP.  
   - Report the optimized Co–C(O) bond length and the C–O bond length inside the adduct (compare to free CO: 1.128 Å).  
   - Examine whether spin state changes upon CO binding (doublet Co-PC → singlet or doublet Co-PC-CO?).

3. **Desorption Energy**  
   - Compute ΔE_des = E(Co-PC) + E(CO) − E(Co-PC-CO) at B3LYP/def2-SVP.  
   - Repeat at B3LYP/def2-TZVP for a higher-quality estimate.  
   - Add zero-point energy (ZPE) correction by computing vibrational frequencies for each species.

4. **Thermochemistry at 298 K**  
   - Use the harmonic approximation to compute ΔH and ΔG for desorption at 298 K and 1 atm.  
   - Discuss whether CO is likely to poison the Co-PC catalyst under ambient conditions.

5. *(Optional)* **Potential Energy Surface Along Co–CO Distance**  
   - Scan the Co–C distance from 1.8 Å to 4.0 Å (step 0.1 Å) with all other coordinates relaxed.  
   - Plot E vs. Co–C distance to visualize the binding well and dissociation limit.

### Deliverables

- Optimized geometries for Co-PC, CO, and Co-PC-CO.  
- Table of ΔE, ΔH, ΔG (kcal/mol) for CO desorption at B3LYP/def2-SVP and def2-TZVP.  
- Discussion of spin-state change and its effect on the binding energy.  
- *(Optional)* PES scan plot.

---

## HW 3 — Water O–H Bond Scan: Deriving the O–H Force Constant

**Objective**: Perform a constrained O–H bond length scan in water and extract the harmonic force constant from the resulting potential energy curve.

### Background

Molecular mechanics force fields represent covalent bonds as harmonic springs:  
V(r) = ½ k (r − r₀)²  
where k is the force constant (units: kcal mol⁻¹ Å⁻²) and r₀ is the equilibrium bond length. DFT calculations provide a first-principles reference for these parameters by scanning the bond length and fitting the resulting energy profile.

### Tasks

1. **Geometry Optimization of Water**  
   - Optimize H₂O at B3LYP/def2-TZVP.  
   - Record the equilibrium O–H bond length r₀ and H–O–H angle.

2. **O–H Bond Length Scan**  
   - Fix one O–H bond length and vary it from 0.70 Å to 1.40 Å in steps of 0.05 Å.  
   - At each step, perform a single-point energy calculation (or relax all other degrees of freedom).  
   - Collect the relative energy ΔE = E(r) − E(r₀) in kcal/mol.

3. **Harmonic Fit**  
   - Fit the DFT energies near the minimum (e.g., r₀ ± 0.15 Å) to V(r) = ½ k (r − r₀)².  
   - Extract the O–H force constant k in kcal mol⁻¹ Å⁻².  
   - Convert to SI units (N/m) for comparison with experiment (k_exp ≈ 780 N/m for O–H).

4. **Comparison with Vibrational Frequency**  
   - Compute the harmonic vibrational frequency from your fitted k:  
     ν = (1/2π) √(k/μ), where μ = m_O m_H / (m_O + m_H) is the reduced mass.  
   - Compare to the B3LYP/def2-TZVP harmonic frequency from a normal-mode analysis.  
   - Compare to the experimental O–H stretch frequency (~3657 cm⁻¹ for the symmetric stretch).

5. **Morse Potential Fit** *(Optional)*  
   - Fit the full scan to a Morse potential: V(r) = D_e [1 − e^(−α(r−r₀))]².  
   - Extract D_e (dissociation energy) and α, and compare D_e to the O–H bond dissociation energy (~119 kcal/mol).

### Deliverables

- Table of O–H scan energies (r in Å, ΔE in kcal/mol).  
- Plot of ΔE vs. r with both the DFT data points and the fitted harmonic (and Morse, optional) curves.  
- Extracted force constant k (kcal mol⁻¹ Å⁻² and N/m).  
- Comparison table of DFT frequency, scan-derived frequency, and experimental frequency.

### Suggested Input Snippet

```python
import numpy as np
from pyscf import gto, dft

r0 = 0.9584          # initial O-H guess (Angstrom)
r_scan = np.arange(0.70, 1.45, 0.05)
energies = []

for r in r_scan:
    mol = gto.M(
        atom=f"O 0 0 0; H 0 0 {r}; H 0 0.757 -0.469",
        basis="def2-TZVP", charge=0, spin=0,
    )
    mf = dft.RKS(mol, xc="b3lyp").run()
    energies.append(mf.e_tot)
```

---

## HW 4 — n-Butane C–C–C–C Dihedral Scan and Force-Field Comparison

**Objective**: Scan the central C–C–C–C dihedral angle in n-butane using DFT and compare the resulting torsion profile to CHARMM and OPLS force-field predictions.

### Background

Torsion (dihedral) potentials govern the conformational flexibility of hydrocarbon chains and are a key component of biomolecular force fields. The n-butane C–C–C–C dihedral has three minima (anti at 180°, two gauche at ±60°) and two distinct barriers (gauche–gauche at ±120°, eclipsed at 0°). Matching DFT to force-field predictions validates the force-field parameters.

### Tasks

1. **DFT Dihedral Scan**  
   - Optimize n-butane at B3LYP/6-31G* to obtain the reference geometry.  
   - Scan the central C1–C2–C3–C4 dihedral from 0° to 360° in steps of 10° (or 15°).  
   - At each step, constrain the dihedral and relax all other coordinates.  
   - Plot ΔE (kcal/mol) vs. dihedral angle φ.

2. **Key Conformational Energies**  
   - Extract and report:
     | Conformer | φ (°) | ΔE (kcal/mol) |
     |-----------|--------|---------------|
     | Anti      | 180    | 0.00 (reference) |
     | Gauche    | ±60    | ~0.9 kcal/mol |
     | Eclipsed (C–C) | ±120 | ~3.4 kcal/mol |
     | Fully eclipsed | 0  | ~4.5–5.0 kcal/mol |

3. **CHARMM Force-Field Torsion Profile**  
   - Using the CHARMM36 alkane parameters for the C–C–C–C dihedral:
     ```
     ! CHARMM CGenFF / CHARMM36 C–C–C–C dihedral terms (kcal/mol)
     ! Kchi  n  delta
       0.2000  3  0.00   ! gauche periodicity
       0.2500  2  0.00
       0.1600  1  0.00
     ```
   - Compute V_CHARMM(φ) = Σ Kchi [1 + cos(n·φ − delta)] and plot alongside DFT.

4. **OPLS Force-Field Torsion Profile**  
   - Using OPLS-AA parameters for the alkane C–C–C–C dihedral (Ryckaert–Bellemans form):
     ```
     ! OPLS-AA C–C–C–C dihedral (kcal/mol)
     ! F1        F2        F3        F4
       1.3006   -0.3137    0.2284    0.0000
     ```
     V_OPLS(φ) = F1(1 + cosφ)/2 + F2(1 − cos2φ)/2 + F3(1 + cos3φ)/2
   - Plot V_OPLS(φ) alongside the DFT and CHARMM profiles.

5. **Quantitative Comparison**  
   - Tabulate the anti→gauche barrier, gauche→anti barrier, and gauche–gauche barrier for DFT, CHARMM, and OPLS.  
   - Compute the root-mean-square deviation (RMSD) of each force field from DFT over all scanned angles.

6. *(Optional)* **Fourier Fitting**  
   - Fit the DFT dihedral profile to a truncated Fourier series:  
     V(φ) = Σₙ [aₙ cos(nφ) + bₙ sin(nφ)], n = 1…6  
   - Derive force-field-ready dihedral parameters directly from DFT.

### Deliverables

- DFT scan data table (φ in degrees, ΔE in kcal/mol).  
- Overlay plot of DFT, CHARMM, and OPLS dihedral profiles.  
- Comparison table of barrier heights (kcal/mol).  
- RMSD (kcal/mol) of CHARMM and OPLS vs. DFT.  
- *(Optional)* Fourier-fit coefficients and comparison.

### Suggested Input Snippet

```python
import numpy as np
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize

# After optimizing n-butane, constrain the dihedral and scan
dihedrals = np.arange(0, 370, 10)
energies = []

for phi in dihedrals:
    # Set dihedral constraint in your geometry optimizer
    # (use ASE or geomeTRIC with dihedral constraints)
    # Then run a constrained optimization and record energy
    pass

# CHARMM torsion profile (illustrative)
def V_charmm(phi_deg):
    phi = np.radians(phi_deg)
    return (0.2000 * (1 + np.cos(3 * phi)) +
            0.2500 * (1 + np.cos(2 * phi)) +
            0.1600 * (1 + np.cos(1 * phi)))

# OPLS torsion profile (Ryckaert-Bellemans)
def V_opls(phi_deg):
    phi = np.radians(phi_deg)
    F1, F2, F3 = 1.3006, -0.3137, 0.2284
    return (F1 * (1 + np.cos(phi)) / 2 +
            F2 * (1 - np.cos(2 * phi)) / 2 +
            F3 * (1 + np.cos(3 * phi)) / 2)
```

---

## General Notes

- All DFT calculations use the **B3LYP** functional unless otherwise stated.  
- Energies are reported relative to the lowest-energy structure in each scan (ΔE in kcal/mol).  
- Include convergence checks (SCF and geometry) in your submitted notebooks.  
- Submit as a Jupyter notebook (`.ipynb`) with all cells executed and output visible.  
- Use PySCF for all calculations unless ORCA or Jaguar is specifically requested.

## Grading Rubric (per HW)

| Item | Points |
|------|--------|
| Correct calculation setup and convergence | 30 |
| Accurate numerical results and units | 30 |
| Quality of plots and tables | 20 |
| Discussion and physical interpretation | 20 |
| **Total** | **100** |
