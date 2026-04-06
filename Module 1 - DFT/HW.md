# Ch121a — Homework problems

> **Course**: Ch121a — Quantum Chemistry & DFT in Practice   
> **Tools**: PySCF, ORCA, Jaguar, ASE, py3Dmol

---

## HW 1 — Cobalt Phthalocyanine (Co-PC): Geometry and Electronic Structure

**Objective**: Perform DFT geometry optimization of Co-PC and analyze its electronic structure.

### Background

Cobalt phthalocyanine (Co-PC) is a planar macrocyclic complex with a Co²⁺ center coordinated by four isoindole nitrogen atoms. It is an important catalyst for oxygen reduction and CO₂ reduction.

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
   - Visualize the spin density (α − β electron density) using py3Dmol or Avogadro2.

4. **Frontier Orbitals**  
   - Plot the HOMO and LUMO of Co-PC.  
   - Identify the dominant metal d-orbital character of each frontier orbital.

---

## HW 2 — Co-PC-CO Intermediate: CO Adsorption and Desorption Energy

**Objective**: Model CO binding to Co-PC and compute the CO desorption energy.

### Background

Co-PC can catalyze CO₂ reduction to CO. Once CO is produced, it must desorb from the Co center for the catalyst to turn over. The Co-PC-CO intermediate (CO bound axially to Co) can be a resting state.  
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

---

## HW 3 — Water O–H Bond Scan: Deriving the O–H Force Constant

**Objective**: Perform a constrained O–H bond length scan in water and extract the harmonic force constant from the resulting potential energy curve.

### Background

Molecular mechanics force fields represent covalent bonds as harmonic springs:  
V(r) = ½ k (r − r₀)²  
where k is the force constant (units: kcal mol⁻¹ Å⁻²) and r₀ is the equilibrium bond length. DFT calculations provide a first-principles reference for these parameters by scanning the bond potential energy surface.

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

---

## HW 4 — n-Butane C–C–C–C Dihedral Scan and Force-Field Comparison

**Objective**: Scan the central C–C–C–C dihedral angle in n-butane using DFT and compare the resulting torsion profile to CHARMM and OPLS force-field predictions.

### Background

Torsion (dihedral) potentials govern the conformational flexibility of hydrocarbon chains and are a key component of biomolecular force fields. The n-butane C–C–C–C dihedral has three minima (anti and two gauche) separated by eclipsed barriers.

### Tasks

1. **DFT Dihedral Scan**  
   - Optimize n-butane at B3LYP/6-31G* to obtain the reference geometry.  
   - Scan the central C1–C2–C3–C4 dihedral from 0° to 360° in steps of 10° (or 15°).  
   - At each step, constrain the dihedral and relax all other coordinates.  
   - Plot ΔE (kcal/mol) vs. dihedral angle φ.

2. **Conformational Energies**  
   
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

---

## Looking Ahead: From Single Molecules to Complex Systems

The homework problems so far have focused on isolated molecules and small finite fragments — a natural starting point for applying quantum chemistry and DFT. In coming weeks, we will scale up dramatically: instead of tens or hundreds of atoms treated quantum mechanically, we will simulate systems with **thousands to millions of atoms**, such as solvated biomolecules, polymers, liquid-phase reactions, and materials interfaces.

This leap in scale is made possible by **interatomic potentials**, also known as **force fields**. Rather than solving the electronic Schrödinger equation at every step, force fields represent the potential energy surface of a system using simple, computationally cheap analytical functions — harmonic bonds and angles, torsion potentials, van der Waals interactions, and electrostatics. You have already encountered this idea directly: in HW 3 you derived an O–H harmonic force constant from a DFT bond scan, and in HW 4 you compared DFT torsion profiles to the CHARMM and OPLS force fields. Those exercises were not merely academic — they mirror the actual workflow used to *develop* and *validate* force fields.




The parameters that populate these force fields (bond force constants, equilibrium geometries, partial charges, torsion barriers, and more) are overwhelmingly derived from **high-quality ab initio and DFT reference data**. Widely used force fields such as CHARMM, OPLS-AA, AMBER, and GAFF are systematically parameterized against quantum chemical calculations of exactly the kind you have been performing. Machine-learned interatomic potentials (MLIPs) take this further, training neural networks or Gaussian process models directly on large DFT datasets to achieve near-quantum accuracy at a fraction of the cost.

Understanding where force-field parameters come from, and their limitations, is essential for any researcher who uses molecular simulation. The ab initio and DFT skills developed in this course are therefore not just ends in themselves: they form the **quantitative foundation** upon which large-scale atomistic simulation is built.
