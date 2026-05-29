**Ch121a | Module 3: Periodic DFT**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ppt-2/Ch121a-DFT/blob/main/Module3_Periodic-DFT/notebooks/06_applications_of_pwdft.ipynb)

# Notebook 6: Applications of Plane-wave DFT

---


## Learning Objectives

- Understand where plane-wave periodic DFT is used in modern materials research
- Connect core observables (energies, barriers, bands, spectra) to real applications
- Recognize typical model choices and limitations for different application domains


## 1. Catalysis and Surface Chemistry

Plane-wave DFT is central to **heterogeneous catalysis** because catalysts are typically solids (metals, oxides, carbides) and reactions occur at surfaces.

Common quantities:

- Adsorption energies of intermediates (\(*CO, *OH, *H, *O\))
- Reaction barriers from NEB calculations
- Surface stability vs. composition, coverage, and environment
- Electronic descriptors (d-band center, charge transfer, projected DOS)

Representative use-cases:

- CO oxidation on Pt-group metals
- Oxygen evolution/reduction on transition-metal oxides
- CO₂ reduction on Cu-based catalysts

Practical model choices:

- Slab models with vacuum spacing and dipole corrections when needed
- Dense in-plane k-point sampling for metallic catalysts
- Spin-polarization and +U/hybrid corrections for correlated oxides


## 2. Semiconductors and Electronic Materials

For semiconductors, plane-wave DFT is widely used to predict and interpret:

- Equilibrium structure and elastic response
- Band structures, effective masses, and density of states
- Defect formation energies and charge transition levels
- Dopant behavior and diffusion pathways

Examples:

- Si, GaAs, and SiC for electronics
- TiO₂ and ZnO for photocatalysis/optoelectronics
- Perovskite oxides for functional devices

Important caveat: semi-local functionals (LDA/GGA) often underestimate band gaps, so hybrid functionals, GW, or scissor corrections may be required for quantitative electronic properties.


## 3. Light–Matter Interactions

Ground-state plane-wave DFT provides the starting point for studying optical and photo-driven behavior:

- Dielectric response and optical spectra
- Band alignment for photoelectrodes and interfaces
- Excited-state workflows built on DFT (e.g., TDDFT/BSE in compatible codes)

Typical materials contexts:

- Photocatalysts (TiO₂, hematite, oxynitrides)
- 2D semiconductors (MoS₂, WS₂) and van der Waals heterostructures
- Plasmonic metals and metal/semiconductor interfaces

Because excitons and many-body effects can be strong, beyond-DFT methods are often needed for quantitatively accurate absorption onsets and excitations.


## 4. Batteries and Electrochemical Energy Storage

Plane-wave DFT is a core tool in battery materials discovery and mechanism analysis.

What is commonly computed:

- Lithiation/sodiation voltages from total-energy differences
- Migration barriers (Li⁺/Na⁺ diffusion) using NEB
- Phase stability and decomposition tendencies
- Surface/interface reactivity (electrode–electrolyte, SEI formation)

Typical systems:

- Layered oxides, olivines, spinels, conversion materials
- Solid electrolytes and coating materials
- Anode materials (graphite, Si, alloy and conversion chemistries)

For transition-metal redox systems, DFT+U or hybrid functionals are frequently required to obtain realistic voltage trends.


## 5. Why plane-wave DFT remains foundational

Across catalysis, semiconductors, photonics, and batteries, plane-wave DFT is dominant because it offers:

- A transferable framework for periodic systems (bulk, surfaces, interfaces)
- Reproducible convergence control (ENCUT, k-mesh, slab size)
- Compatibility with robust workflows (geometry optimization, NEB, DOS/bands, AIMD)
- Tight integration with high-throughput databases and materials screening pipelines

In practice, plane-wave DFT is often the **first-principles baseline**: detailed predictions may require +U, hybrids, GW, or explicit finite-temperature/electrochemical treatments, but those workflows typically start from a converged PW-DFT model.


## 6. Further Reading

- Nørskov et al., *J. Catal.* **209**, 275 (2002) — trends in heterogeneous catalysis
- Jain et al., *APL Mater.* **1**, 011002 (2013) — Materials Project and high-throughput DFT
- Ceder & Persson, MRS Bulletin **35**, 693 (2010) — battery materials from first principles
- Onida, Reining, Rubio, *Rev. Mod. Phys.* **74**, 601 (2002) — electronic excitations in solids

---
*Ch121a | Caltech | Module 3 — Notebook 6 of 6*

