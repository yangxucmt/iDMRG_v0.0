This is the iDMRG code I used in my work. I basically build everything from scratch to suit my specific needs. The code borrowed significant insights and practices from ITensors and TenPy.

Next steps
1. Add QN-conservation
2. Support fermion.
3. Add single-site iDMRG.
4. Gauge a MPS.


Comments and feedbacks are welcomed. And if you want to use the code in your own work, please contact me.

# iDMRG.jl

*A modern, Julia-native implementation of infinite DMRG (iDMRG) with support for symmetries, iMPO generation, and high performance.*

## Overview

`iDMRG.jl` is a lightweight and modular Julia package for performing infinite-system density matrix renormalization group (iDMRG) calculations on 1D quantum lattice models. It is designed with a clean API, research usability, and extensibility in mind.

### Features

- ✅ Two-site iDMRG algorithm
- ✅ iMPO generator for standard lattice models (e.g., XXZ, Hubbard)
- ✅ Minimal, hackable design for prototyping and teaching

## Installation

Clone the repository and activate it in your Julia environment:

```bash
git clone https://github.com/yourusername/idmrg.jl
cd idmrg.jl
julia --project
] instantiate


# iDMRG.jl v0.0 Release Plan

## Overall Goal

A clean, documented, and tested Julia package implementing:

- Two-site iDMRG
- iMPO generator
- Basic symmetry support (e.g., U(1))
- Working examples (e.g., XXZ chain)
- Lightweight test suite
- Usable interface for researchers

