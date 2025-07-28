# iDMRG.jl

This repository provides a lightly refined implementation of the iDMRG (infinite DMRG) routines used in [arXiv:2503.22792]. It was developed from scratch using core ITensor functionalities to address the specific needs of that project, while drawing inspiration and best practices from both the ITensor and TenPy ecosystems. Any remaining errors are my own responsibility.

Future goals:
1. Add QN-conservation
2. Support fermion.

Comments and feedback are welcome. If you plan to build on this code in your own work or notice any issues, feel free to reach out at yangxusolidstate@gmail.com, or just open an issue or pull request here.


### Features

- ✅ Two-site iDMRG algorithm
- ✅ iMPO generator for standard lattice models
- ✅ Minimal, hackable design for prototyping
