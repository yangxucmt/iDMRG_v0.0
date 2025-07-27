# iDMRG.jl

This repository contains a lightly cleaned-up implementation of the iDMRG (infinite DMRG) routines used in [arXiv:2503.22792]. It was written from scratch to meet the specific requirements of that project, while adopting many ideas and best practices from the ITensor and TenPy ecosystems. Any remaining mistakes are mine.

Future goals:
1. Add QN-conservation
2. Support fermion.

Comments and feedback are welcome. If you plan to build on this code in your own work or notice any issues, feel free to reach out at yangxusolidstate@gmail.com, or just open an issue or pull request here.


### Features

- ✅ Two-site iDMRG algorithm
- ✅ iMPO generator for standard lattice models
- ✅ Minimal, hackable design for prototyping
