# iDMRG.jl

This repository contains a lightly cleaned-up implementation of the iDMRG (infinite DMRG) routines used in [arXiv:2503.22792]. It was written from scratch to meet the specific requirements of that project, while adopting many ideas and best practices from the ITensor and TenPy ecosystems. Any remaining mistakes are mine.

Features:
1. Draw figures for Kitaev model

Future plans of expanding the codebase:
1. Add QN-conservation
2. Support fermion.

Comments and feedbacks are welcomed. And if you want to use the code in your own work or have spotted any bugs, please contact me at yangxusolidstate@gmail.com.

### Features

- ✅ Two-site iDMRG algorithm
- ✅ iMPO generator for standard lattice models
- ✅ Minimal, hackable design for prototyping
