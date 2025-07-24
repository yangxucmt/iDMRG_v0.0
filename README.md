This is the iDMRG code I used in my work . I basically build everything from scratch to suit my specific needs. The code borrowed significant insights and practices from ITensors and TenPy.

There are still many features I want to add to my iDMRG code:
1. Implement the quantum number conservation
2. Design the API of MPO building
3. Demo of a 2d TFIM
4. Generate an iMPS by gauging an existing MPS



Implementing the quantum number conservation still involves many testing:
1. Modify the iDMRG sweep code without noise (-Apr 21)
2. Modify the iDMRG sweep code with noise  (-Apr 24)
3. Test my MPO generation (-Apr 26)
4. Test iDMRG initial state (all the "dag()" places)


Comments and feedbacks are welcomed. And if you want to use the code in your own work, please contact me.

# iDMRG.jl

*A modern, Julia-native implementation of infinite DMRG (iDMRG) with support for symmetries, iMPO generation, and high performance.*

## Overview

`iDMRG.jl` is a lightweight and modular Julia package for performing infinite-system density matrix renormalization group (iDMRG) calculations on 1D quantum lattice models. It is designed with a clean API, research usability, and extensibility in mind.

### Features

- ✅ Two-site iDMRG algorithm
- ✅ iMPO generator for standard lattice models (e.g., XXZ, Hubbard)
- ✅ U(1) symmetry support (block-sparse tensors)
- ✅ Basic parallelization for observables and MPO building
- ✅ Minimal, hackable design for prototyping and teaching

## Installation

Clone the repository and activate it in your Julia environment:

```bash
git clone https://github.com/yourusername/idmrg.jl
cd idmrg.jl
julia --project
] instantiate


# iDMRG.jl v0.1 Release Plan

## Overall Goal

A clean, documented, and tested Julia package implementing:

- Two-site iDMRG
- iMPO generator
- Basic symmetry support (e.g., U(1))
- Working examples (e.g., XXZ chain)
- Lightweight test suite
- Usable interface for researchers

---

## Weekly Timeline (3-4 Weeks)

### Week 1: Finalize Code and Tests

**Goals:**

- Clean up and freeze main codebase
- Test symmetry support
- Ensure iMPO generation covers key models

**To-Do:**

- Refactor public APIs (`run_dmrg`, `build_impo`, etc.)
- Choose one U(1)-symmetric model (e.g., XXZ, Hubbard)
- Add unit tests for block sparsity and correctness
- Compare performance with and without symmetry
- Add basic logging/timing utilities

---

### Week 2: Example Scripts + User Interface

**Goals:**

- Create runnable examples
- Simplify user interaction

**To-Do:**

- Add `examples/run_xxz.jl`
- (Optional) Add `examples/run_hubbard.jl`
- Allow options like `use_symmetry = true`
- Save observables such as energy, entanglement entropy, etc.

---

### Week 3: Documentation + Testing Framework

**Goals:**

- Prepare documentation and installability
- Set up test framework

**To-Do:**

- Write README with installation, usage, features
- Add docstrings for all public functions and types
- Create `Project.toml`
- Write `test/test_dmrg.jl`
- (Optional) Set up GitHub Actions CI

---

### Week 4: Polish and Tag v0.1

**Goals:**

- Final cleanup and public release

**To-Do:**

- Tag release `v0.1.0`
- Write release notes in `CHANGELOG.md`
- Archive example run with convergence logs
- (Optional) Publish to JuliaHub/JuliaRegistry
- Share release publicly (Discourse, Slack, Twitter)

---

## Suggested Directory Structure

idmrg/
├── src/
│ ├── IDMPS.jl
│ ├── DMRG.jl
│ ├── MPOGenerator.jl
│ ├── Symmetry.jl
│ └── idmrg.jl
├── examples/
│ ├── run_xxz.jl
│ └── run_hubbard.jl
├── test/
│ └── runtests.jl
├── README.md
├── Project.toml
└── LICENSE
