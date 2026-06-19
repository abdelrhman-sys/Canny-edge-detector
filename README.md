# Bare-Metal RISC-V Canny Edge Detector

A C++ implementation of a Canny Edge Detection pipeline designed to run on a bare-metal RISC-V environment using QEMU. Because this runs without a standard OS file system, image data is piped directly into the emulator's memory using Standard I/O.

## Current Progress
- [x] Phase 1: Bare-Metal Environment Setup & Makefile
- [x] Phase 2: Scalar Baseline Pipeline
  - [x] 2.1: Raw Image I/O (via `stdin`/`stdout`)
  - [x] 2.2: 5x5 Integer Gaussian Blur (Sum = 273)
  - [x] 2.3: Sobel Gradients (Structure of Arrays)
  - [x] 2.4: Gradient Magnitude (Two-Pass L1 Norm Normalization)
  - [x] 2.5: Gradient Direction (Integer Cross-Multiplication)
- [x] Phase 3: Testing (Automated Host vs QEMU-Side Equivalence Script)
- [x] Phase 4: Compiler Optimization Sweep (Bypassed: `-O3` already active)
- [ ] Phase 5: Profiling
- [x] Phase 6: RVV Intrinsic Optimization
  - [x] 6.1: Vector Strip-Mining (`__riscv_vsetvl_e*`) and dynamic LMUL selection
  - [x] 6.2: Vectorized Gaussian Blur (Fixed-point arithmetic & bit-shifting replacement for scalar division)
  - [x] 6.3: Vectorized Sobel Gradients (Data widening `u8` to `i16` and `vsll` bit-shifting)
  - [x] 6.4: Vectorized Magnitude (Hardware vector reduction via `vredmaxu`)


## How to Run & View Results
Since we are working with a bare-metal environment, profiling results are printed to the terminal, and binary image data is piped into files.

1. **Prepare the Image:** `python3 convert_image.py`
2. **Compile the Binary:** `make canny_rv`
3. **Execute via QEMU:** `make run`
4. ** Convert the Output:** `python3 view_edges.py`
5. **Check the result:** Open `edges.png`