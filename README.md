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
- [x] Phase 3: Testing (Host and QEMU-Side)
  - [x] 3.1: GoogleTest Unit Tests (Gaussian, Sobel, Magnitude) — 6/6 passing
  - [x] 3.2: QEMU Assert Tests at VLEN=128/256/512 — 6/6 passing
- [ ] Phase 4: Compiler Optimization Sweep
- [ ] Phase 5: Profiling
- [ ] Phase 6: RVV Intrinsic Optimization

## How to View Results
Since we are working with raw binary data, use the provided Python script to convert the output to a viewable PNG:

1. **Run the pipeline:** `make run`
2. **Convert the output:** `python3 view_edges.py`
3. **Check the result:** Open `edges.png`
