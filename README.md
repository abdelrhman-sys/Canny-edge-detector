# Bare-Metal RISC-V Canny Edge Detector

A C++ implementation of the Canny Edge Detection algorithm designed to run on a bare-metal RISC-V environment using QEMU. Because this runs without a standard OS file system, image data is piped directly into the emulator's memory using Standard I/O.

## Current Progress
- [x] Phase 1: Bare-Metal Environment Setup & Makefile
- [x] Phase 2.1: Raw Image I/O (via `stdin` and `stdout`)
- [x] Phase 2.2: Gaussian Blur Kernel
- [x] Phase 2.3: Sobel Filter (Gradients & Magnitude)
- [ ] Phase 2.4: Non-Maximum Suppression (Thinning)
- [ ] Phase 2.5: Hysteresis Thresholding

## How to View Results
Since we are working with raw binary data, use the provided Python script to convert the output to a viewable PNG:

1. **Run the pipeline:** `make run`
2. **Convert the output:** `python3 view_edges.py`
3. **Check the result:** Open `edges.png`

## Prerequisites
To build and run this project, you need the following installed on your Ubuntu/WSL system:
* `riscv64-unknown-elf` toolchain (Bare-metal C++ compiler)
* `qemu-riscv64` (User-mode RISC-V emulator)
* `python3` and `numpy` (To generate the initial test images)

## Quick Start

**1. Generate the test image:**
```bash
python3 make_test_image.py
make clean
make run
