# ==========================================
# Toolchain & Environment Settings
# ==========================================
HOST_CXX = g++
RV_CXX = riscv64-unknown-elf-g++

# ==========================================
# Compiler Flags
# ==========================================
# Host testing flags (Requires Google Test built in ~/local_gtest)
HOST_FLAGS = -O3 -std=c++17 -I$(HOME)/local_gtest/include -L$(HOME)/local_gtest/lib -lgtest -lgtest_main -pthread

# Base RISC-V flags for bare-metal Newlib (Static linking required for QEMU user-mode)
RV_BASE_FLAGS = -std=c++17 -march=rv64gcv -mabi=lp64d -static -Wall -Wextra

# Standard optimized build flags
RV_FLAGS = -O3 $(RV_BASE_FLAGS)

# ==========================================
# Primary Targets
# ==========================================
all: test canny_rv

# Phase 3: Host-side Unit Tests
test:
	$(HOST_CXX) host_tests.cpp main_rv.cpp -DTESTING -o host_tests $(HOST_FLAGS)
	./host_tests

# Phase 6: Compile the optimized RVV binary
canny_rv:
	$(RV_CXX) main_rv.cpp -o canny_rv $(RV_FLAGS)

# Execute the main RVV binary on QEMU (Default VLEN 256)
run: canny_rv
	qemu-riscv64 -cpu rv64,v=true,vlen=256 ./canny_rv < test_input.raw > test_output.raw

# ==========================================
# Phase 3: QEMU Equivalence Testing (VLEN Sweep)
# ==========================================
qemu_assert_test:
	$(RV_CXX) qemu_assert_test.cpp main_rv.cpp -DTESTING -o qemu_assert_test $(RV_FLAGS)

test_qemu: qemu_assert_test
	@echo "--- VLEN=128 ---"
	qemu-riscv64 -cpu rv64,v=true,vlen=128 ./qemu_assert_test
	@echo "--- VLEN=256 ---"
	qemu-riscv64 -cpu rv64,v=true,vlen=256 ./qemu_assert_test
	@echo "--- VLEN=512 ---"
	qemu-riscv64 -cpu rv64,v=true,vlen=512 ./qemu_assert_test

# ==========================================
# Phase 4: Compiler Optimization Sweep
# ==========================================
phase4_all: canny_O0 canny_O2 canny_O3 canny_Os canny_Ofast canny_autovec

canny_O0:
	$(RV_CXX) -O0 $(RV_BASE_FLAGS) main_rv.cpp -o canny_O0

canny_O2:
	$(RV_CXX) -O2 $(RV_BASE_FLAGS) main_rv.cpp -o canny_O2

canny_O3:
	$(RV_CXX) -O3 $(RV_BASE_FLAGS) main_rv.cpp -o canny_O3

canny_Os:
	$(RV_CXX) -Os $(RV_BASE_FLAGS) main_rv.cpp -o canny_Os

canny_Ofast:
	$(RV_CXX) -Ofast $(RV_BASE_FLAGS) main_rv.cpp -o canny_Ofast

canny_autovec:
	$(RV_CXX) -O3 -ftree-vectorize -fopt-info-vec-all $(RV_BASE_FLAGS) main_rv.cpp -o canny_autovec

# NEW TARGET: Run the full Phase 4 and Phase 6 timing sweep automatically
run_timing_sweep: phase4_all canny_rv
	@echo "\n=== SCALAR BASELINE TIMINGS ==="
	@echo "--- -O0 ---"
	@qemu-riscv64 -cpu rv64,v=true ./canny_O0 < test_input.raw > /dev/null
	@echo "--- -O2 ---"
	@qemu-riscv64 -cpu rv64,v=true ./canny_O2 < test_input.raw > /dev/null
	@echo "--- -O3 ---"
	@qemu-riscv64 -cpu rv64,v=true ./canny_O3 < test_input.raw > /dev/null
	@echo "--- Auto-Vectorized ---"
	@qemu-riscv64 -cpu rv64,v=true ./canny_autovec < test_input.raw > /dev/null
	@echo "\n=== RVV OPTIMIZED TIMINGS ==="
	@echo "--- RVV (VLEN=128) ---"
	@qemu-riscv64 -cpu rv64,v=true,vlen=128 ./canny_rv < test_input.raw > /dev/null
	@echo "--- RVV (VLEN=256) ---"
	@qemu-riscv64 -cpu rv64,v=true,vlen=256 ./canny_rv < test_input.raw > /dev/null
	@echo "\n=== BINARY SIZES ==="
	@ls -lh canny_O0 canny_O2 canny_O3 canny_autovec canny_rv | awk '{print $$9 ": " $$5}'

# ==========================================
# Utility
# ==========================================
clean:
	rm -f host_tests canny_* qemu_assert_test *.o test_output.raw

# --- NEW: full pipeline in one command (prepare -> compile -> run -> convert) ---
edges: 
	python3 convert_image.py
	$(MAKE) canny_rv
	$(MAKE) run
	python3 view_edges.py

.PHONY: edges