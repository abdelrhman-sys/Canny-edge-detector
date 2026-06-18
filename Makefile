# Compilers
HOST_CXX = g++
RV_CXX = riscv64-unknown-elf-g++

# Compiler Flags
HOST_FLAGS = -O3 -std=c++17 -I$(HOME)/local_gtest/include -L$(HOME)/local_gtest/lib -lgtest -lgtest_main -pthread
RV_FLAGS = -O3 -std=c++17 -march=rv64gcv -mabi=lp64d

# Targets
all: test canny_rv

# 1. Compile host-side GoogleTests
test:
	$(HOST_CXX) host_tests.cpp main_rv.cpp -DTESTING -o host_tests $(HOST_FLAGS)
	./host_tests

# 2. Cross-compile for RISC-V
canny_rv:
	$(RV_CXX) main_rv.cpp -o canny_rv $(RV_FLAGS)

# 3. Execute RISC-V binary on QEMU
run: canny_rv
	qemu-riscv64 -cpu rv64,v=true,vlen=256 ./canny_rv < test_input.raw > test_output.raw

# 4. Cross-compile QEMU assert test
qemu_assert_test:
	$(RV_CXX) qemu_assert_test.cpp main_rv.cpp -DTESTING -o qemu_assert_test $(RV_FLAGS)

# 5. Run QEMU assert test at VLEN 128 / 256 / 512
test_qemu: qemu_assert_test
	@echo "--- VLEN=128 ---"
	qemu-riscv64 -cpu rv64,v=true,vlen=128 ./qemu_assert_test
	@echo "--- VLEN=256 ---"
	qemu-riscv64 -cpu rv64,v=true,vlen=256 ./qemu_assert_test
	@echo "--- VLEN=512 ---"
	qemu-riscv64 -cpu rv64,v=true,vlen=512 ./qemu_assert_test

# 6. Clean up
clean:
	rm -f host_tests canny_* qemu_assert_test *.o

# Base RISC-V flags
RV_BASE_FLAGS = -std=c++17 -march=rv64gcv -mabi=lp64d -static -Wall -Wextra -std=c++17


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

# Auto-vectorization analysis target
canny_autovec:
	$(RV_CXX) -O3 -ftree-vectorize -fopt-info-vec-all $(RV_BASE_FLAGS) main_rv.cpp -o canny_autovec