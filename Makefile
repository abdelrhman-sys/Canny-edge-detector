# Compilers
HOST_CXX = g++
RV_CXX = riscv64-unknown-elf-g++

# Compiler Flags
HOST_FLAGS = -O3 -std=c++17 -I$(HOME)/local_gtest/include -L$(HOME)/local_gtest/lib -lgtest -lgtest_main -pthread
RV_FLAGS = -O3 -std=c++17 -march=rv64gcv -mabi=lp64d

# Source Files (You will add your Canny files here later)
# SRC = canny.cpp main.cpp

# Targets
all: test canny_rv

# 1. Compile host-side GoogleTests
test:
	$(HOST_CXX) host_tests.cpp -o host_tests $(HOST_FLAGS)
	./host_tests

# 2. Cross-compile for RISC-V
canny_rv:
	$(RV_CXX) main_rv.cpp -o canny_rv $(RV_FLAGS)

# 3. Execute RISC-V binary on QEMU
run: canny_rv
	qemu-riscv64 -cpu rv64,v=true,vlen=256 ./canny_rv < test_input.raw > test_output.raw

# 4. Clean up generated files
clean:
	rm -f host_tests canny_rv *.o
