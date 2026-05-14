#include <iostream>
#include <riscv_vector.h>

int main() {
    // 1. Setup some basic arrays
    int32_t a[] = {10, 20, 30, 40};
    int32_t b[] = {5,  5,  5,  5};
    int32_t c[4] = {0};

    // 2. Request a vector length for 4 elements of 32-bit integers at LMUL=1
    size_t vl = __riscv_vsetvl_e32m1(4);

    // 3. Load arrays 'a' and 'b' into vector registers
    vint32m1_t vec_a = __riscv_vle32_v_i32m1(a, vl);
    vint32m1_t vec_b = __riscv_vle32_v_i32m1(b, vl);

    // 4. Add the two vectors together
    vint32m1_t vec_c = __riscv_vadd_vv_i32m1(vec_a, vec_b, vl);

    // 5. Store the result back into array 'c'
    __riscv_vse32_v_i32m1(c, vec_c, vl);

    // 6. Print the results
    std::cout << "Vector Length used: " << vl << std::endl;
    std::cout << "Result: ";
    for (int i = 0; i < 4; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
