#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <time.h>
#include <cmath>
#include <riscv_vector.h>

// Phase 2.2: 5x5 Gaussian Blur (Sum = 273)
void gaussian_blur(const uint8_t* input, uint8_t* output, int width, int height) {
    const int16_t kernel[5][5] = {
        {1,  4,  7,  4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1,  4,  7,  4, 1}
    };

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int32_t sum = 0; // Prevent overflow during accumulation
            
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int px = x + kx;
                    int py = y + ky;
                    
                    // Zero-padding boundary condition
                    int16_t val = 0;
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        val = input[py * width + px];
                    }
                    sum += val * kernel[ky + 2][kx + 2];
                }
            }
            
            sum /= 273; // Normalize by the sum of the kernel
            if (sum > 255) sum = 255;
            if (sum < 0) sum = 0;
            output[y * width + x] = (uint8_t)sum;
        }
    }
}

// Phase 2.3: Sobel Gradients (Structure of Arrays)
void sobel_gradients(const uint8_t* input, int16_t* gx_out, int16_t* gy_out, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int gx = 0, gy = 0;
            gx += input[(y - 1) * width + (x - 1)] * -1; gx += input[(y - 1) * width + (x + 1)] * 1;
            gx += input[y * width + (x - 1)]       * -2; gx += input[y * width + (x + 1)]       * 2;
            gx += input[(y + 1) * width + (x - 1)] * -1; gx += input[(y + 1) * width + (x + 1)] * 1;

            gy += input[(y - 1) * width + (x - 1)] * 1; gy += input[(y - 1) * width + x] * 2; gy += input[(y - 1) * width + (x + 1)] * 1;
            gy += input[(y + 1) * width + (x - 1)] *-1; gy += input[(y + 1) * width + x] *-2; gy += input[(y + 1) * width + (x + 1)] *-1;

            gx_out[y * width + x] = (int16_t)gx;
            gy_out[y * width + x] = (int16_t)gy;
        }
    }
}

// Phase 2.4: Gradient Magnitude (Two-Pass L1 Norm) - SCALAR VERSION (Kept for tests)
void compute_magnitude(const int16_t* gx, const int16_t* gy, uint8_t* output, int width, int height) {
    int max_mag = 0;
    for (int i = 0; i < width * height; i++) {
        int mag = abs(gx[i]) + abs(gy[i]);
        if (mag > max_mag) max_mag = mag;
    }
    if (max_mag == 0) max_mag = 1;
    for (int i = 0; i < width * height; i++) {
        int mag = abs(gx[i]) + abs(gy[i]);
        output[i] = (uint8_t)((mag * 255) / max_mag);
    }
}

// Phase 6: Vectorized Gradient Magnitude - FAST RVV VERSION
void compute_magnitude_rvv(const int16_t* gx, const int16_t* gy, uint8_t* output, int width, int height) {
    int num_pixels = width * height;
    
    // Ask hardware for max vector length for 16-bit elements
    size_t max_vl = __riscv_vsetvlmax_e16m1();
    vint16m1_t v_max_accum = __riscv_vmv_v_x_i16m1(0, max_vl); 

    // --- PASS 1: RVV Strip-Mining to find max magnitude ---
    int i = 0;
    while (i < num_pixels) {
        size_t vl = __riscv_vsetvl_e16m1(num_pixels - i);
        
        vint16m1_t v_gx = __riscv_vle16_v_i16m1(&gx[i], vl);
        vint16m1_t v_gy = __riscv_vle16_v_i16m1(&gy[i], vl);

        // Absolute value hack: max(val, -val)
        vint16m1_t v_neg_gx = __riscv_vrsub_vx_i16m1(v_gx, 0, vl); 
        vint16m1_t v_abs_gx = __riscv_vmax_vv_i16m1(v_gx, v_neg_gx, vl);

        vint16m1_t v_neg_gy = __riscv_vrsub_vx_i16m1(v_gy, 0, vl);
        vint16m1_t v_abs_gy = __riscv_vmax_vv_i16m1(v_gy, v_neg_gy, vl);

        // Add them together: mag = abs(gx) + abs(gy)
        vint16m1_t v_mag = __riscv_vadd_vv_i16m1(v_abs_gx, v_abs_gy, vl);
        
        // Find the maximum value in the vector
        v_max_accum = __riscv_vredmax_vs_i16m1_i16m1(v_mag, v_max_accum, vl);
        
        i += vl;
    }

    // Extract the final maximum value to standard scalar variable
    int16_t max_mag = __riscv_vmv_x_s_i16m1_i16(v_max_accum);
    if (max_mag == 0) max_mag = 1;

    // --- PASS 2: Scalar Normalization ---
    for (int j = 0; j < num_pixels; j++) {
        int mag = std::abs(gx[j]) + std::abs(gy[j]);
        output[j] = (uint8_t)((mag * 255) / max_mag);
    }
}

// Phase 2.5: Gradient Direction (Integer Cross-Multiplication)
void compute_direction(const int16_t* gx, const int16_t* gy, uint8_t* output, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        int ax = abs(gx[i]), ay = abs(gy[i]);
        uint8_t dir = 0;
        if (ay * 5 < ax * 2) dir = 0;
        else if (ay * 5 > ax * 12) dir = 2;
        else {
            if ((gx[i] > 0 && gy[i] > 0) || (gx[i] < 0 && gy[i] < 0)) dir = 1;
            else dir = 3;
        }
        output[i] = dir * 50; // Scaled for visual debugging
    }
}

#ifndef TESTING
int main() {
    int width = 256, height = 256;
    int num_pixels = width * height;

    uint8_t* img_in = (uint8_t*)aligned_alloc(64, num_pixels);
    uint8_t* img_blur = (uint8_t*)aligned_alloc(64, num_pixels);
    int16_t* img_gx = (int16_t*)aligned_alloc(64, num_pixels * sizeof(int16_t));
    int16_t* img_gy = (int16_t*)aligned_alloc(64, num_pixels * sizeof(int16_t));
    uint8_t* img_mag = (uint8_t*)aligned_alloc(64, num_pixels);
    uint8_t* img_dir = (uint8_t*)aligned_alloc(64, num_pixels);

    memset(img_gx, 0, num_pixels * sizeof(int16_t));
    memset(img_gy, 0, num_pixels * sizeof(int16_t));

    if (fread(img_in, 1, num_pixels, stdin) == (size_t)num_pixels) {
        
        // Phase 2 Pipeline
        gaussian_blur(img_in, img_blur, width, height);
        sobel_gradients(img_blur, img_gx, img_gy, width, height);
        
        // HERE: Call the new blazing-fast RISC-V Vectorized magnitude function!
        compute_magnitude_rvv(img_gx, img_gy, img_mag, width, height);
        
        compute_direction(img_gx, img_gy, img_dir, width, height);
        
        // Output magnitude for visual verification (goes to test_output.raw)
        fwrite(img_mag, 1, num_pixels, stdout);
        
    } else {
        std::cerr << "Failed to read image." << std::endl;
    }

    // --- MEASURE TIME BEFORE FREEING MEMORY ---
    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < 100; i++) {
        gaussian_blur(img_in, img_blur, width, height);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    double time_taken = diff.count();
    
    // Print using cerr so it shows on the terminal screen
    std::cerr << "Gaussian 5x5 average time per run: " << time_taken / 100.0 << " seconds" << std::endl;

    // --- FREE MEMORY AT THE VERY END ---
    free(img_in); free(img_blur); free(img_gx); free(img_gy); free(img_mag); free(img_dir);

    return 0;
}
#endif