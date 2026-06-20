#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <time.h>
#include <iomanip>

// Only include the vector header if we are compiling for RISC-V
#ifdef __riscv
#include <riscv_vector.h>
#endif

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


// Phase 6.2: Vectorized 5x5 Gaussian Blur (RVV)
#ifdef __riscv
void gaussian_blur_rvv(const uint8_t* input, uint8_t* output, int width, int height) {
    const int16_t kernel[25] = {
        1,  4,  7,  4, 1,
        4, 16, 26, 16, 4,
        7, 26, 41, 26, 7,
        4, 16, 26, 16, 4,
        1,  4,  7,  4, 1
    };

    for (int y = 2; y < height - 2; y++) {
        int x = 2;
        int pixels_left = width - 4; 

        while (pixels_left > 0) {
            size_t vl = __riscv_vsetvl_e32m8(pixels_left);
            vint32m8_t v_sum = __riscv_vmv_v_x_i32m8(0, vl);

            int k_idx = 0;
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    const uint8_t* pixel_ptr = &input[(y + ky) * width + (x + kx)];
                    
                    vuint8m2_t v_pixels = __riscv_vle8_v_u8m2(pixel_ptr, vl);
                    vuint16m4_t v_pixels_16 = __riscv_vwaddu_vx_u16m4(v_pixels, 0, vl);
                    vuint32m8_t v_pixels_32 = __riscv_vwaddu_vx_u32m8(v_pixels_16, 0, vl);
                    
                    int32_t weight = kernel[k_idx++];
                    v_sum = __riscv_vmacc_vx_i32m8(v_sum, weight, __riscv_vreinterpret_v_u32m8_i32m8(v_pixels_32), vl);
                }
            }

            v_sum = __riscv_vmul_vx_i32m8(v_sum, 3841, vl);
            v_sum = __riscv_vsra_vx_i32m8(v_sum, 20, vl);

            v_sum = __riscv_vmax_vx_i32m8(v_sum, 0, vl);
            v_sum = __riscv_vmin_vx_i32m8(v_sum, 255, vl);

            vuint16m4_t v_out_16 = __riscv_vncvt_x_x_w_u16m4(__riscv_vreinterpret_v_i32m8_u32m8(v_sum), vl);
            vuint8m2_t v_out_8   = __riscv_vncvt_x_x_w_u8m2(v_out_16, vl);

            __riscv_vse8_v_u8m2(&output[y * width + x], v_out_8, vl);

            x += vl;
            pixels_left -= vl;
        }
    }
}
#endif


// Phase 6.3: Vectorized Sobel Gradients (RVV)
#ifdef __riscv
void sobel_gradients_rvv(const uint8_t* input, int16_t* gx_out, int16_t* gy_out, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        int x = 1;
        int pixels_left = width - 2;

        while (pixels_left > 0) {
            size_t vl = __riscv_vsetvl_e16m4(pixels_left);

            const uint8_t* row_t = &input[(y - 1) * width + x];
            const uint8_t* row_m = &input[y * width + x];
            const uint8_t* row_b = &input[(y + 1) * width + x];

            vuint8m2_t t0_8 = __riscv_vle8_v_u8m2(row_t - 1, vl);
            vuint8m2_t m0_8 = __riscv_vle8_v_u8m2(row_m - 1, vl);
            vuint8m2_t b0_8 = __riscv_vle8_v_u8m2(row_b - 1, vl);

            vuint8m2_t t1_8 = __riscv_vle8_v_u8m2(row_t, vl);
            vuint8m2_t b1_8 = __riscv_vle8_v_u8m2(row_b, vl);

            vuint8m2_t t2_8 = __riscv_vle8_v_u8m2(row_t + 1, vl);
            vuint8m2_t m2_8 = __riscv_vle8_v_u8m2(row_m + 1, vl);
            vuint8m2_t b2_8 = __riscv_vle8_v_u8m2(row_b + 1, vl);

            #define WIDEN(v) __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vwaddu_vx_u16m4(v, 0, vl))
            
            vint16m4_t t0 = WIDEN(t0_8); vint16m4_t m0 = WIDEN(m0_8); vint16m4_t b0 = WIDEN(b0_8);
            vint16m4_t t1 = WIDEN(t1_8);                              vint16m4_t b1 = WIDEN(b1_8);
            vint16m4_t t2 = WIDEN(t2_8); vint16m4_t m2 = WIDEN(m2_8); vint16m4_t b2 = WIDEN(b2_8);

            vint16m4_t sum_left = __riscv_vadd_vv_i16m4(t0, b0, vl);
            vint16m4_t m0_2     = __riscv_vsll_vx_i16m4(m0, 1, vl); 
            sum_left            = __riscv_vadd_vv_i16m4(sum_left, m0_2, vl);

            vint16m4_t sum_right = __riscv_vadd_vv_i16m4(t2, b2, vl);
            vint16m4_t m2_2      = __riscv_vsll_vx_i16m4(m2, 1, vl);
            sum_right            = __riscv_vadd_vv_i16m4(sum_right, m2_2, vl);

            vint16m4_t gx = __riscv_vsub_vv_i16m4(sum_right, sum_left, vl);

            vint16m4_t sum_top = __riscv_vadd_vv_i16m4(t0, t2, vl);
            vint16m4_t t1_2    = __riscv_vsll_vx_i16m4(t1, 1, vl);
            sum_top            = __riscv_vadd_vv_i16m4(sum_top, t1_2, vl);

            vint16m4_t sum_bot = __riscv_vadd_vv_i16m4(b0, b2, vl);
            vint16m4_t b1_2    = __riscv_vsll_vx_i16m4(b1, 1, vl);
            sum_bot            = __riscv_vadd_vv_i16m4(sum_bot, b1_2, vl);

            vint16m4_t gy = __riscv_vsub_vv_i16m4(sum_top, sum_bot, vl);

            __riscv_vse16_v_i16m4(&gx_out[y * width + x], gx, vl);
            __riscv_vse16_v_i16m4(&gy_out[y * width + x], gy, vl);

            x += vl;
            pixels_left -= vl;
        }
    }
}
#endif


// Phase 6.4: Vectorized Gradient Magnitude (with RVV Reduction)
#ifdef __riscv
void compute_magnitude_rvv(const int16_t* gx, const int16_t* gy, uint8_t* output, int width, int height) {
    int num_pixels = width * height;

    int pixels_left = num_pixels;
    const int16_t* ptr_gx = gx;
    const int16_t* ptr_gy = gy;
    
    vuint16m1_t v_max_val = __riscv_vmv_s_x_u16m1(0, __riscv_vsetvlmax_e16m1()); 

    while (pixels_left > 0) {
        size_t vl = __riscv_vsetvl_e16m8(pixels_left);

        vint16m8_t v_gx = __riscv_vle16_v_i16m8(ptr_gx, vl);
        vint16m8_t v_gy = __riscv_vle16_v_i16m8(ptr_gy, vl);

        vint16m8_t v_gx_abs = __riscv_vmax_vv_i16m8(v_gx, __riscv_vneg_v_i16m8(v_gx, vl), vl);
        vint16m8_t v_gy_abs = __riscv_vmax_vv_i16m8(v_gy, __riscv_vneg_v_i16m8(v_gy, vl), vl);

        vuint16m8_t v_mag = __riscv_vreinterpret_v_i16m8_u16m8(__riscv_vadd_vv_i16m8(v_gx_abs, v_gy_abs, vl));

        v_max_val = __riscv_vredmaxu_vs_u16m8_u16m1(v_mag, v_max_val, vl);

        ptr_gx += vl;
        ptr_gy += vl;
        pixels_left -= vl;
    }

    uint16_t max_mag = __riscv_vmv_x_s_u16m1_u16(v_max_val);
    if (max_mag == 0) max_mag = 1;

    pixels_left = num_pixels;
    ptr_gx = gx;
    ptr_gy = gy;
    uint8_t* ptr_out = output;

    while (pixels_left > 0) {
        size_t vl = __riscv_vsetvl_e16m8(pixels_left);

        vint16m8_t v_gx = __riscv_vle16_v_i16m8(ptr_gx, vl);
        vint16m8_t v_gy = __riscv_vle16_v_i16m8(ptr_gy, vl);
        
        vint16m8_t v_gx_abs = __riscv_vmax_vv_i16m8(v_gx, __riscv_vneg_v_i16m8(v_gx, vl), vl);
        vint16m8_t v_gy_abs = __riscv_vmax_vv_i16m8(v_gy, __riscv_vneg_v_i16m8(v_gy, vl), vl);
        vuint16m8_t v_mag = __riscv_vreinterpret_v_i16m8_u16m8(__riscv_vadd_vv_i16m8(v_gx_abs, v_gy_abs, vl));

        vuint16m8_t v_mag_255 = __riscv_vmul_vx_u16m8(v_mag, 255, vl);
        vuint16m8_t v_norm = __riscv_vdivu_vx_u16m8(v_mag_255, max_mag, vl);

        vuint8m4_t v_out = __riscv_vncvt_x_x_w_u8m4(v_norm, vl);

        __riscv_vse8_v_u8m4(ptr_out, v_out, vl);

        ptr_gx += vl;
        ptr_gy += vl;
        ptr_out += vl;
        pixels_left -= vl;
    }
}
#endif


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

// Phase 2.4: Gradient Magnitude (Two-Pass L1 Norm)
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
        output[i] = dir * 50; 
    }
}

#ifndef TESTING

// Phase 5: Bare-Metal Cycle Profiler (Only compiles for RISC-V)
#ifdef __riscv
static inline uint64_t get_cycles() {
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}
#endif

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

    if (fread(img_in, 1, num_pixels, stdin) != (size_t)num_pixels) {
        std::cerr << "Failed to read image." << std::endl;
        return 1;
    }

    std::cerr << "\n========================================\n";
    std::cerr << " PHASE 4: SCALAR BASELINE TIMINGS (ms)\n";
    std::cerr << "========================================\n";
    
    auto start_time = std::chrono::steady_clock::now();
    for(int i = 0; i < 100; i++) gaussian_blur(img_in, img_blur, width, height);
    auto end_time = std::chrono::steady_clock::now();
    std::cerr << "Gaussian 5x5: " << std::chrono::duration<double, std::milli>(end_time - start_time).count() / 100.0 << " ms\n";

    start_time = std::chrono::steady_clock::now();
    for(int i = 0; i < 100; i++) sobel_gradients(img_blur, img_gx, img_gy, width, height);
    end_time = std::chrono::steady_clock::now();
    std::cerr << "Sobel Gx/Gy:  " << std::chrono::duration<double, std::milli>(end_time - start_time).count() / 100.0 << " ms\n";

    start_time = std::chrono::steady_clock::now();
    for(int i = 0; i < 100; i++) compute_magnitude(img_gx, img_gy, img_mag, width, height);
    end_time = std::chrono::steady_clock::now();
    std::cerr << "Magnitude:    " << std::chrono::duration<double, std::milli>(end_time - start_time).count() / 100.0 << " ms\n";

    start_time = std::chrono::steady_clock::now();
    for(int i = 0; i < 100; i++) compute_direction(img_gx, img_gy, img_dir, width, height);
    end_time = std::chrono::steady_clock::now();
    std::cerr << "Direction:    " << std::chrono::duration<double, std::milli>(end_time - start_time).count() / 100.0 << " ms\n";


#ifdef __riscv
    std::cerr << "\n========================================\n";
    std::cerr << " PHASE 6: RVV OPTIMIZED TIMINGS (ms)\n";
    std::cerr << "========================================\n";

    start_time = std::chrono::steady_clock::now();
    for(int i = 0; i < 100; i++) gaussian_blur_rvv(img_in, img_blur, width, height);
    end_time = std::chrono::steady_clock::now();
    std::cerr << "Gaussian 5x5: " << std::chrono::duration<double, std::milli>(end_time - start_time).count() / 100.0 << " ms\n";

    start_time = std::chrono::steady_clock::now();
    for(int i = 0; i < 100; i++) sobel_gradients_rvv(img_blur, img_gx, img_gy, width, height);
    end_time = std::chrono::steady_clock::now();
    std::cerr << "Sobel Gx/Gy:  " << std::chrono::duration<double, std::milli>(end_time - start_time).count() / 100.0 << " ms\n";

    start_time = std::chrono::steady_clock::now();
    for(int i = 0; i < 100; i++) compute_magnitude_rvv(img_gx, img_gy, img_mag, width, height);
    end_time = std::chrono::steady_clock::now();
    std::cerr << "Magnitude:    " << std::chrono::duration<double, std::milli>(end_time - start_time).count() / 100.0 << " ms\n";

    // Note: Direction remains scalar as per profiling requirements
    std::cerr << "Direction:    Scalar fallback used.\n";

    std::cerr << "\n========================================\n";
    std::cerr << " PHASE 5: HARDWARE CYCLE PROFILING\n";
    std::cerr << "========================================\n";
    
    int iterations = 100;
    uint64_t cycles_gaussian = 0, cycles_sobel = 0, cycles_mag = 0, cycles_dir = 0;
    uint64_t start, end;

    for (int i = 0; i < iterations; i++) {
        start = get_cycles();
        gaussian_blur_rvv(img_in, img_blur, width, height);
        cycles_gaussian += (get_cycles() - start);

        start = get_cycles();
        sobel_gradients_rvv(img_blur, img_gx, img_gy, width, height);
        cycles_sobel += (get_cycles() - start);

        start = get_cycles();
        compute_magnitude_rvv(img_gx, img_gy, img_mag, width, height);
        cycles_mag += (get_cycles() - start);

        start = get_cycles();
        compute_direction(img_gx, img_gy, img_dir, width, height);
        cycles_dir += (get_cycles() - start);
    }

    double total_cycles = cycles_gaussian + cycles_sobel + cycles_mag + cycles_dir;
    std::cerr << std::fixed << std::setprecision(2);
    std::cerr << "Gaussian:   " << (cycles_gaussian / total_cycles) * 100.0 << "% (" << cycles_gaussian << " cycles)\n";
    std::cerr << "Sobel:      " << (cycles_sobel / total_cycles) * 100.0    << "% (" << cycles_sobel << " cycles)\n";
    std::cerr << "Magnitude:  " << (cycles_mag / total_cycles) * 100.0      << "% (" << cycles_mag << " cycles)\n";
    std::cerr << "Direction:  " << (cycles_dir / total_cycles) * 100.0      << "% (" << cycles_dir << " cycles)\n";
    std::cerr << "----------------------------------------\n";
#endif

    // Output final magnitude for visual verification (goes to test_output.raw)
    fwrite(img_mag, 1, num_pixels, stdout);

    free(img_in); free(img_blur); free(img_gx); free(img_gy); free(img_mag); free(img_dir);
    return 0;
}
#endif