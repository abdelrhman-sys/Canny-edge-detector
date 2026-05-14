#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

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
        output[i] = dir * 50; // Scaled for visual debugging
    }
}

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
        compute_magnitude(img_gx, img_gy, img_mag, width, height);
        compute_direction(img_gx, img_gy, img_dir, width, height);
        
        // Output magnitude for visual verification
        fwrite(img_mag, 1, num_pixels, stdout);
        
    } else {
        std::cerr << "Failed to read image." << std::endl;
    }

    free(img_in); free(img_blur); free(img_gx); free(img_gy); free(img_mag); free(img_dir);
    return 0;
}
