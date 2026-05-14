#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "image_io.h"

// Phase 2.2: Gaussian Blur
void gaussian_blur(const uint8_t* input, uint8_t* output, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sum = 0;
            sum += input[(y - 1) * width + (x - 1)] * 1;
            sum += input[(y - 1) * width + x]       * 2;
            sum += input[(y - 1) * width + (x + 1)] * 1;
            sum += input[y * width + (x - 1)]       * 2;
            sum += input[y * width + x]             * 4;
            sum += input[y * width + (x + 1)]       * 2;
            sum += input[(y + 1) * width + (x - 1)] * 1;
            sum += input[(y + 1) * width + x]       * 2;
            sum += input[(y + 1) * width + (x + 1)] * 1;
            output[y * width + x] = sum / 16;
        }
    }
}

// Phase 2.3: Sobel Gradients (Structure of Arrays - 16-bit output)
void sobel_gradients(const uint8_t* input, int16_t* gx_out, int16_t* gy_out, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int gx = 0;
            int gy = 0;

            gx += input[(y - 1) * width + (x - 1)] * -1;
            gx += input[(y - 1) * width + (x + 1)] * 1;
            gx += input[y * width + (x - 1)]       * -2;
            gx += input[y * width + (x + 1)]       * 2;
            gx += input[(y + 1) * width + (x - 1)] * -1;
            gx += input[(y + 1) * width + (x + 1)] * 1;

            gy += input[(y - 1) * width + (x - 1)] * 1;
            gy += input[(y - 1) * width + x]       * 2;
            gy += input[(y - 1) * width + (x + 1)] * 1;
            gy += input[(y + 1) * width + (x - 1)] * -1;
            gy += input[(y + 1) * width + x]       * -2;
            gy += input[(y + 1) * width + (x + 1)] * -1;

            gx_out[y * width + x] = (int16_t)gx;
            gy_out[y * width + x] = (int16_t)gy;
        }
    }
}

// Phase 2.4: Gradient Magnitude (L1 Norm with Two-Pass Normalization)
void compute_magnitude(const int16_t* gx, const int16_t* gy, uint8_t* output, int width, int height) {
    int max_mag = 0;
    
    // Pass 1: Find the global maximum magnitude
    for (int i = 0; i < width * height; i++) {
        int mag = abs(gx[i]) + abs(gy[i]);
        if (mag > max_mag) {
            max_mag = mag;
        }
    }

    // Prevent divide-by-zero on completely black images
    if (max_mag == 0) max_mag = 1;

    // Pass 2: Normalize to 0-255
    for (int i = 0; i < width * height; i++) {
        int mag = abs(gx[i]) + abs(gy[i]);
        output[i] = (uint8_t)((mag * 255) / max_mag);
    }
}

// Phase 2.5: Gradient Direction (Integer Cross-Multiplication)
void compute_direction(const int16_t* gx, const int16_t* gy, uint8_t* output, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        int ax = abs(gx[i]);
        int ay = abs(gy[i]);
        uint8_t dir = 0;

        // Compare using tan(22.5) ~ 2/5 and tan(67.5) ~ 12/5
        if (ay * 5 < ax * 2) {
            dir = 0; // Horizontal edge
        } else if (ay * 5 > ax * 12) {
            dir = 2; // Vertical edge
        } else {
            // Diagonal edge. Check signs to see which diagonal.
            if ((gx[i] > 0 && gy[i] > 0) || (gx[i] < 0 && gy[i] < 0)) {
                dir = 1; // 45 degrees
            } else {
                dir = 3; // 135 degrees
            }
        }
        
        // We multiply by 50 just so the values (0, 50, 100, 150) are visibly different 
        // if we save this buffer as an image for debugging!
        output[i] = dir * 50; 
    }
}

int main() {
    int width = 256;
    int height = 256;
    int num_pixels = width * height;

    // Use aligned_alloc as required by the PDF for future RVV vectorization
    uint8_t* img_in = (uint8_t*)aligned_alloc(64, num_pixels);
    uint8_t* img_blur = (uint8_t*)aligned_alloc(64, num_pixels);
    int16_t* img_gx = (int16_t*)aligned_alloc(64, num_pixels * sizeof(int16_t));
    int16_t* img_gy = (int16_t*)aligned_alloc(64, num_pixels * sizeof(int16_t));
    uint8_t* img_mag = (uint8_t*)aligned_alloc(64, num_pixels);
    uint8_t* img_dir = (uint8_t*)aligned_alloc(64, num_pixels);

    // Initialize arrays to zero
    memset(img_gx, 0, num_pixels * sizeof(int16_t));
    memset(img_gy, 0, num_pixels * sizeof(int16_t));

    std::cerr << "Reading image..." << std::endl;
    if (fread(img_in, 1, num_pixels, stdin) == (size_t)num_pixels) {
        
        std::cerr << "1. Applying Gaussian Blur..." << std::endl;
        gaussian_blur(img_in, img_blur, width, height);
        
        std::cerr << "2. Calculating Sobel Gradients (SoA)..." << std::endl;
        sobel_gradients(img_blur, img_gx, img_gy, width, height);
        
        std::cerr << "3. Calculating Normalized Magnitude..." << std::endl;
        compute_magnitude(img_gx, img_gy, img_mag, width, height);
        
        std::cerr << "4. Calculating Gradient Direction..." << std::endl;
        compute_direction(img_gx, img_gy, img_dir, width, height);

        // For now, let's output the MAGNITUDE image to stdout so we can view it
        std::cerr << "Writing magnitude image..." << std::endl;
        fwrite(img_mag, 1, num_pixels, stdout);
        std::cerr << "Done!" << std::endl;
        
    } else {
        std::cerr << "Failed to read image." << std::endl;
    }

    free(img_in); free(img_blur); free(img_gx); free(img_gy); free(img_mag); free(img_dir);
    return 0;
}
