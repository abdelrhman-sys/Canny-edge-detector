#include <iostream>
#include <cstdio>
#include <cstdlib> // Added for abs()
#include "image_io.h"

// 1. Gaussian Blur (Smooths out the noise)
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

// 2. Sobel Filter (Finds the edges)
void sobel_filter(const uint8_t* input, uint8_t* output, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int gx = 0;
            int gy = 0;

            // Apply Gx Kernel (Detects Vertical Edges)
            gx += input[(y - 1) * width + (x - 1)] * -1;
            gx += input[(y - 1) * width + (x + 1)] * 1;
            gx += input[y * width + (x - 1)]       * -2;
            gx += input[y * width + (x + 1)]       * 2;
            gx += input[(y + 1) * width + (x - 1)] * -1;
            gx += input[(y + 1) * width + (x + 1)] * 1;

            // Apply Gy Kernel (Detects Horizontal Edges)
            gy += input[(y - 1) * width + (x - 1)] * 1;
            gy += input[(y - 1) * width + x]       * 2;
            gy += input[(y - 1) * width + (x + 1)] * 1;
            gy += input[(y + 1) * width + (x - 1)] * -1;
            gy += input[(y + 1) * width + x]       * -2;
            gy += input[(y + 1) * width + (x + 1)] * -1;

            // Calculate Fast Magnitude (Absolute sum)
            int magnitude = abs(gx) + abs(gy);

            // Clamp the value so it doesn't overflow past white (255)
            if (magnitude > 255) magnitude = 255;
            
            output[y * width + x] = (uint8_t)magnitude;
        }
    }
}

int main() {
    int width = 256;
    int height = 256;

    uint8_t* img_in = allocate_image(width, height);
    uint8_t* img_blur = allocate_image(width, height);
    uint8_t* img_edges = allocate_image(width, height);

    std::cerr << "Reading image..." << std::endl;
    if (fread(img_in, 1, width * height, stdin) == (size_t)(width * height)) {
        
        std::cerr << "Applying Gaussian Blur..." << std::endl;
        gaussian_blur(img_in, img_blur, width, height);
        
        std::cerr << "Applying Sobel Filter..." << std::endl;
        sobel_filter(img_blur, img_edges, width, height); 
        
        std::cerr << "Writing edge image..." << std::endl;
        fwrite(img_edges, 1, width * height, stdout);
        std::cerr << "Done!" << std::endl;
        
    } else {
        std::cerr << "Failed to read image." << std::endl;
    }

    free(img_in);
    free(img_blur);
    free(img_edges);
    return 0;
}
