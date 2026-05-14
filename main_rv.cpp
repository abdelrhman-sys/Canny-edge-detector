#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include "image_io.h"

// 2.2 Gaussian Blur
void gaussian_blur(const uint8_t* input, uint8_t* output, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sum = 0;
            sum += input[(y - 1) * width + (x - 1)] * 1; sum += input[(y - 1) * width + x] * 2; sum += input[(y - 1) * width + (x + 1)] * 1;
            sum += input[y * width + (x - 1)]       * 2; sum += input[y * width + x]       * 4; sum += input[y * width + (x + 1)]       * 2;
            sum += input[(y + 1) * width + (x - 1)] * 1; sum += input[(y + 1) * width + x] * 2; sum += input[(y + 1) * width + (x + 1)] * 1;
            output[y * width + x] = sum / 16;
        }
    }
}

// 2.3 Sobel Gradients
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

// 2.4 Gradient Magnitude
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

// 2.5 Gradient Direction
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
        output[i] = dir; 
    }
}

// NEW: Non-Maximum Suppression
void non_max_suppression(const uint8_t* mag, const uint8_t* dir, uint8_t* output, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int current = mag[y * width + x];
            int n1 = 0, n2 = 0;
            
            switch (dir[y * width + x]) {
                case 0: n1 = mag[y * width + (x - 1)]; n2 = mag[y * width + (x + 1)]; break;
                case 1: n1 = mag[(y - 1) * width + (x + 1)]; n2 = mag[(y + 1) * width + (x - 1)]; break;
                case 2: n1 = mag[(y - 1) * width + x]; n2 = mag[(y + 1) * width + x]; break;
                case 3: n1 = mag[(y - 1) * width + (x - 1)]; n2 = mag[(y + 1) * width + (x + 1)]; break;
            }
            
            if (current >= n1 && current >= n2) output[y * width + x] = current;
            else output[y * width + x] = 0;
        }
    }
}

// NEW: Hysteresis Thresholding
void hysteresis(uint8_t* img, int width, int height, uint8_t low_thresh, uint8_t high_thresh) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            if (img[y * width + x] >= high_thresh) img[y * width + x] = 255;
            else if (img[y * width + x] < low_thresh) img[y * width + x] = 0;
            else {
                bool connected = false;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (img[(y + dy) * width + (x + dx)] >= high_thresh) connected = true;
                    }
                }
                img[y * width + x] = connected ? 255 : 0;
            }
        }
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
    uint8_t* img_nms = (uint8_t*)aligned_alloc(64, num_pixels); 

    memset(img_gx, 0, num_pixels * sizeof(int16_t));
    memset(img_gy, 0, num_pixels * sizeof(int16_t));

    if (fread(img_in, 1, num_pixels, stdin) == (size_t)num_pixels) {
        
        gaussian_blur(img_in, img_blur, width, height);
        sobel_gradients(img_blur, img_gx, img_gy, width, height);
        compute_magnitude(img_gx, img_gy, img_mag, width, height);
        compute_direction(img_gx, img_gy, img_dir, width, height);
        
        non_max_suppression(img_mag, img_dir, img_nms, width, height);
        hysteresis(img_nms, width, height, 30, 80); 

        fwrite(img_nms, 1, num_pixels, stdout);
        
    }

    free(img_in); free(img_blur); free(img_gx); free(img_gy); free(img_mag); free(img_dir); free(img_nms);
    return 0;
}
