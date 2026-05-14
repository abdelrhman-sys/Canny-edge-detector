#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <iostream>
#include <cstdio>   // Added for FILE*, fopen, fread
#include <cstdlib>
#include <cstdint>

// Function to allocate 64-byte aligned memory
uint8_t* allocate_image(int width, int height) {
    size_t size = width * height;
    size_t aligned_size = (size + 63) & ~63; 
    
    uint8_t* ptr = (uint8_t*)aligned_alloc(64, aligned_size);
    if (!ptr) {
        std::cerr << "Memory allocation failed!" << std::endl;
        exit(1);
    }
    return ptr;
}

// Function to load a raw grayscale image
bool load_raw_image(const char* filename, uint8_t* buffer, int width, int height) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }
    size_t bytes_read = fread(buffer, 1, width * height, file);
    fclose(file);
    
    // Returns true if we successfully read all the bytes we expected
    return bytes_read == (size_t)(width * height);
}

// Function to save a raw grayscale image
bool save_raw_image(const char* filename, uint8_t* buffer, int width, int height) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return false;
    }
    size_t bytes_written = fwrite(buffer, 1, width * height, file);
    fclose(file);
    
    return bytes_written == (size_t)(width * height);
}

#endif
