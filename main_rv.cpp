#include <iostream>
#include <cstdio>
#include "image_io.h" // We still need this for allocate_image!

int main() {
    int width = 256;
    int height = 256;

    uint8_t* img_buffer = allocate_image(width, height);

    // CRITICAL: We must use std::cerr for all text logging so it prints to 
    // the terminal and doesn't accidentally get written into our output image!
    std::cerr << "Reading image from standard input..." << std::endl;
    
    // Read raw bytes directly from standard input (stdin)
    if (fread(img_buffer, 1, width * height, stdin) == (size_t)(width * height)) {
        std::cerr << "Image loaded successfully!" << std::endl;
        
        // Write raw bytes directly to standard output (stdout)
        fwrite(img_buffer, 1, width * height, stdout);
        std::cerr << "Image saved successfully!" << std::endl;
    } else {
        std::cerr << "Failed to read the image data." << std::endl;
    }

    free(img_buffer);
    return 0;
}
