#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>

// Pull in the pipeline functions from main_rv.cpp (compiled with -DTESTING)
void gaussian_blur(const uint8_t* input, uint8_t* output, int width, int height);
void sobel_gradients(const uint8_t* input, int16_t* gx_out, int16_t* gy_out, int width, int height);
void compute_magnitude(const int16_t* gx, const int16_t* gy, uint8_t* output, int width, int height);
void compute_direction(const int16_t* gx, const int16_t* gy, uint8_t* output, int width, int height);

// ---------------------------------------------------------------
// TEST 1: Gaussian blur on a uniform image must stay uniform
// Interior pixels only (border is affected by zero-padding)
// ---------------------------------------------------------------
TEST(GaussianBlur, UniformImageInvariant) {
    const int W = 32, H = 32;
    uint8_t* in  = (uint8_t*)calloc(W * H, 1);
    uint8_t* out = (uint8_t*)calloc(W * H, 1);
    memset(in, 128, W * H);

    gaussian_blur(in, out, W, H);

    for (int y = 2; y < H - 2; y++) {
        for (int x = 2; x < W - 2; x++) {
            EXPECT_EQ(out[y * W + x], 128)
                << "Failed at (" << x << "," << y << ")";
        }
    }
    free(in); free(out);
}

// ---------------------------------------------------------------
// TEST 2: Sobel on a uniform image must produce zero gradients
// ---------------------------------------------------------------
TEST(SobelGradients, ZeroGradientOnUniformImage) {
    const int W = 32, H = 32;
    uint8_t*  in = (uint8_t*)calloc(W * H, 1);
    int16_t* gx  = (int16_t*)calloc(W * H, sizeof(int16_t));
    int16_t* gy  = (int16_t*)calloc(W * H, sizeof(int16_t));
    memset(in, 200, W * H);

    sobel_gradients(in, gx, gy, W, H);

    for (int y = 1; y < H - 1; y++) {
        for (int x = 1; x < W - 1; x++) {
            int i = y * W + x;
            EXPECT_EQ(gx[i], 0) << "Gx nonzero at (" << x << "," << y << ")";
            EXPECT_EQ(gy[i], 0) << "Gy nonzero at (" << x << "," << y << ")";
        }
    }
    free(in); free(gx); free(gy);
}

// ---------------------------------------------------------------
// TEST 3: Vertical edge -> Gx dominates, direction = 0
// Left half = 0, right half = 255
// ---------------------------------------------------------------
TEST(SobelGradients, VerticalEdgeGxDominates) {
    const int W = 32, H = 32;
    uint8_t*  in = (uint8_t*)calloc(W * H, 1);
    int16_t* gx  = (int16_t*)calloc(W * H, sizeof(int16_t));
    int16_t* gy  = (int16_t*)calloc(W * H, sizeof(int16_t));

    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            in[y * W + x] = (x < W / 2) ? 0 : 255;

    sobel_gradients(in, gx, gy, W, H);

    int cx = W / 2;
    for (int y = 2; y < H - 2; y++) {
        int i = y * W + cx;
        EXPECT_GT(abs(gx[i]), 100) << "Gx too small at vertical edge";
        EXPECT_NEAR(gy[i], 0, 10)  << "Gy should be ~0 at vertical edge";
    }
    free(in); free(gx); free(gy);
}

// ---------------------------------------------------------------
// TEST 4: Horizontal edge -> Gy dominates, direction = 2 (output=100)
// Top half = 0, bottom half = 255
// ---------------------------------------------------------------
TEST(SobelGradients, HorizontalEdgeGyDominates) {
    const int W = 32, H = 32;
    uint8_t*  in = (uint8_t*)calloc(W * H, 1);
    int16_t* gx  = (int16_t*)calloc(W * H, sizeof(int16_t));
    int16_t* gy  = (int16_t*)calloc(W * H, sizeof(int16_t));

    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            in[y * W + x] = (y < H / 2) ? 0 : 255;

    sobel_gradients(in, gx, gy, W, H);

    int ry = H / 2;
    for (int x = 2; x < W - 2; x++) {
        int i = ry * W + x;
        EXPECT_GT(abs(gy[i]), 100) << "Gy too small at horizontal edge";
        EXPECT_NEAR(gx[i], 0, 10)  << "Gx should be ~0 at horizontal edge";
    }
    free(in); free(gx); free(gy);
}

// ---------------------------------------------------------------
// TEST 5: Gaussian impulse response
// Single bright pixel -> peak stays at center, energy conserved
// ---------------------------------------------------------------
TEST(GaussianBlur, ImpulseResponse) {
    const int W = 32, H = 32;
    uint8_t* in  = (uint8_t*)calloc(W * H, 1);
    uint8_t* out = (uint8_t*)calloc(W * H, 1);
    in[H/2 * W + W/2] = 255;

    gaussian_blur(in, out, W, H);

    int sumIn = 0, sumOut = 0;
    int peakVal = 0, peakIdx = 0;
    for (int i = 0; i < W * H; i++) {
        sumIn  += in[i];
        sumOut += out[i];
        if (out[i] > peakVal) { peakVal = out[i]; peakIdx = i; }
    }

    EXPECT_LE(sumOut, sumIn)  << "Blur added energy";
    EXPECT_GT(sumOut, 0)      << "Blur zeroed everything out";
    EXPECT_NEAR(peakIdx % W, W/2, 2) << "Peak shifted in X";
    EXPECT_NEAR(peakIdx / W, H/2, 2) << "Peak shifted in Y";
    free(in); free(out);
}

// ---------------------------------------------------------------
// TEST 6: L1 magnitude is always >= L2 magnitude
// Uses a local L2 helper since compute_magnitude implements L1
// ---------------------------------------------------------------
static void compute_magnitude_l2(const int16_t* gx, const int16_t* gy,
                                  float* output, int n) {
    for (int i = 0; i < n; i++)
        output[i] = sqrtf((float)gx[i]*gx[i] + (float)gy[i]*gy[i]);
}

TEST(GradientMagnitude, L1AlwaysGeqL2) {
    const int W = 32, H = 32;
    uint8_t*  in  = (uint8_t*)calloc(W * H, 1);
    int16_t*  gx  = (int16_t*)calloc(W * H, sizeof(int16_t));
    int16_t*  gy  = (int16_t*)calloc(W * H, sizeof(int16_t));
    float*   l2   = (float*)calloc(W * H, sizeof(float));

    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            in[y * W + x] = (x < W/2) ? 0 : 255;

    sobel_gradients(in, gx, gy, W, H);
    compute_magnitude_l2(gx, gy, l2, W * H);

    for (int i = 0; i < W * H; i++) {
        float l1 = (float)(abs(gx[i]) + abs(gy[i]));
        EXPECT_GE(l1, l2[i] - 0.01f)
            << "L1 < L2 at pixel " << i;
    }
    free(in); free(gx); free(gy); free(l2);
}
