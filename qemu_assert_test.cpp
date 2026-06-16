#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>

void gaussian_blur(const uint8_t* input, uint8_t* output, int width, int height);
void sobel_gradients(const uint8_t* input, int16_t* gx_out, int16_t* gy_out, int width, int height);
void compute_magnitude(const int16_t* gx, const int16_t* gy, uint8_t* output, int width, int height);
void compute_direction(const int16_t* gx, const int16_t* gy, uint8_t* output, int width, int height);

static void test_gaussian_uniform() {
    const int W = 32, H = 32;
    uint8_t* in  = (uint8_t*)calloc(W * H, 1);
    uint8_t* out = (uint8_t*)calloc(W * H, 1);
    memset(in, 128, W * H);
    gaussian_blur(in, out, W, H);
    for (int y = 2; y < H - 2; y++)
        for (int x = 2; x < W - 2; x++)
            assert(out[y * W + x] == 128 && "Gaussian uniform invariant failed");
    free(in); free(out);
    printf("  test_gaussian_uniform:       PASS\n");
}

static void test_sobel_zero_gradient() {
    const int W = 32, H = 32;
    uint8_t*  in = (uint8_t*)calloc(W * H, 1);
    int16_t* gx  = (int16_t*)calloc(W * H, sizeof(int16_t));
    int16_t* gy  = (int16_t*)calloc(W * H, sizeof(int16_t));
    memset(in, 200, W * H);
    sobel_gradients(in, gx, gy, W, H);
    for (int y = 1; y < H - 1; y++)
        for (int x = 1; x < W - 1; x++) {
            assert(gx[y * W + x] == 0 && "Gx nonzero on uniform image");
            assert(gy[y * W + x] == 0 && "Gy nonzero on uniform image");
        }
    free(in); free(gx); free(gy);
    printf("  test_sobel_zero_gradient:    PASS\n");
}

static void test_vertical_edge() {
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
        assert(abs(gx[i]) > 100 && "Gx too small at vertical edge");
        assert(abs(gy[i]) <= 10  && "Gy too large at vertical edge");
    }
    free(in); free(gx); free(gy);
    printf("  test_vertical_edge:          PASS\n");
}

static void test_horizontal_edge() {
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
        assert(abs(gy[i]) > 100 && "Gy too small at horizontal edge");
        assert(abs(gx[i]) <= 10  && "Gx too large at horizontal edge");
    }
    free(in); free(gx); free(gy);
    printf("  test_horizontal_edge:        PASS\n");
}

static void test_pipeline_produces_nonzero_magnitude() {
    const int W = 32, H = 32;
    uint8_t*  in  = (uint8_t*)calloc(W * H, 1);
    uint8_t*  blr = (uint8_t*)calloc(W * H, 1);
    int16_t*  gx  = (int16_t*)calloc(W * H, sizeof(int16_t));
    int16_t*  gy  = (int16_t*)calloc(W * H, sizeof(int16_t));
    uint8_t*  mag = (uint8_t*)calloc(W * H, 1);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            in[y * W + x] = (x < W / 2) ? 0 : 255;
    gaussian_blur(in, blr, W, H);
    sobel_gradients(blr, gx, gy, W, H);
    compute_magnitude(gx, gy, mag, W, H);
    int has_nonzero = 0;
    for (int i = 0; i < W * H; i++)
        if (mag[i] > 0) { has_nonzero = 1; break; }
    assert(has_nonzero && "Full pipeline produced zero magnitude on edge image");
    free(in); free(blr); free(gx); free(gy); free(mag);
    printf("  test_pipeline_nonzero_mag:   PASS\n");
}

static void test_determinism() {
    const int W = 32, H = 32;
    uint8_t* in   = (uint8_t*)calloc(W * H, 1);
    uint8_t* out1 = (uint8_t*)calloc(W * H, 1);
    uint8_t* out2 = (uint8_t*)calloc(W * H, 1);
    for (int i = 0; i < W * H; i++)
        in[i] = (uint8_t)(i * 7 + 13);
    gaussian_blur(in, out1, W, H);
    gaussian_blur(in, out2, W, H);
    for (int i = 0; i < W * H; i++)
        assert(out1[i] == out2[i] && "Gaussian is not deterministic");
    free(in); free(out1); free(out2);
    printf("  test_determinism:            PASS\n");
}

int main() {
    printf("=== QEMU Assert Tests (RISC-V rv64gcv) ===\n");
    test_gaussian_uniform();
    test_sobel_zero_gradient();
    test_vertical_edge();
    test_horizontal_edge();
    test_pipeline_produces_nonzero_magnitude();
    test_determinism();
    printf("==========================================\n");
    printf("All QEMU assert tests PASSED.\n");
    return 0;
}
