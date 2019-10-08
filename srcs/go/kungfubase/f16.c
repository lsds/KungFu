#include "f16.h"

#include <stdint.h>

#ifdef ENABLE_F16
#include <immintrin.h>

// -mavx # for _mm256_add_ps
// -mf16c # for _mm256_cvtph_ps, _mm256_cvtps_ph

// FIXME: inline error when building TF extension
//   Undefined symbols for architecture x86_64:
//     "_batch_float16_sum", referenced from:
//         _float16_sum in libkungfu-base.a(kungfu_half.c.o)
// inline
void batch_float16_sum(void *z, const void *x, const void *y)
{
    __m256 x_m256   = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)x));
    __m256 y_m256   = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)y));
    __m256 z_m256   = _mm256_add_ps(x_m256, y_m256);
    __m128i z_m128i = _mm256_cvtps_ph(z_m256, 0);
    _mm_storeu_si128((__m128i *)z, z_m128i);
}

void float16_sum(void *pz, const void *px, const void *py, int len)
{
    uint16_t *z       = (uint16_t *)pz;
    const uint16_t *x = (const uint16_t *)px;
    const uint16_t *y = (const uint16_t *)py;

    const int len_aligned         = (len / 8) * 8;
    uint16_t *const z_end_aligned = z + len_aligned;

    for (; z < z_end_aligned; z += 8, x += 8, y += 8) {
        batch_float16_sum(z, x, y);
    }

    if (len_aligned < len) {
        const int m = len - len_aligned;
        uint16_t wx[8];
        uint16_t wy[8];
        uint16_t wz[8];
        for (int i = 0; i < m; ++i) {
            wx[i] = x[i];
            wy[i] = y[i];
        }
        batch_float16_sum(wz, wx, wy);
        for (int i = 0; i < m; ++i) { z[i] = wz[i]; }
    }
}

#else

#include <stdio.h>
#include <stdlib.h>

void float16_sum(void *z, const void *x, const void *y, int len)
{
    fprintf(stderr, "f16 support not enabled\n");
    exit(1);
}

#endif
