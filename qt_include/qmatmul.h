#ifndef __NCAST_QMATMUL__
#define __NCAST_QMATMUL__

#include "quant.h"

#define NCAST_MATMUL(A, B, Y, M, K, N) \
for (int i = 0; i < M; i++) { \
    for (int j = 0; j < N; j++) { \
        Y[i*M+j] = 0.0f; \
        for (int k = 0; k < K; k++) { \
            Y[i*M+j] += A[i*K+k] * B[k*N+j]; \
        } \
    } \
}

#define NCAST_QMATMUL(A, Sa, Za, B, Sb, Zb, Y, Sy, Zy, M, K, N) \
float32_t Saby = (Sa*Sb) / Sy; \
for (int i = 0; i < M; i++) { \
    for (int j = 0; j < N; j++) { \
        float32_t acc = 0.0f; \
        for (int k = 0; k < K; k++) { \
            float32_t temp00 = A[i*K+k] - Za; \
            float32_t temp01 = B[k*N+j] - Zb; \
            acc += temp00 * temp01; \
        } \
        acc *= Saby; \
        acc += Zy; \
        NCAST_CLIP_INT8(acc) \
        Y[i*M+j] = acc; \
    } \
}

#define NCAST_QMATMUL_FIX(A, Sa, Za, B, Sb, Zb, Sy, Zy, Y, M, K, N) \
{ \
int64_t Sab = (int64_t)(Sa * Sb) << 32; \
int32_t Saby = (Sab / Sy) >> 32; \
for (int i = 0; i < M; i++) { \
    for (int j = 0; j < N; j++) { \
        int32_t acc = 0; \
        for (int k = 0; k < K; k++) { \
            int32_t temp00 = A[i*K+k] - Za; \
            int32_t temp01 = B[k*N+j] - Zb; \
            acc += temp00 * temp01; \
        } \
        int32_t temp02 = (acc * Saby) / FIX_BASE; \
        int32_t temp03 = temp02 + Zy; \
        NCAST_CLIP_INT8(temp03) \
        Y[i*N+j] = (int8_t)temp03; \
    } \
} \
}

#endif // __NCAST_QMATMUL__