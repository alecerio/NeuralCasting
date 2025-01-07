#ifndef __NCAST_QMUL__
#define __NCAST_QMUL__

#include "quant.h"

#define NCAST_MUL(A, B, Y, SIZE) \
for(int i=0; i<SIZE; i++) { \
    Y[i] = A[i] * B[i]; \
}

#define NCAST_QMUL(A, Sa, Za, B, Sb, Zb, Y, Sy, Zy, SIZE) \
{ \
float32_t Saby = (Sa * Sb) / Sy; \
for(int i=0; i<SIZE; i++) { \
    int32_t temp00 = A[i] - Za; \
    int32_t temp01 = B[i] - Zb; \
    Y[i] = (uint32_t) Saby * (temp00 * temp01) + Zy; \
    NCAST_CLIP_INT8(Y[i]) \
} \
}

#define NCAST_QMUL_FIX(A, Sa, Za, B, Sb, Zb, Sy, Zy, Y, SIZE) \
{ \
int64_t Sab = (int64_t)(Sa * Sb) << 32; \
int32_t Saby = (Sab / Sy) >> 32; \
for(int i=0; i<SIZE; i++) { \
    int32_t temp00 = A[i] - Za; \
    int32_t temp01 = B[i] - Zb; \
    int32_t temp02 = temp00 * temp01; \
    int32_t temp03 = (Saby * temp02) / FIX_BASE; \
    int32_t temp04 = temp03 + Zy; \
    NCAST_CLIP_INT8(temp04) \
    Y[i] = (int8_t) temp04; \
} \
}


#endif // __NCAST_QMUL__
