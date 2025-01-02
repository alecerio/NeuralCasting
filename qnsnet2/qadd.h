#ifndef __QADD__
#define __QADD__

#include "quant.h"

#define NCAST_ADD(A, B, Y, SIZE) \
for(int i=0; i<SIZE; i++) { \
    Y[i] = A[i] + B[i]; \
}

#define NCAST_QADD(A, Sa, Za, B, Sb, Zb, Y, Sy, Zy, SIZE) \
{ \
float32_t Say = (Sa / Sy); \
float32_t Sby = (Sb / Sy); \
for(int i=0; i<SIZE; i++) { \
    int32_t temp00 = (int32_t)A[i] - Za; \
    int32_t temp01 = (int32_t)B[i] - Zb; \
    int32_t temp02 = (int32_t)(Say * temp00); \
    int32_t temp03 = (int32_t)(Sby * temp01); \
    int32_t temp04 = temp02 + temp03 + Zy; \
    NCAST_CLIP_INT8(temp04) \
    Y[i] =  (int8_t) temp04;\
} \
}

#define NCAST_QADD_FIXED(A, Sa, Za, B, Sb, Zb, Sy, Zy, Y, SIZE) \
{ \
int32_t Sby = (Sb * FIX_BASE / Sy); \
int32_t Say = (Sa * FIX_BASE / Sy); \
for(int i=0; i<SIZE; i++) { \
    int32_t temp00 = (int32_t)A[i] - Za; \
    int32_t temp01 = (int32_t)B[i] - Zb; \
    int32_t temp02 = (int32_t)(Say * temp00) / (int32_t)FIX_BASE; \
    int32_t temp03 = (int32_t)(Sby * temp01) / (int32_t)FIX_BASE; \
    int32_t temp04 = temp02 + temp03 + Zy; \
    NCAST_CLIP_INT8(temp04) \
    Y[i] =  (int8_t) temp04;\
} \
}

#endif // __QADD__