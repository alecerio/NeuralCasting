#ifndef __NCAST_QUANT__
#define __NCAST_QUANT__

#include <stdint.h>

#define FIX_BASE (1 << 16)

typedef float float32_t;

#define NCAST_CLIP_INT8(X) \
if(X < INT8_MIN) \
    X = INT8_MIN; \
else if(X >= INT8_MAX-1) \
    X = INT8_MAX-1;

#define NCAST_ROUND(X) \
(X >= 0) ? (int32_t)(X + 0.5) : (int32_t)(X - 0.5)

#define NCAST_SCALF(r_max, r_min, q_max, q_min, S) \
S = (r_max - r_min) / (q_max - q_min);

#define NCAST_ZERO(q_min, r_min, S, Z) \
Z = NCAST_ROUND(q_min - (r_min/S));

#define NCAST_QUANT8(W, WQ, SIZE, S, Z) \
for(int i=0; i<SIZE; i++) { \
    int32_t temp = NCAST_ROUND((W[i]/S)+Z); \
    NCAST_CLIP_INT8(temp) \
    WQ[i] = (uint8_t) temp; \
}

#define NCAST_QUANT8_FIXED(W, WQ, SIZE, S, Z) \
for(int i=0; i<SIZE; i++) { \
    int32_t temp = NCAST_ROUND(((W[i]*FIX_BASE)/S)+Z); \
    NCAST_CLIP_INT8(temp) \
    WQ[i] = (uint8_t) temp; \
}

#define NCAST_DQUANT8(WQ, W, SIZE, S, Z) \
for(int i=0; i<SIZE; i++) { \
    W[i] = S * ((int32_t)WQ[i] - Z); \
}

#define NCAST_DQUANT8_FIXED(WQ, W, SIZE, S, Z) \
for(int i=0; i<SIZE; i++) { \
    W[i] = ((float32_t)S / (float32_t)FIX_BASE) * ((int32_t)WQ[i] - Z); \
}

#endif // __NCAST_QUANT__
