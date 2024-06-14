#ifndef __NCUTILS__
#define __NCUTILS__

#include <stdio.h>
#include <stdint.h>
#include <limits.h>

#define NCAST_PRINT_MAT(X, SX, SY, T, HEADER) \
printf(" ########################### \n"); \
printf(HEADER); \
printf("\n"); \
for(int i=0; i<SX; i++) { \
    for(int j=0; j<SY; j++) { \
        printf(""#T" ", X[i*SY+j]); \
    } \
    printf("\n"); \
}

#define NCAST_REL_OP(X, SL, Y, OP) \
Y = X[0]; \
for(int i=1; i<SL; i++) { \
    if(X[i] OP Y) \
        Y = X[i]; \
}

#define NCAST_ROUND(X) \
(X >= 0) ? (int32_t)(X + 0.5) : (int32_t)(X - 0.5)

#define NCAST_ABS(X) \
(X > 0) ? X : (-X)

#define NCAST_ACCESS_LUT(LUT, X, Y, MIN_RANGE, MAX_RANGE, UPPER, LOWER, LUT_SIZE) \
if(X < MIN_RANGE) \
    Y = LOWER; \
else if(X > MAX_RANGE) \
    Y = UPPER; \
else { \
    int32_t MAX_INDEX = LUT_SIZE - 1; \
    float x = MAX_INDEX - (MAX_INDEX * (MAX_RANGE - X)) / (MAX_RANGE - MIN_RANGE); \
    float dx = x - (int32_t)x; \
    float y0 = LUT[(int32_t)x]; \
    float y1 = LUT[((int32_t)x)+1]; \
    Y = y0 + (y1 - y0) * dx; \
}

#endif // __NCUTILS__