#ifndef __NCAST_UTILS__
#define __NCAST_UTILS__

#include <stdio.h>

#define NCAST_PRINT_MAT(X, SX, SY, T, HEADER) \
printf(" ########################### \n"); \
printf(HEADER); \
printf("\n"); \
for(int i=0; i<SX; i++) { \
    for(int j=0; j<SY; j++) { \
        printf(""#T", ", X[i*SY+j]); \
    } \
    printf("\n"); \
}

#define GEN_QUANTIZED_MATRIX(N, W, T, SIZE) \
printf("static %s %s[%d]={", T, N, SIZE); \
for(int i=0; i<SIZE; i++) { \
    printf("%d, ", W[i]); \
} \
printf("};\n");

#endif // __NCAST_UTILS__
