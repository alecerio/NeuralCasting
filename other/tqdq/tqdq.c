#include "ncast_lib.h"
#include <stdio.h>

#define SIZE (5)
#define Q (31)

int main() {
    float x[SIZE] = {-0.2f, -0.1f, 0.0f, 3.5f, 4.0f};
    NC_PRINTTNSFLOAT(x, 0, SIZE-1);
    float s; int32_t z; NC_SZ_SINT8(x,SIZE,s,z);
    int64_t sfx; NC_SFX_SINT8(x,SIZE,s,sfx,Q);

    printf("%f, %d, %ld\n", s, z, sfx);
    
    int8_t xq[SIZE];
    NC_QLIN_FXS8(x,xq,SIZE,sfx,z,Q);

    float xr[SIZE];
    NC_DQLIN_FXS8(xq,xr,SIZE,sfx,z,Q);

    NC_PRINTTNSINT(xq,0,SIZE-1,PRId8);
    NC_PRINTTNSFLOAT(xr,0,SIZE-1);

    return 0;
}
