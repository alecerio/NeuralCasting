#include "ncast_lib.h"
#include <stdio.h>

#define SIZE (5)
#define Q (31)

int main() {
    float a[SIZE] = {-0.2f, -0.1f, 0.0f, 3.5f, 4.0f};
    float b[SIZE] = {-0.6f, -0.3f, 0.0f, 1.0f, 0.5f};

    float c[SIZE]; NC_FADD(a,b,c,SIZE);
    

    float sa; int32_t za; NC_SZ_SINT8(a,SIZE,sa,za);
    int64_t sfxa; NC_SFX_SINT8(a,SIZE,sa,sfxa,Q);

    float sb; int32_t zb; NC_SZ_SINT8(b,SIZE,sb,zb);
    int64_t sfxb; NC_SFX_SINT8(b,SIZE,sb,sfxb,Q);

    float sc; int32_t zc; NC_SZ_SINT8(c,SIZE,sc,zc);
    int64_t sfxc; NC_SFX_SINT8(c,SIZE,sc,sfxc,Q);
    
    int8_t aq[SIZE];
    NC_QLIN_FXS8(a,aq,SIZE,sfxa,za,Q);

    int8_t bq[SIZE];
    NC_QLIN_FXS8(b,bq,SIZE,sfxb,zb,Q);

    int8_t cq[SIZE];
    NC_QLADD_FXS8(aq,bq,cq,SIZE,sfxa,za,sfxb,zb,sfxc,zc,Q);

    float cr[SIZE];
    NC_DQLIN_FXS8(cq,cr,SIZE,sfxc,zc,Q);

    NC_PRINTTNSFLOAT(cr,0,SIZE-1);

    return 0;
}
