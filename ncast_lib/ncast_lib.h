#ifndef NCAST_LIB_H
#define NCAST_LIB_H

#include <stdint.h>

/***********************************************************************************************************
 *                                          PATMOS PRAGMA IN MACROS
 ***********************************************************************************************************/

#define CUSTOM_PRAGMA(x) _Pragma(#x)

/***********************************************************************************************************
 *                                              TEST UTILITIES
 ***********************************************************************************************************/

#define NC_OUTTNS(OUTFILENAME,X,SIZE,DTYPE) \
{ \
  FILE *f = fopen(OUTFILENAME, "w"); \
  if (!f) return -1; \
  for (size_t i = 0; i < SIZE; i++) { \
    if (fprintf(f, DTYPE, X[i]) < 0) { \
      fclose(f); \
      return -2; \
    } \
    if (i + 1 < SIZE) { \
      fputc(',', f); \
    } \
  } \
  fclose(f); \
}

/***********************************************************************************************************
 *                                        QUANTIZATION SUPPORT MACROS
 ***********************************************************************************************************/

/***************************************************
 * Macro: NC_QLIN_FXS8
 * Description:
 *   Performs linear quantization on an input array,
 *   converting floating-point values to int8.
 *
 * Parameters:
 *   X    - Input array (float)
 *   Y    - Output array (int8_t)
 *   SIZE - Number of elements
 *   SFX  - Scale factor
 *   ZFX  - Zero-point factor
 *   Q    - Quantization factor (bit shift)
 ***************************************************/

#define NC_QLIN_FXS8(X,Y,SIZE,SFX,ZFX,Q) \
{ \
    int32_t zfx = (int32_t)ZFX; \
    int32_t sfx = (int32_t)SFX; \
    int32_t a1 = zfx * sfx; \
    for(int i=0; i<SIZE; i++) { \
      float f0 = X[i] * (float)((int32_t)(1) << Q); \
      int64_t a0 = (int32_t) f0; \
      int64_t a2 = a0 + a1; \
      int64_t a4 = a2 / sfx; \
      NC_CLIP_SINT8(a4); \
      int8_t a5 = (int8_t)a4; \
      Y[i] = a5; \
    } \
}

/***************************************************
 * Macro: NC_DQLIN_FXS8
 * Description:
 *   Performs linear dequantization on an input array,
 *   converting int8 values back to floating-point.
 *
 * Parameters:
 *   X    - Input array (int8_t)
 *   Y    - Output array (float)
 *   SIZE - Number of elements
 *   SFX  - Scale factor
 *   ZFX  - Zero-point factor
 *   Q    - Quantization factor (bit shift)
 ***************************************************/

#define NC_DQLIN_FXS8(X,Y,SIZE,SFX,ZFX,Q) \
{ \
  int32_t sfx = (int32_t) SFX; \
  int32_t zfx = (int32_t) ZFX; \
  for (int i=0; i<SIZE; i++) { \
    int32_t a0 = (int32_t) X[i] - zfx; \
    int32_t a1 = (int32_t)sfx * a0; \
    float a2 = (float)a1 / (float)((int32_t)1 << Q); \
    Y[i] = a2; \
  } \
}

/***************************************************
 * Macro: NC_CLIP_SINT8
 * Description:
 *   Clips a value to the int8_t range.
 *
 * Parameters:
 *   X - Value to be clipped (modified in place)
 ***************************************************/

#define NC_CLIP_SINT8(X) \
{ \
    if(X < INT8_MIN) \
        X = INT8_MIN; \
    else if(X > INT8_MAX) \
        X = INT8_MAX;  \
}

/***********************************************************************************************************
 *                                              NC OPERATORS
 ***********************************************************************************************************/

/***************************************************
 * Macro op structure: NC_<OP-NAME>_<SF-TYPE><S/U><N-BITS>
 * <OP-NAME>: contracted name for the operator (e.g., QLINADD for Quantized Linear Addition)
 * <SF-TYPE>: scaling factor type (e.g., FX for fixed point)
 * <S/U>: if quantization is signed (S) or unsigned (U)
 * <N-BITS>: number of bits for the quantization (e.g., 8 for 8-bit)
 ***************************************************/

/***************************************************
 * Macro: NC_QLADD_FXS8
 * Description:
 *   Performs quantized element-wise addition between
 *   two int8 input arrays and produces an int8 output.
 *
 * Parameters:
 *   AQ      - First input array (int8_t)
 *   BQ      - Second input array (int8_t)
 *   CQ      - Output array (int8_t)
 *   SIZE    - Number of elements
 *   SFXA    - Scale factor for AQ (fixed point)
 *   ZFXA    - Zero-point for AQ
 *   SFXB    - Scale factor for BQ (fixed point) 
 *   ZFXB    - Zero-point for BQ
 *   SFXC    - Scale factor for CQ (fixed point)
 *   ZFXC    - Zero-point for CQ
 *   ACCTYPE - Accumulator type (e.g., int32_t, int64_t)
 ***************************************************/

#define NC_QLADD_FXS8(AQ,BQ,CQ,SIZE,SFXA,ZFXA,SFXB,ZFXB,SFXC,ZFXC,ACCTYPE) \
{ \
ACCTYPE a4 = (ACCTYPE)SFXC * ZFXC; \
CUSTOM_PRAGMA(loopbound min 0 max SIZE) \
for(int i=0; i<SIZE; i++) { \
  ACCTYPE a0 = AQ[i] - (ACCTYPE)ZFXA; \
  ACCTYPE a2 = (ACCTYPE)SFXA * a0; \
  ACCTYPE a1 = BQ[i] - (ACCTYPE)ZFXB; \
  ACCTYPE a3 = (ACCTYPE)SFXB * a1; \
  ACCTYPE a5 = a2 + a3 + a4; \
  ACCTYPE a6 = (int32_t) a5  / (int32_t) SFXC; \
  CQ[i] = (int8_t) a6; \
} \
}

#define NC_QLINCONV_FXS8(XQ,WQ,BQ,YQ,KS,CIN,LIN,COUT,LOUT,PAD,DIL,STR,SFXX,ZX,SFXW,ZW,SFXY,ZY,SFXB,ZB,Q,ACCTYPE) \
{ \
CUSTOM_PRAGMA(loopbound min 0 max COUT) \
for(int o=0; o < COUT; o++) { \
  ACCTYPE n0 = (ACCTYPE)SFXB * (BQ[o] - ZB); \
  ACCTYPE d0 = (ACCTYPE)SFXY; \
  ACCTYPE a0 = n0 / d0; \
CUSTOM_PRAGMA(loopbound min 0 max LOUT) \
  for(int t=0; t < LOUT; t++) { \
    ACCTYPE acc = 0; \
CUSTOM_PRAGMA(loopbound min 0 max CIN) \
    for(int c=0; c < CIN; c++) { \
CUSTOM_PRAGMA(loopbound min 0 max KS) \
        for(int k=0; k < KS; k++) { \
            acc += (WQ[o*CIN*KS + c*KS + k] - ZW) * (XQ[c*LIN + t+k-PAD]-ZX); \
        } \
        ACCTYPE t0 = (ACCTYPE)SFXW * (ACCTYPE)SFXX * acc; \
        ACCTYPE t1 = (ACCTYPE)SFXY << Q; \
        ACCTYPE a1 = t0 / t1; \
        ACCTYPE a2 = a0 + a1 + ZY; \
        YQ[o*LOUT+t] = (int8_t) a2; \
    } \
  } \
} \
}

#define NC_QLMUL_FXS8(AQ,BQ,CQ,SIZE,SFXA,ZFXA,SFXB,ZFXB,SFXC,ZFXC,Q) \
{ \
int64_t a0 = (int64_t)SFXA * (int64_t)SFXB; \
int64_t a4 = (int64_t)SFXC * ((int64_t)1 << Q); \
CUSTOM_PRAGMA(loopbound min 0 max SIZE) \
for(int i; i < SIZE; i++) { \
  int64_t a1 = AQ[i] - ZFXA; \
  int64_t a2 = BQ[i] - ZFXB; \
  int64_t a3 = a0 * a1 * a2; \
  int64_t a5, r5; IDIVMOD64(a3, a4, a5, r5) \
  int64_t a6 = a5 + ZFXC; \
  CQ[i] = a6; \
} \
}

#define NC_TR2D(X,Y,COLS,ROWS) \
{ \
CUSTOM_PRAGMA(loopbound min 0 max ROWS) \
for(int i=0; i < ROWS; i++) { \
  CUSTOM_PRAGMA(loopbound min 0 max COLS) \
  for(int j=0; j < COLS; j++) { \
    Y[j*ROWS+i]= X[i*COLS+j]; \
  } \
} \
}

#define NC_RELU(X,Y,SIZE,SFXX,ZFXX,SFXY,ZFXY) \
{ \
CUSTOM_PRAGMA(loopbound min 0 max SIZE) \
for(int i=0; i < SIZE; i++) { \
  if(X[i] >= ZFXX) { \
    int64_t a0 = (int64_t)SFXX * (X[i] - ZFXX); \
    int64_t a1 = a0 / SFXY; \
    int64_t a2 = a1 + ZFXY; \
    NC_CLIP_SINT8(a2); \
    Y[i] = (int8_t) (a2); \
  } \
  else { \
    Y[i] = (int8_t) ZFXY; \
  } \
} \
}

#define NC_QLSUB_FXS8(AQ,BQ,CQ,SIZE,SFXA,ZFXA,SFXB,ZFXB,SFXC,ZFXC) \
{ \
CUSTOM_PRAGMA(loopbound min 0 max SIZE) \
for(int i=0; i<SIZE; i++) { \
  int64_t a0 = (int64_t) AQ[i] - ZFXA; \
  int64_t a2 = SFXA * a0; \
  int64_t a1 = (int64_t) BQ[i] - ZFXB; \
  int64_t a3 = SFXB * a1; \
  int64_t a4 = SFXC * ZFXC; \
  int64_t a5 = a2 - a3 + a4; \
  int64_t a6 = (int32_t)a5 / (int32_t)SFXC; \
  NC_CLIP_SINT8(a6); \
  CQ[i] = (int8_t) a6; \
} \
}

#define NC_QTANHLUT_FXS8_SIZE (256)
#define NC_QTANHLUT_FXS8_STEP (1)
#define NC_QTANHLUT_FXS8_SFXLUTX (101058056)
#define NC_QTANHLUT_FXS8_ZFXLUTX (0)
#define NC_QTANHLUT_FXS8_SFXLUTY (16842802)
#define NC_QTANHLUT_FXS8_ZFXLUTY (0)
extern const int8_t NC_QTANHLUT_FXS8[NC_QTANHLUT_FXS8_SIZE];

#define NC_QTANH_FXS8(X,Y,SIZE,SFXX,ZFXX) \
{ \
CUSTOM_PRAGMA(loopbound min 0 max SIZE) \
for(int i=0; i<SIZE; i++) { \
  int64_t a0 = SFXX * (X[i] - ZFXX); \
  int64_t lutx = (a0 / NC_QTANHLUT_FXS8_SFXLUTX) + NC_QTANHLUT_FXS8_ZFXLUTX; \
  int32_t idxlut = ((int32_t) lutx + (int32_t) (128)) / NC_QTANHLUT_FXS8_STEP; \
  Y[i] = NC_QTANHLUT_FXS8[idxlut]; \
} \
}

#define NC_UNSQUEEZE(X,Y,SIZE) \
{ \
memcpy(Y,X,SIZE*sizeof(*Y)); \
}

#define NC_QLPRELU_FXS8(X,W,Y,SIZE,SFXX,ZFXX,SFXY,ZFXY,Q) \
{ \
CUSTOM_PRAGMA(loopbound min 0 max SIZE) \
for(int i=0; i<SIZE; i++) { \
  int64_t a0; \
  if(X[i] >= ZFXX) \
    a0 = (int64_t) (SFXX) * (X[i] - ZFXX) << Q; \
  else \
    a0 = (int64_t) (W) * (int64_t) (SFXX) * (X[i] - ZFXX); \
  int64_t a1 = (int64_t) (SFXY) << Q; \
  int64_t a2 = (a0 / a1); \
  int64_t a3 = a2 + ZFXY; \
  NC_CLIP_SINT8(a3); \
  Y[i] = (int8_t)(a3); \
} \
}

#define NC_QSIGMOIDLUT_FXS8_SIZE (256)
#define NC_QSIGMOIDLUT_FXS8_STEP (1)
#define NC_QSIGMOIDLUT_FXS8_SFXLUTX (101058056)
#define NC_QSIGMOIDLUT_FXS8_ZFXLUTX (0)
#define NC_QSIGMOIDLUT_FXS8_SFXLUTY (8379858)
#define NC_QSIGMOIDLUT_FXS8_ZFXLUTY (-129)
extern const int8_t NC_QSIGMOIDLUT_FXS8[NC_QSIGMOIDLUT_FXS8_SIZE];

#define NC_QSIGMOID_FXS8(X,Y,SIZE,SFXX,ZFXX) \
{ \
CUSTOM_PRAGMA(loopbound min 0 max SIZE) \
for(int i=0; i<SIZE; i++) { \
  int64_t a0 = SFXX * (X[i] - ZFXX); \
  int64_t lutx = (a0 / NC_QSIGMOIDLUT_FXS8_SFXLUTX) + NC_QSIGMOIDLUT_FXS8_ZFXLUTX; \
  int32_t idxlut = ((int32_t) lutx + (int32_t) (128)) / NC_QSIGMOIDLUT_FXS8_STEP; \
  Y[i] = NC_QSIGMOIDLUT_FXS8[idxlut]; \
} \
}

#define NC_QLINMATMUL(AQ,BQ,CQ,M,N,K,SFXA,SFXB,ZA,ZB,SFXY,ZY,Q,QS) \
{ \
int64_t a0 = (int64_t) SFXA * (int64_t) SFXB >> QS; \
int64_t a3 = (int64_t) SFXY * ((int64_t)1 << (Q-QS)); \
CUSTOM_PRAGMA(loopbound min 0 max M) \
for(int i=0; i<M; i++) { \
CUSTOM_PRAGMA(loopbound min 0 max N) \
    for(int j=0; j<N; j++) { \
        int64_t acc = 0; \
CUSTOM_PRAGMA(loopbound min 0 max K) \
        for(int k=0; k<K; k++) { \
            acc += ((int64_t) AQ[i*M+k] - ZA) * ((int64_t) BQ[k*N+j] - ZB); \
        } \
        int64_t acca0 = acc * a0; \
        int64_t q1, r1; IDIVMOD64(acca0, a3, q1, r1) \
        CQ[i*N+j] = (int8_t)(q1 + ZY); \
    } \
} \
}

#define UDIVMOD64(n, d, q_out, r_out) \
{ \
uint64_t _n = (n); \
uint64_t _d = (d); \
uint64_t _q = 0; \
uint64_t _r = 0; \
CUSTOM_PRAGMA(loopbound min 0 max 63) \
for (int _i = 63; _i >= 0; _i--) { \
    _r = (_r << 1) | ((_n >> _i) & 1ULL); \
    if (_r >= _d) { \
        _r -= _d; \
        _q |= (1ULL << _i); \
    } \
} \
(q_out) = _q; \
(r_out) = _r; \
}

#define IDIVMOD64(n, d, q_out, r_out) \
{ \
int64_t _n = (n); \
int64_t _d = (d); \
uint64_t _un = (_n < 0) ? -(uint64_t)_n : (uint64_t)_n; \
uint64_t _ud = (_d < 0) ? -(uint64_t)_d : (uint64_t)_d; \
uint64_t _uq = 0; \
uint64_t _ur = 0; \
CUSTOM_PRAGMA(loopbound min 0 max 63) \
for (int _i = 63; _i >= 0; _i--) { \
    _ur = (_ur << 1) | ((_un >> _i) & 1ULL); \
    if (_ur >= _ud) { \
        _ur -= _ud; \
        _uq |= (1ULL << _i); \
    } \
} \
int64_t _q = (_n < 0) ^ (_d < 0) ? -(int64_t)_uq : (int64_t)_uq; \
int64_t _r = (_n < 0) ? -(int64_t)_ur : (int64_t)_ur; \
(q_out) = _q; \
  (r_out) = _r; \
}

#endif // NCAST_LIB_H