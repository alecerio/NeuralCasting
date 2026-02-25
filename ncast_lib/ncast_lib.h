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

/***************************************************
 * Macro: NC_QLINCONV_FXS8
 * Description:
 *   Performs 1D quantized convolution between input
 *   activations and weights, producing an int8 output.
 *
 * Parameters:
 *   XQ      - Input feature map (int8_t)
 *   WQ      - Weights (int8_t)
 *   BQ      - Bias (int32_t)
 *   YQ      - Output feature map (int8_t)
 *   KS      - Kernel size
 *   CIN     - Number of input channels
 *   LIN     - Input length
 *   COUT    - Number of output channels
 *   LOUT    - Output length
 *   PAD     - Padding
 *   DIL     - Dilation (currently unused in loop)
 *   STR     - Stride
 *   SFXX    - Scale factor for input (fixed point)
 *   ZX      - Zero-point for input 
 *   SFXW    - Scale factor for weights (fixed point)
 *   ZW      - Zero-point for weights
 *   SFXY    - Scale factor for output (fixed point)
 *   ZY      - Zero-point for output
 *   SFXB    - Scale factor for bias (fixed point)
 *   ZB      - Zero-point for bias
 *   Q       - Quantization shift
 *   ACCTYPE - Accumulator type (e.g., int32_t, int64_t)
 ***************************************************/

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

/***************************************************
 * Macro: NC_QLINMATMUL
 * Description:
 *   Performs quantized matrix multiplication between
 *   two int8 matrices and produces an int8 output.
 *
 * Parameters:
 *   AQ      - Input matrix A (int8_t), shape [M x K]
 *   BQ      - Input matrix B (int8_t), shape [K x N]
 *   CQ      - Output matrix (int8_t), shape [M x N]
 *   M       - Number of rows of A and C
 *   N       - Number of columns of B and C
 *   K       - Shared dimension (columns of A, rows of B)
 *   SFXA    - Scale factor for A (fixed point)
 *   SFXB    - Scale factor for B (fixed point)
 *   ZA      - Zero-point for A
 *   ZB      - Zero-point for B
 *   SFXY    - Scale factor for output (fixed point)
 *   ZY      - Zero-point for output
 *   Q       - Quantization shift
 *   QS      - Scale adjustment shift
 *   ACCTYPE - Accumulator type (e.g., int32_t, int64_t)
 ***************************************************/

#define NC_QLINMM_FXS8(AQ,BQ,CQ,M,N,K,SFXA,SFXB,ZA,ZB,SFXY,ZY,Q,ACCTYPE) \
{ \
ACCTYPE a0 = ((ACCTYPE) SFXA * (ACCTYPE) SFXB) / (ACCTYPE) SFXY; \
ACCTYPE a3 =  ((ACCTYPE)1 << Q); \
CUSTOM_PRAGMA(loopbound min 0 max M) \
for(int i=0; i<M; i++) { \
CUSTOM_PRAGMA(loopbound min 0 max N) \
    for(int j=0; j<N; j++) { \
        ACCTYPE acc = 0; \
CUSTOM_PRAGMA(loopbound min 0 max K) \
        for(int k=0; k<K; k++) { \
            acc += ((ACCTYPE) AQ[i*M+k] - ZA) * ((ACCTYPE) BQ[k*N+j] - ZB); \
        } \
        ACCTYPE acca0 = acc * a0; \
        ACCTYPE q1 = acca0 / a3; \
        ACCTYPE q2 = q1 + ZY; \
        NC_CLIP_SINT8(q2) \
        CQ[i*N+j] = (int8_t)q2; \
    } \
} \
}

/***************************************************
 * Macro: NC_QLMUL_FXS8
 * Description:
 *   Performs quantized element-wise multiplication
 *   between two int8 input arrays and produces an
 *   int8 output.
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
 *   Q       - Quantization shift
 *   ACCTYPE - Accumulator type (e.g., int32_t, int64_t)
 ***************************************************/

#define NC_QLMUL_FXS8(AQ,BQ,CQ,SIZE,SFXA,ZFXA,SFXB,ZFXB,SFXC,ZFXC,Q,ACCTYPE) \
{ \
ACCTYPE a0 = (ACCTYPE)SFXA * (ACCTYPE)SFXB / (ACCTYPE)SFXC; \
CUSTOM_PRAGMA(loopbound min 0 max SIZE) \
for(int i=0; i < SIZE; i++) { \
  ACCTYPE a1 = AQ[i] - ZFXA; \
  ACCTYPE a2 = BQ[i] - ZFXB; \
  ACCTYPE a3 = a0 * a1 * a2; \
  ACCTYPE a5 = a3 / ((ACCTYPE)1 << Q); \
  ACCTYPE a6 = a5 + ZFXC; \
  CQ[i] = (int8_t) a6; \
} \
}

/***************************************************
 * Macro: NC_QLPRELU_FXS8
 * Description:
 *   Applies quantized PReLU (Parametric ReLU) activation
 *   on an int8 input array and produces an int8 output.
 *
 * Parameters:
 *   X     - Input array (int8_t)
 *   W     - PReLU slope (int32_t)
 *   Y     - Output array (int8_t)
 *   SIZE  - Number of elements
 *   SFXX  - Scale factor for input (fixed point)
 *   ZFXX  - Zero-point for input
 *   SFXY  - Scale factor for output (fixed point)
 *   ZFXY  - Zero-point for output
 *   Q     - Quantization shift
 *   ACCTYPE - Accumulator type (e.g., int32_t, int64_t)
 ***************************************************/

#define NC_QLPRELU_FXS8(X,W,Y,SIZE,SFXX,ZFXX,SFXY,ZFXY,Q,ACCTYPE) \
{ \
CUSTOM_PRAGMA(loopbound min 0 max SIZE) \
for(int i=0; i<SIZE; i++) { \
  ACCTYPE a0; \
  if(X[i] >= ZFXX) \
    a0 = (ACCTYPE) (SFXX) * (X[i] - ZFXX) << Q; \
  else \
    a0 = (ACCTYPE) (W) * (ACCTYPE) (SFXX) * (X[i] - ZFXX); \
  ACCTYPE a1 = (ACCTYPE) (SFXY) << Q; \
  ACCTYPE a2 = (a0 / a1); \
  ACCTYPE a3 = a2 + ZFXY; \
  NC_CLIP_SINT8(a3); \
  Y[i] = (int8_t)(a3); \
} \
}

/***************************************************
 * Macro: NC_RELU_FXS8
 * Description:
 *   Applies quantized ReLU activation on an int8 input
 *   array and produces an int8 output.
 *
 * Parameters:
 *   X       - Input array (int8_t)
 *   Y       - Output array (int8_t)
 *   SIZE    - Number of elements
 *   SFXX    - Scale factor for input (fixed point)
 *   ZFXX    - Zero-point for input
 *   SFXY    - Scale factor for output (fixed point)
 *   ZFXY    - Zero-point for output
 *   ACCTYPE - Accumulator type (e.g., int32_t, int64_t)
 ***************************************************/

#define NC_RELU_FXS8(X,Y,SIZE,SFXX,ZFXX,SFXY,ZFXY, ACCTYPE) \
{ \
CUSTOM_PRAGMA(loopbound min 0 max SIZE) \
for(int i=0; i < SIZE; i++) { \
  if(X[i] >= ZFXX) { \
    ACCTYPE a0 = (ACCTYPE)SFXX * (X[i] - ZFXX); \
    ACCTYPE a1 = a0 / SFXY; \
    ACCTYPE a2 = a1 + ZFXY; \
    NC_CLIP_SINT8(a2); \
    Y[i] = (int8_t) (a2); \
  } \
  else { \
    Y[i] = (int8_t) ZFXY; \
  } \
} \
}

/***************************************************
 * Macro: NC_QSIGMOID_FXS8
 * Description:
 *   Applies quantized sigmoid activation using a
 *   lookup table (LUT) on an int8 input array.
 *
 * Parameters:
 *   X       - Input array (int8_t)
 *   Y       - Output array (int8_t)
 *   SIZE    - Number of elements
 *   SFXX    - Scale factor for input (fixed point)
 *   ZFXX    - Zero-point for input
 *   ACCTYPE - Accumulator type (e.g., int32_t, int64_t)
 ***************************************************/

#define NC_QSIGMOIDLUT_FXS8_SIZE (256)
#define NC_QSIGMOIDLUT_FXS8_STEP (1)
#define NC_QSIGMOIDLUT_FXS8_SFXLUTX (1542)
#define NC_QSIGMOIDLUT_FXS8_ZFXLUTX (0)
#define NC_QSIGMOIDLUT_FXS8_SFXLUTY (128)
#define NC_QSIGMOIDLUT_FXS8_ZFXLUTY (-129)
extern const int8_t NC_QSIGMOIDLUT_FXS8[NC_QSIGMOIDLUT_FXS8_SIZE];

#define NC_QSIGMOID_FXS8(X,Y,SIZE,SFXX,ZFXX,ACCTYPE) \
{ \
CUSTOM_PRAGMA(loopbound min 0 max SIZE) \
for(int i=0; i<SIZE; i++) { \
  ACCTYPE a0 = SFXX * (X[i] - ZFXX); \
  ACCTYPE lutx = (a0 / NC_QSIGMOIDLUT_FXS8_SFXLUTX) + NC_QSIGMOIDLUT_FXS8_ZFXLUTX; \
  int idxlut = ((int) lutx + (int) (128)) / NC_QSIGMOIDLUT_FXS8_STEP; \
  Y[i] = NC_QSIGMOIDLUT_FXS8[idxlut]; \
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