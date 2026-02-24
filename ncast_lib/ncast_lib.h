#ifndef NCAST_LIB_H
#define NCAST_LIB_H

#include <stdint.h>
//#include <stdlib.h>
//#include <inttypes.h>
//#include <stdio.h>
//#include <string.h>

#define CUSTOM_PRAGMA(x) _Pragma(#x)

#define NCAST_ABS(INPUT,OUTPUT,SIZE) \
    do { \
        for (size_t i = 0; i < SIZE; i++) { \
            OUTPUT[i] = (INPUT[i] < 0) ? -INPUT[i] : INPUT[i]; \
        } \
    } while (0)

#define NCAST_BATCH_NORM(X, Y, SIZE, COUT, SCALE, B, MEAN, VAR, BN_EPS)         \
  do {                                                                    \
    const int _C = (int)(COUT);                                           \
    const int _SIZE = (int)(SIZE);                                        \
    const int _L = (_C > 0) ? (_SIZE / _C) : 0;                           \
                                                                          \
    if (_C <= 0 || _L <= 0 || (_L * _C) != _SIZE) break;                  \
                                                                          \
    for (int _c = 0; _c < _C; ++_c) {                                     \
      const float _scale = (SCALE)[_c];                                   \
      const float _bias  = (B)[_c];                                       \
      const float _mean  = (MEAN)[_c];                                    \
      const float _invstd = 1.0f / sqrtf((VAR)[_c] + (float)BN_EPS);       \
                                                                          \
      const int _base = _c * _L;                                          \
      for (int _l = 0; _l < _L; ++_l) {                                   \
        const int _i = _base + _l;                                        \
        (Y)[_i] = ((X)[_i] - _mean) * _invstd * _scale + _bias;           \
      }                                                                   \
    }                                                                     \
  } while (0)

#define NCAST_DEQUANTIZE_LIN(X, Y, SIZE, SCALE, ZERO)      \
  do {                                                     \
    for (int _i = 0; _i < (int)(SIZE); ++_i) {             \
      (Y)[_i] = ((float)(X)[_i] - (float)(ZERO))           \
                * (float)(SCALE);                          \
    }                                                      \
  } while (0)

#define NCAST_PRELU(X, Y, SIZE, SLOPE)                    \
  do {                                             \
    const int _N = (int)(SIZE);                    \
    const float _s = (float)(SLOPE);               \
    for (int _i = 0; _i < _N; ++_i) {               \
      const float _x = (X)[_i];                    \
      (Y)[_i] = (_x >= 0.0f) ? _x : (_s * _x);     \
    }                                              \
  } while (0)

#define NCAST_QLINEAR_ADD(A, B, C, SIZE, SCALE_A, ZERO_A, SCALE_B, ZERO_B, SCALE_C, ZERO_C) \
  do {                                                                                   \
    const int _N = (int)(SIZE);                                                          \
    const float _sa = (float)(SCALE_A);                                                  \
    const float _sb = (float)(SCALE_B);                                                  \
    const float _sc = (float)(SCALE_C);                                                  \
    const int _za = (int)(ZERO_A);                                                       \
    const int _zb = (int)(ZERO_B);                                                       \
    const int _zc = (int)(ZERO_C);                                                       \
                                                                                         \
    for (int _i = 0; _i < _N; ++_i) {                                                    \
      const float _fa = ((float)(A)[_i] - (float)_za) * _sa;                              \
      const float _fb = ((float)(B)[_i] - (float)_zb) * _sb;                              \
      const float _fc = _fa + _fb;                                                       \
                                                                                         \
      int _q = (int)lroundf(_fc / _sc) + _zc;                                            \
      _q = NCAST_SAT_S8(_q);                                                                   \
      (C)[_i] = (int8_t)_q;                                                              \
    }                                                                                    \
  } while (0)

#define NCAST_QLINEAR_MUL(A, B, C, SIZE, SCALE_A, ZERO_A, SCALE_B, ZERO_B, SCALE_C, ZERO_C) \
  do {                                                                                   \
    const int _N = (int)(SIZE);                                                          \
    const float _sa = (float)(SCALE_A);                                                  \
    const float _sb = (float)(SCALE_B);                                                  \
    const float _sc = (float)(SCALE_C);                                                  \
    const int _za = (int)(ZERO_A);                                                       \
    const int _zb = (int)(ZERO_B);                                                       \
    const int _zc = (int)(ZERO_C);                                                       \
                                                                                         \
    for (int _i = 0; _i < _N; ++_i) {                                                    \
      const float _fa = ((float)(A)[_i] - (float)_za) * _sa;                              \
      const float _fb = ((float)(B)[_i] - (float)_zb) * _sb;                              \
      const float _fc = _fa * _fb;                                                       \
      int _q = (int)lroundf(_fc / _sc) + _zc;                                            \
      _q = NCAST_SAT_S8(_q);                                                                   \
      (C)[_i] = (int8_t)_q;                                                              \
    }                                                                                    \
  } while (0)

#define NCAST_QLINEAR_SIGMOID(X, Y, SIZE, SCALE_X, ZERO_X, SCALE_Y, ZERO_Y)    \
  do {                                                                      \
    const int _N = (int)(SIZE);                                             \
    const float _sx = (float)(SCALE_X);                                     \
    const float _sy = (float)(SCALE_Y);                                     \
    const int _zx = (int)(ZERO_X);                                          \
    const int _zy = (int)(ZERO_Y);                                          \
    for (int _i = 0; _i < _N; ++_i) {                                       \
      const float _x = ((float)(X)[_i] - (float)_zx) * _sx;                 \
      const float _y = 1.0f / (1.0f + expf(-_x));                           \
      int _q = (int)lroundf(_y / _sy) + _zy;                                \
      _q = NCAST_SAT_S8(_q);                                                      \
      (Y)[_i] = (int8_t)_q;                                                 \
    }                                                                       \
  } while (0)

#define NCAST_QUANTIZE_LINEAR(X, Y, SIZE, SCALE_Y, ZERO_Y)              \
  do {                                                               \
    const int _N = (int)(SIZE);                                      \
    const float _sy = (float)(SCALE_Y);                              \
    const int _zy = (int)(ZERO_Y);                                   \
                                                                     \
    for (int _i = 0; _i < _N; ++_i) {                                \
      int _q = (int)lroundf(((float)(X)[_i]) / _sy) + _zy;           \
      _q = NCAST_SAT_S8(_q);                                               \
      (Y)[_i] = (int8_t)_q;                                          \
    }                                                                \
  } while (0)

#define NCAST_QLINEAR_CONV(X, W, B, Y, CIN, COUT, LIN,                                   \
                     SCALE_X, ZERO_X, SCALE_W, ZERO_W, SCALE_Y, ZERO_Y,            \
                     DIL, GROUP, KERNEL, PADS, STRIDES)                            \
  do {                                                                             \
    const int _CIN  = (int)(CIN);                                                  \
    const int _COUT = (int)(COUT);                                                 \
    const int _LIN  = (int)(LIN);                                                  \
                                                                                   \
    const int _G  = (int)(GROUP);                                                  \
    const int _K  = (int)((KERNEL)[0]);                                            \
    const int _d  = (int)((DIL)[0]);                                               \
    const int _s  = (int)((STRIDES)[0]);                                           \
    const int _pl = (int)((PADS)[0]);                                              \
    const int _pr = (int)((PADS)[1]);                                              \
                                                                                   \
    /* Lout = floor((Lin + pl+pr - (d*(K-1)+1))/s) + 1 */                          \
    const int _Keff = _d * (_K - 1) + 1;                                           \
    const int _LOUT = (_LIN + _pl + _pr - _Keff) / _s + 1;                         \
                                                                                   \
    /* group partition */                                                          \
    const int _CIN_G  = (_G > 0) ? (_CIN / _G) : 0;                                \
    const int _COUT_G = (_G > 0) ? (_COUT / _G) : 0;                               \
                                                                                   \
    const int _zx = (int)(ZERO_X);                                                 \
    const int _zw = (int)(ZERO_W);                                                 \
    const int _zy = (int)(ZERO_Y);                                                 \
                                                                                   \
    /* requant multiplier */                                                       \
    const float _m = ((float)(SCALE_X) * (float)(SCALE_W)) / (float)(SCALE_Y);     \
                                                                                   \
    /* basic validity guard (avoids UB if misconfigured) */                        \
    if (_G <= 0 || _CIN_G <= 0 || _COUT_G <= 0) break;                             \
    if ((_CIN_G * _G) != _CIN || (_COUT_G * _G) != _COUT) break;                   \
    if (_K <= 0 || _s <= 0 || _LOUT <= 0) break;                                   \
                                                                                   \
    for (int _co = 0; _co < _COUT; ++_co) {                                        \
      const int _g = _co / _COUT_G;                                                \
      const int _ci0 = _g * _CIN_G;                                                \
      const int _w_base = _co * (_CIN_G * _K);                                     \
      const int32_t _bias = (B) ? (int32_t)(B)[_co] : 0;                           \
                                                                                   \
      for (int _yo = 0; _yo < _LOUT; ++_yo) {                                      \
        int64_t _acc = (int64_t)_bias;                                             \
        const int _x_origin = _yo * _s - _pl;                                      \
                                                                                   \
        for (int _cig = 0; _cig < _CIN_G; ++_cig) {                                \
          const int _ci = _ci0 + _cig;                                             \
          const int _x_base = _ci * _LIN;                                          \
          const int _w_c_base = _w_base + _cig * _K;                               \
                                                                                   \
          for (int _k = 0; _k < _K; ++_k) {                                        \
            const int _xi = _x_origin + _k * _d;                                   \
            if ((unsigned)_xi < (unsigned)_LIN) {                                 \
              const int _xq = (int)((int8_t)(X)[_x_base + _xi]) - _zx;             \
              const int _wq = (int)((int8_t)(W)[_w_c_base + _k]) - _zw;            \
              _acc += (int64_t)_xq * (int64_t)_wq;                                 \
            }                                                                      \
          }                                                                        \
        }                                                                          \
                                                                                   \
        int _q = (int)lroundf((float)_acc * _m) + _zy;                             \
        _q = NCAST_SAT_S8(_q);                                                           \
        (Y)[_co * _LOUT + _yo] = (int8_t)_q;                                       \
      }                                                                            \
    }                                                                              \
  } while (0)

#define NCAST_SAT_S8(x) ((x) < -128 ? -128 : ((x) > 127 ? 127 : (x)))

#define NCAST_QUANT8(W, WQ, SIZE, S, Z) \
do { \
    for(int i=0; i<SIZE; i++) { \
        int32_t temp = _NCAST_ROUND((W[i]/S)+Z); \
        _NCAST_CLIP_INT8(temp) \
        WQ[i] = (uint8_t) temp; \
    } \
} while(0)

#define _NCAST_CLIP_INT8(X) \
do { \
    if(X < INT8_MIN) \
        X = INT8_MIN; \
    else if(X >= INT8_MAX-1) \
        X = INT8_MAX-1;  \
} while(0)

#define _NCAST_ROUND(X) \
do { \
    (X >= 0) ? (int32_t)(X + 0.5) : (int32_t)(X - 0.5); \
} while(0)

// -------------------------------------------------------------------
// -------------------------------------------------------------------

#define NC_OUTTNS(OUTFILENAME,X,SIZE,DTYPE) \
do { \
  FILE *_UI(f) = fopen(OUTFILENAME, "w"); \
  if (!_UI(f)) return -1; \
  for (size_t _UI(i) = 0; _UI(i) < SIZE; _UI(i)++) { \
    if (fprintf(_UI(f), DTYPE, X[_UI(i)]) < 0) { \
      fclose(_UI(f)); \
      return -2; \
    } \
    if (_UI(i) + 1 < SIZE) { \
      fputc(',', _UI(f)); \
    } \
  } \
  fclose(_UI(f)); \
  return 0; \
} while(0)

#define NC_PRINTTNSINT(X,START,END,DTYPE) \
do { \
  for(int i=START; i<=END; i++) { \
    printf("%" DTYPE "", X[i]); \
    if(i < END) \
      printf(","); \
  } \
  printf("\n"); \
} while(0)

#define NC_PRINTTNSFLOAT(X,START,END) \
do { \
  for(int i=START; i<=END; i++) { \
    printf("%f", X[i]); \
    if(i < END) \
      printf(","); \
  } \
  printf("\n"); \
} while(0)


#define NC_MAX(X,SIZE,MAX) \
do { \
  MAX = X[0]; \
  for(int i=1; i<SIZE; i++) { \
    if(X[i] > MAX) \
      MAX = X[i]; \
  } \
} while(0)

#define NC_MIN(X,SIZE,MIN) \
do { \
  MIN = X[0]; \
  for(int i=0; i<SIZE; i++) { \
    if(X[i] < MIN) \
      MIN = X[i]; \
  } \
} while(0)

#define NC_MIN_MAX(X,SIZE,MIN,MAX) \
do { \
  MIN = X[0]; \
  MAX = X[0]; \
  for(int i=0; i<SIZE; i++) { \
    if(X[i] < MIN) \
      MIN = X[i]; \
    if(X[i] > MAX) \
      MAX = X[i]; \
  } \
} while(0)

#define NC_ROUND(X) \
do { \
  float integer = (float)((int)X); \
  float frac = X - integer; \
  if (frac >= 0.5f) \
    X = integer + 1.00001f; \
  else \
    X = integer; \
} while(0)

#define NC_SZ_SINT8(X,SIZE,S,Z) \
do { \
  float maxv, minv; \
  NC_MIN_MAX(X,SIZE,minv,maxv); \
  S = (maxv - minv) / 255.0f; \
  float f0 = (float)INT8_MIN - (minv / S); \
  NC_ROUND(f0); \
  Z = (int32_t) f0; \
} while(0)

#define NC_SFX_SINT8(X,SIZE,S,SFX,Q) \
do { \
  float f0 = S * (float)((int64_t)(1) << Q); \
  NC_ROUND(f0); \
  SFX = (int64_t) f0; \
} while(0)

#define CONCAT(a,b) a##b
#define _UI(name) CONCAT(name,__COUNTER__)

#define NC_QLIN_FXS8(X,Y,SIZE,SFX,ZFX,Q) \
do { \
    int64_t _UI(zfx) = (int64_t)ZFX; \
    int64_t _UI(sfx) = (int64_t)SFX; \
    int64_t _UI(a1) = _UI(zfx) * _UI(sfx); \
    for(int _UI(i)=0; _UI(i)<SIZE; _UI(i)++) { \
      float _UI(f0) = X[_UI(i)] * (float)((int64_t)(1) << Q); \
      int64_t _UI(a0) = (int64_t) _UI(f0); \
      int64_t _UI(a2) = _UI(a0) + _UI(a1); \
      int64_t _UI(a4) = _UI(a2) / _UI(sfx); \
      NC_CLIP_SINT8(_UI(a4)); \
      int8_t _UI(a5) = (int8_t)_UI(a4); \
      Y[_UI(i)] = _UI(a5); \
    } \
} while(0)

#define NC_DQLIN_FXS8(X,Y,SIZE,SFX,ZFX,Q) \
do { \
  int64_t _UI(sfx) = (int64_t) SFX; \
  int64_t _UI(zfx) = (int64_t) ZFX; \
  for (int _UI(i)=0; _UI(i)<SIZE; _UI(i)++) { \
    int64_t _UI(a0) = (int64_t) X[_UI(i)] - _UI(zfx); \
    int64_t _UI(a1) = (int64_t)_UI(sfx) * _UI(a0); \
    float _UI(a2) = (float)_UI(a1) / (float)((int64_t)1 << Q); \
    Y[_UI(i)] = _UI(a2); \
  } \
} while(0)

#define NC_CLIP_SINT8(X) \
do { \
    if(X < INT8_MIN) \
        X = INT8_MIN; \
    else if(X > INT8_MAX) \
        X = INT8_MAX;  \
} while(0)

#define NC_QLADD_FXS8(AQ,BQ,CQ,SIZE,SFXA,ZFXA,SFXB,ZFXB,SFXC,ZFXC) \
{ \
int64_t a4 = (int64_t)SFXC * ZFXC; \
CUSTOM_PRAGMA(loopbound min 0 max SIZE) \
for(int i=0; i<SIZE; i++) { \
  int64_t a0 = AQ[i] - (int64_t)ZFXA; \
  int64_t a2 = (int64_t)SFXA * a0; \
  int64_t a1 = BQ[i] - (int64_t)ZFXB; \
  int64_t a3 = (int64_t)SFXB * a1; \
  int64_t a5 = a2 + a3 + a4; \
  int64_t a6 = (int32_t) a5  / (int32_t) SFXC; \
  CQ[i] = (int8_t) a6; \
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

#define NC_QLINCONV_FXS8(XQ,WQ,BQ,YQ,KS,CIN,LIN,COUT,LOUT,PAD,DIL,STR,SFXX,ZX,SFXW,ZW,SFXY,ZY,SFXB,ZB,Q) \
{ \
CUSTOM_PRAGMA(loopbound min 0 max COUT) \
for(int o=0; o < COUT; o++) { \
  int64_t n0 = (int64_t)SFXB * (BQ[o] - ZB); \
  int64_t d0 = (int64_t)SFXY; \
  int64_t a0, r0; IDIVMOD64(n0, d0, a0, r0) \
CUSTOM_PRAGMA(loopbound min 0 max LOUT) \
  for(int t=0; t < LOUT; t++) { \
    int64_t acc = 0; \
CUSTOM_PRAGMA(loopbound min 0 max CIN) \
    for(int c=0; c < CIN; c++) { \
CUSTOM_PRAGMA(loopbound min 0 max KS) \
        for(int k=0; k < KS; k++) { \
            acc += (WQ[o*CIN*KS + c*KS + k] - ZW) * (XQ[c*LIN + t+k-PAD]-ZX); \
        } \
        int64_t t0 = (int64_t)SFXW * (int64_t)SFXX * acc; \
        int64_t t1 = (int64_t)SFXY << Q; \
        int64_t a1, r1; IDIVMOD64(t0, t1, a1, r1) \
        int64_t a2 = a0 + a1 + ZY; \
        YQ[o*LOUT+t] = (int8_t) a2; \
    } \
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