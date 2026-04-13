import numpy as np
import torch
from tests.common.common import compute_s_z, compute_sfx_zfx, quantize_linear, dequantize_linear, quantize_linear_fixed_point, dequentize_linear_fixed_point
from tests.common.common import clear_compilation_folder, generate_main_c, run_bash_command, read_output
from config.config import TEST_TMP_PATH, LIB_DIR

def test_qlinearsoftmax():
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    dim = 0
    x = x-np.max(x)

    # compute torch model
    y_torch = linearsoftmax_torch_float(x, dim)
    
    # compute floating point model
    y = linearsoftmax_float(x, dim)
    assert np.allclose(y_torch, y, atol=1e-6)

    # compute quantization data
    Q = 15
    sx, zx = compute_s_z(x)
    sy, zy = compute_s_z(y)
    sfxx, zfxx = compute_sfx_zfx(sx, zx, Q)
    sfxy, zfxy = compute_sfx_zfx(sy, zy, Q)
    LUT_MIN = -8
    LUT_MAX = 0
    LUT_SIZE = 256

    # compute quantized model
    y_rec1 = linearsoftmax_quant_float(x, sx, zx, sy, zy, LUT_MIN, LUT_MAX, LUT_SIZE)
    assert np.allclose(y_rec1, y_torch, atol=1e-1)

    # compute quantized model fixed point
    y_rec2 = linearsoftmax_quant_fixed_point(x, sfxx, zfxx, sfxy, zfxy, LUT_MIN, LUT_MAX, LUT_SIZE, Q)
    assert np.allclose(y_rec2, y_torch, atol=1e-1)

    # compute quantized model fixed point c
    y_rec3 = softmax_quant_fixed_point_c(x, sfxx, zx, sfxy, zy, Q, LUT_MIN, LUT_MAX, LUT_SIZE)
    assert np.allclose(y_rec3, y_torch, atol=1e-1)

def linearsoftmax_torch_float(a: np.array, dim: int) -> np.array:
    a_torch = torch.from_numpy(a)
    b_torch = torch.softmax(a_torch, dim)
    b = b_torch.numpy()
    return b

def linearsoftmax_float(a: np.array, dim: int) -> np.array:
    a = a - np.max(a, axis=dim, keepdims=True)
    e = np.exp(a)
    return e / np.sum(e, axis=dim, keepdims=True)

def linearsoftmax_quant_float(a: np.array, sa: float, za: int, sy, zy, LUT_MIN: int, LUT_MAX: int, LUT_SIZE: int) -> np.array:
    
    aq = quantize_linear(a, sa, za)
    
    LUTX = np.linspace(LUT_MIN, LUT_MAX, LUT_SIZE, dtype=np.float32)
    LUTY = np.exp(LUTX)
    step = LUTX[1] - LUTX[0]

    slutx, zlutx = compute_s_z(LUTX)
    sluty, zluty = compute_s_z(LUTY)

    qlutx = quantize_linear(LUTX, slutx, zlutx)
    qluty = quantize_linear(LUTY, sluty, zluty)

    xq = ((sa / slutx) * (aq - za) + zlutx).astype(np.int8)

    lut_step = quantize_linear(step, slutx, 0)
    if lut_step == 0:
        lut_step = 1

    index = ((np.int32(xq)+128) / lut_step).astype(np.int32)

    yq = qluty[index]

    a0 = yq - zluty
    a1 = np.sum(yq)
    a2 = len(yq) * zluty
    yyq = (a0 / (a1 - a2)) * (1 / sy) + zy

    yyrec = dequantize_linear(yyq, sy, zy)
    return yyrec

def linearsoftmax_quant_fixed_point(a: np.array, sfxa: int, zfxa: int, sfxy, zfxy, LUT_MIN: int, LUT_MAX: int, LUT_SIZE: int, Q: int) -> np.array:
    aq = quantize_linear_fixed_point(a, sfxa, zfxa, Q)

    LUTX = np.linspace(LUT_MIN, LUT_MAX, LUT_SIZE, dtype=np.float32)
    LUTY = np.exp(LUTX)
    step = LUTX[1] - LUTX[0]

    slutx, zlutx = compute_s_z(LUTX)
    sluty, zluty = compute_s_z(LUTY)
    sfxlutx, zfxlutx = compute_sfx_zfx(slutx, zlutx, Q)
    sfxluty, zfxluty = compute_sfx_zfx(sluty, zluty, Q)

    qlutx = quantize_linear_fixed_point(LUTX, sfxlutx, zfxlutx, Q)
    qluty = quantize_linear_fixed_point(LUTY, sfxluty, zfxluty, Q)

    a0 = np.int64(sfxa) * (aq - zfxa)
    xq = ((a0 / sfxlutx) + zfxlutx).astype(np.int64)

    lut_step = quantize_linear_fixed_point(step, sfxlutx, 0, Q)
    if lut_step == 0:
        lut_step = 1

    index = ((np.int32(xq)+128) / lut_step).astype(np.int32)

    yq = qluty[index]

    a0 = yq - zluty
    a1 = np.sum(yq)
    a2 = len(yq) * zluty

    a3 = a0 * (2**Q)
    a4 = (a1 - a2) * sfxy

    yyq = (a3 // a4) + zfxy

    yyrec = dequentize_linear_fixed_point(yyq, sfxy, zfxy, Q)
    return yyrec

def softmax_quant_fixed_point_c(x: np.array, sfxx: int, zx: int, sfxy, zy, Q: int, LUT_MIN, LUT_MAX: int, LUT_SIZE: int):
    xq = quantize_linear_fixed_point(x, sfxx, zx, Q)

    LUTX = np.linspace(LUT_MIN, LUT_MAX, LUT_SIZE, dtype=np.float32)
    LUTY = np.exp(LUTX)
    step = LUTX[1] - LUTX[0]

    slutx, zlutx = compute_s_z(LUTX)
    sluty, zluty = compute_s_z(LUTY)
    sfxlutx, zfxlutx = compute_sfx_zfx(slutx, zlutx, Q)
    sfxluty, zfxluty = compute_sfx_zfx(sluty, zluty, Q)

    size = len(xq)
    xq_str = ",".join(map(str, xq))

    cname = "main"
    exename = "test"
    outname = "out"
    acctype = "int32_t"

    main = f"""
#include "ncast_lib.h"
#include <stdio.h>

int main() {{ 
    int8_t xq[{size}] = {{ {xq_str} }};

    int8_t yq[{size}];
    NC_QLSMAX_FXS8(xq,yq,{size},{sfxx},{zx},{sfxy},{zy},{Q},{acctype})

    NC_OUTTNS("{TEST_TMP_PATH}/{outname}.txt",yq,{size},"%d");

    return 0;
}}
"""
    
    
    clear_compilation_folder()
    generate_main_c(main, cname)
    run_bash_command(f"gcc -o {TEST_TMP_PATH}/{exename} {TEST_TMP_PATH}/{cname}.c {LIB_DIR}/ncast_lib.c -I {LIB_DIR}")
    run_bash_command(f"{TEST_TMP_PATH}/{exename}")
    yq = read_output(f"{TEST_TMP_PATH}/{outname}.txt")
    y_rec = dequentize_linear_fixed_point(yq, sfxy, zy, Q)
    return y_rec
