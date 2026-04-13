import numpy as np
import torch
from tests.common.common import compute_s_z, compute_sfx_zfx, quantize_linear, dequantize_linear, quantize_linear_fixed_point, dequentize_linear_fixed_point
from tests.common.common import clear_compilation_folder, generate_main_c, run_bash_command, read_output
from config.config import TEST_TMP_PATH, LIB_DIR

def test_tanh():
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    
    # compute torch model
    y_torch = tanh_torch_float(x)
    
    # compute floating point model
    y = tanh_float(x)
    assert np.allclose(y_torch, y, atol=1e-6)

    # compute quantization data
    Q = 15
    sx, zx = compute_s_z(x)
    sfxx, zfxx = compute_sfx_zfx(sx, zx, Q)
    LUT_MIN = -6
    LUT_MAX = 6
    LUT_SIZE = 256

    # compute quantized model
    y_rec1 = tanh_quant_float(x, sx, zx, LUT_MIN, LUT_MAX, LUT_SIZE)
    assert np.allclose(y_rec1, y_torch, atol=1e-1)

    # compute quantized model fixed point
    y_rec2 = tanh_quant_fixed_point(x, sfxx, zfxx, LUT_MIN, LUT_MAX, LUT_SIZE, Q)
    assert np.allclose(y_rec2, y_torch, atol=1e-1)

    # compute quantized model fixed point c
    y_rec3 = tanh_quant_fixed_point_c(x, sfxx, zx, LUT_MIN, LUT_MAX, LUT_SIZE, Q)
    assert np.allclose(y_rec3, y_torch, atol=1e-1)

def tanh_torch_float(a: np.array) -> np.array:
    a_torch = torch.from_numpy(a)
    b_torch = torch.tanh(a_torch)
    b = b_torch.numpy()
    return b

def tanh_float(a: np.array) -> np.array:
    b = np.tanh(a)
    return b

def tanh_quant_float(a: np.array, sa: float, za: int, LUT_MIN: int, LUT_MAX: int, LUT_SIZE: int) -> np.array:
    aq = quantize_linear(a, sa, za)
    
    LUTX = np.linspace(LUT_MIN, LUT_MAX, LUT_SIZE, dtype=np.float32)
    LUTY = np.tanh(LUTX)
    step = LUTX[1] - LUTX[0]

    slutx, zlutx = compute_s_z(LUTX)
    sluty, zluty = compute_s_z(LUTY)

    qlutx = quantize_linear(LUTX, slutx, zlutx)
    qluty = quantize_linear(LUTY, sluty, zluty)

    xq = ((sa / slutx) * (aq - za) + zlutx).astype(np.int8)

    lut_step = quantize_linear(step, slutx, 0)

    index = ((np.int32(xq)+128) / lut_step).astype(np.int32)

    yq = qluty[index]

    y_rec = dequantize_linear(yq, sluty, zluty)

    return y_rec

def tanh_quant_fixed_point(a: np.array, sfxa: int, zfxa: int, LUT_MIN: int, LUT_MAX: int, LUT_SIZE: int, Q: int) -> np.array:
    aq = quantize_linear_fixed_point(a, sfxa, zfxa, Q)
    LUTX = np.linspace(LUT_MIN, LUT_MAX, LUT_SIZE, dtype=np.float32)
    LUTY = np.tanh(LUTX)
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

    index = ((np.int32(xq)+128) / lut_step).astype(np.int32)

    yq = qluty[index]

    y_rec = dequentize_linear_fixed_point(yq, sfxluty, zfxluty, Q)

    return y_rec

def tanh_quant_fixed_point_c(x: np.array, sfxx: int, zx: int, LUT_MIN: int, LUT_MAX: int, LUT_SIZE: int, Q: int):
    xq = quantize_linear_fixed_point(x, sfxx, zx, Q)

    LUTX = np.linspace(LUT_MIN, LUT_MAX, LUT_SIZE, dtype=np.float32)
    LUTY = np.tanh(LUTX)

    slutx, zlutx = compute_s_z(LUTX)
    sluty, zluty = compute_s_z(LUTY)
    sfxluty, _ = compute_sfx_zfx(sluty, zluty, Q)

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
    NC_QTANH_FXS8(xq,yq,{size},{sfxx},{zx},{acctype});

    NC_OUTTNS("{TEST_TMP_PATH}/{outname}.txt",yq,{size},"%d");

    return 0;
}}
"""
    
    
    clear_compilation_folder()
    generate_main_c(main, cname)
    run_bash_command(f"gcc -o {TEST_TMP_PATH}/{exename} {TEST_TMP_PATH}/{cname}.c {LIB_DIR}/ncast_lib.c -I {LIB_DIR}")
    run_bash_command(f"{TEST_TMP_PATH}/{exename}")
    yq = read_output(f"{TEST_TMP_PATH}/{outname}.txt")
    y_rec = dequentize_linear_fixed_point(yq, sfxluty, zluty, Q)
    return y_rec
