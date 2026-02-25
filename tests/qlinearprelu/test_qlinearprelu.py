import numpy as np
import torch 
from tests.common.common import compute_s_z, compute_sfx_zfx, quantize_linear, dequantize_linear, quantize_linear_fixed_point, dequentize_linear_fixed_point
from tests.common.common import clear_compilation_folder, generate_main_c, run_bash_command, read_output
from config.config import TEST_TMP_PATH, LIB_DIR

def test_prelu2():
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    w = np.array([0.25], dtype=np.float32)

    # compute torch model
    y_torch = prelu_torch_float(x, w)

    # compute floating point model
    y_float = prelu_float(x, w)
    assert np.allclose(y_torch, y_float, atol=1e-6)

    # compute quantization data
    Q = 15
    sx, zx = compute_s_z(x)
    sy, zy = compute_s_z(y_float)
    sfxx, zfxx = compute_sfx_zfx(sx, zx, Q)
    sfxy, zfxy = compute_sfx_zfx(sy, zy, Q)

    # compute quantized model
    y_rec1 = prelu_quant_float(x, w, sx, zx, sy, zy)
    assert np.allclose(y_rec1, y_torch, atol=1e-1)
    
    # compute quantized model fixed point
    y_rec2 = prelu_quant_fixed_point(x, w, sfxx, zfxx, sfxy, zfxy, Q)
    assert np.allclose(y_rec2, y_torch, atol=1e-1)

    # compute quantized model fixed point c
    y_rec3 = prelu_quant_fixed_point_c(x, w[0], sfxx, zx, sfxy, zy, Q)
    assert np.allclose(y_rec3, y_torch, atol=1e-1)

def prelu_torch_float(x: np.array, w: np.array) -> np.array:
    x_torch = torch.from_numpy(x)
    w_torch = torch.from_numpy(w)
    y_torch = torch.nn.functional.prelu(x_torch, w_torch)
    y = y_torch.numpy()
    return y

def prelu_float(x: np.array, w: np.array) -> np.array:
    y = np.where(x >= 0, x, w * x)
    return y

def prelu_quant_float(x: np.array, w: np.array, sx: float, zx: int, sy: float, zy: int) -> np.array:
    x_quant = quantize_linear(x, sx, zx)
    
    alpha = np.where(x_quant >= zx, 1.0, w[0])
    a0 = alpha * sx / sy
    a1 = x_quant - zx
    a2 = np.round(a0 * a1 + zy).astype(np.int8)
    y_quant = np.clip(a2, -128, 127).astype(np.int8)
    y_rec = dequantize_linear(y_quant, sy, zy)
    
    return y_rec

def prelu_quant_fixed_point(x: np.array, w: np.array, sfxx: int, zfxx: int, sfxy: int, zfxy: int, Q: int) -> np.array:
    x_quant = quantize_linear_fixed_point(x, sfxx, zfxx, Q)
    wfx = np.round(w * (2.**Q)).astype(np.int64)
    alpha = np.where(x_quant >= zfxx, (2**Q), wfx[0])
    a0 = np.int64(alpha) * np.int64(sfxx) * (x_quant - zfxx)
    a1 = np.int64(sfxy) * (2**Q)
    a2 = np.round(a0 / a1).astype(np.int64)
    a3 = a2 + zfxy
    y_quant = np.clip(a3, -128, 127).astype(np.int8)

    y_rec = dequentize_linear_fixed_point(y_quant, sfxy, zfxy, Q)

    return y_rec

def prelu_quant_fixed_point_c(x: np.array, w: np.float32, sfxx: int, zx: int, sfxy: int, zy: int, Q: int):
    xq = quantize_linear_fixed_point(x, sfxx, zx, Q)
    wfx = np.round(w * (2.**Q)).astype(np.int64)

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
    NC_QLPRELU_FXS8(xq,{wfx},yq,{size},{sfxx},{zx},{sfxy},{zy},{Q},{acctype});

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
