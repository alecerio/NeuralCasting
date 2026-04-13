
import numpy as np
import torch
from tests.common.common import compute_s_z, compute_sfx_zfx, quantize_linear, dequantize_linear, quantize_linear_fixed_point, dequentize_linear_fixed_point
from tests.common.common import clear_compilation_folder, generate_main_c, run_bash_command, read_output
from config.config import TEST_TMP_PATH, LIB_DIR

def test_transpose():
    x = np.array([[-1.0, -0.5], [0.2, 0.8], [0.0, 1.2]], dtype=np.float32)
    size = 1
    for s in x.shape:
        size = size * s

    # compute torch model
    y_torch = transpose_torch_float(x)

    # compute floating point model
    y_float = transpose_float(x)
    assert np.allclose(y_torch, y_float, atol=1e-6)
    assert y_torch.shape == y_float.shape

    # compute quantization data
    Q = 31
    sx, zx = compute_s_z(x)
    sfxx, zfxx = compute_sfx_zfx(sx, zx, Q)

    # compute quantized model
    y_rec1 = transpose_quant_float(x, sx, zx)
    assert np.allclose(y_rec1, y_torch, atol=1e-1)
    assert y_rec1.shape == y_torch.shape

    # compute quantized model fixed point
    y_rec2 = transpose_quant_fixed_point(x, sfxx, zfxx)
    assert np.allclose(y_rec2, y_torch, atol=1e-1)
    assert y_rec2.shape == y_torch.shape

    # compute quantized model fixed point c
    y_rec3 = transpose_quant_fixed_point_c(x, sfxx, zfxx, Q)
    assert np.allclose(y_rec3, y_torch.reshape(1,size).squeeze(), atol=1e-1) 

def transpose_torch_float(x: np.array) -> np.array:
    x_torch = torch.from_numpy(x)
    y_torch = x_torch.transpose(0, 1)
    y = y_torch.numpy()
    return y

def transpose_float(x: np.array) -> np.array:
    y = x.transpose(1, 0)
    return y

def transpose_quant_float(x: np.array, sx: float, zx: int) -> np.array:
    x_quant = quantize_linear(x, sx, zx)
    y_quant = x_quant.transpose(1, 0)
    y_rec = dequantize_linear(y_quant, sx, zx)
    return y_rec

def transpose_quant_fixed_point(x: np.array, sfxx: int, zfxx: int) -> np.array:
    x_quant = quantize_linear_fixed_point(x, sfxx, zfxx, Q=31)
    y_quant = x_quant.transpose(1, 0)
    y_rec = dequentize_linear_fixed_point(y_quant, sfxx, zfxx, Q=31)
    return y_rec

def transpose_quant_fixed_point_c(x: np.array, sfxx: int, zfxx: int, Q: int) ->np.array:
    aq = quantize_linear_fixed_point(x,sfxx,zfxx, Q)
    rows, cols = aq.shape
    l = 1
    for s in aq.shape:
        l = l*s
    aq = aq.reshape(1, l).squeeze()
    size = len(aq)
    aq_str = ",".join(map(str, aq))

    cname = "main"
    exename = "test"
    outname = "out"

    main = f"""
#include "ncast_lib.h"
#include <stdio.h>

int main() {{ 
    int8_t aq[{size}] = {{ {aq_str} }};

    int8_t cq[{size}];
    NC_TR2D(aq,cq,{cols},{rows});

    NC_OUTTNS("{TEST_TMP_PATH}/{outname}.txt",cq,{size},"%d");

    return 0;
}}
"""
    
    clear_compilation_folder()
    generate_main_c(main, cname)
    run_bash_command(f"gcc -o {TEST_TMP_PATH}/{exename} {TEST_TMP_PATH}/{cname}.c -I {LIB_DIR}")
    run_bash_command(f"{TEST_TMP_PATH}/{exename}")
    cq = read_output(f"{TEST_TMP_PATH}/{outname}.txt")
    c_rec = dequentize_linear_fixed_point(cq, sfxx, zfxx, Q)
    return c_rec
    
