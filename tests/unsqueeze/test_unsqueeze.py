import numpy as np
import torch
from tests.common.common import compute_s_z, compute_sfx_zfx, quantize_linear, dequantize_linear, quantize_linear_fixed_point, dequentize_linear_fixed_point
from tests.common.common import clear_compilation_folder, generate_main_c, run_bash_command, read_output
from config.config import TEST_TMP_PATH, LIB_DIR

def test_unsqueeze():
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)

    # compute torch model
    y_torch = unsqueeze_torch_float(x)

    # compute floating point model
    y_float = unsqueeze_float(x)
    assert np.allclose(y_torch, y_float, atol=1e-6)
    assert y_torch.shape == y_float.shape

    # compute quantization data
    Q = 31
    sx, zx = compute_s_z(x)
    sfxx, zfxx = compute_sfx_zfx(sx, zx, Q)

    # compute quantized model
    y_rec1 = unsqueeze_quant_float(x, sx, zx)
    assert np.allclose(y_rec1, y_torch, atol=1e-1)
    assert y_rec1.shape == y_torch.shape

    # compute quantized model fixed point
    y_rec2 = unsqueeze_quant_fixed_point(x, sfxx, zfxx)
    assert np.allclose(y_rec2, y_torch, atol=1e-1)
    assert y_rec2.shape == y_torch.shape

    # compute quantized model fixed point c
    y_rec3 = unsqueeze_quant_fixed_point_c(x, sfxx, zfxx, Q)
    assert np.allclose(y_rec3, y_torch, atol=1e-1)

def unsqueeze_torch_float(x: np.array) -> np.array:
    x_torch = torch.from_numpy(x)
    y_torch = torch.unsqueeze(x_torch, dim=0)
    y = y_torch.numpy()
    return y

def unsqueeze_float(x: np.array) -> np.array:
    y = np.expand_dims(x, axis=0)
    return y

def unsqueeze_quant_float(x: np.array, sx: float, zx: int) -> np.array:
    x_quant = quantize_linear(x, sx, zx)
    y_quant = np.expand_dims(x_quant, axis=0)
    y_rec = dequantize_linear(y_quant, sx, zx)
    return y_rec

def unsqueeze_quant_fixed_point(x: np.array, sfxx: int, zfxx: int) -> np.array:
    x_quant = quantize_linear_fixed_point(x, sfxx, zfxx, Q=31)
    y_quant = np.expand_dims(x_quant, axis=0)
    y_rec = dequentize_linear_fixed_point(y_quant, sfxx, zfxx, Q=31)
    return y_rec

def unsqueeze_quant_fixed_point_c(x: np.array, sfxx: int, zx: int, Q: int) -> np.array:
    xq = quantize_linear_fixed_point(x, sfxx, zx, Q)

    size = len(xq)
    xq_str = ",".join(map(str, xq))

    cname = "main"
    exename = "test"
    outname = "out"

    main = f"""
#include "ncast_lib.h"
#include <stdio.h>

int main() {{ 
    int8_t xq[{size}] = {{ {xq_str} }};

    int8_t yq[{size}];
    NC_UNSQUEEZE(xq,yq,{size});

    NC_OUTTNS("{TEST_TMP_PATH}/{outname}.txt",yq,{size},"%d");

    return 0;
}}
"""
    
    
    clear_compilation_folder()
    generate_main_c(main, cname)
    run_bash_command(f"gcc -o {TEST_TMP_PATH}/{exename} {TEST_TMP_PATH}/{cname}.c {LIB_DIR}/ncast_lib.c -I {LIB_DIR}")
    run_bash_command(f"{TEST_TMP_PATH}/{exename}")
    yq = read_output(f"{TEST_TMP_PATH}/{outname}.txt")
    y_rec = dequentize_linear_fixed_point(yq, sfxx, zx, Q)
    return y_rec