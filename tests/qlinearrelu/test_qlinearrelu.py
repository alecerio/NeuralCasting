import numpy as np
from tests.common.common import compute_s_z, compute_sfx_zfx, quantize_linear, dequantize_linear, quantize_linear_fixed_point, dequentize_linear_fixed_point
from tests.common.common import clear_compilation_folder, generate_main_c, run_bash_command, read_output
from config.config import TEST_TMP_PATH, LIB_DIR

def test_qlinearrelu():
    x = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.5], dtype=np.float32)

    # compute floating point model
    y_float = qlinearrelu_float(x)

    # compute quantization data
    Q = 15
    sx, zx = compute_s_z(x)
    sy, zy = compute_s_z(y_float)
    sfxx, zfxx = compute_sfx_zfx(sx, zx, Q)
    sfxy, zfxy = compute_sfx_zfx(sy, zy, Q)

    # compute quantized model
    y_rec1 = qlinearrelu_quant_float(x, sx, zx, sy, zy)
    assert np.allclose(y_rec1, y_float, atol=1e-1)

    # compute quantized model fixed point
    y_rec2 = qlinearelu_quant_fixed_point(x, sfxx, zfxx, sfxy, zfxy, Q)
    assert np.allclose(y_rec2, y_float, atol=1e-1)

    # compute quantized model fixed point c
    y_rec3 = qlinearrelu_quant_fixed_point_c(x, sfxx, zfxx, sfxy, zy, Q)
    assert np.allclose(y_rec3, y_float, atol=1e-1)


def qlinearrelu_float(a: np.array) -> np.array:
    y = np.maximum(0.0, a)
    return y

def qlinearrelu_quant_float(x: np.array, sx: float, zx: int, sy: float, zy: int) -> np.array:
    x_quant = quantize_linear(x, sx, zx)
    
    alpha = np.where(x_quant >= zx, 1.0, 0.0)
    a0 = alpha * sx / sy
    a1 = x_quant - zx
    a2 = np.round(a0 * a1 + zy).astype(np.int8)
    y_quant = np.clip(a2, -128, 127).astype(np.int8)
    y_rec = dequantize_linear(y_quant, sy, zy)

    return y_rec

def qlinearelu_quant_fixed_point(x: np.array, sfxx: int, zfxx: int, sfxy: int, zfxy: int, Q: int) -> np.array:
    x_quant = quantize_linear_fixed_point(x, sfxx, zfxx, Q)

    a3 = np.zeros_like(x_quant, dtype=np.int8)
    for i in range(len(x_quant)):
        if x_quant[i] >= zfxx:
            a0 = np.int64(sfxx) * (x_quant[i] - zfxx)
            a1 = np.round(a0 / sfxy).astype(np.int64)
            a2 = a1 + zfxy
            a3[i] = np.clip(a2, -128, 127).astype(np.int8)
        else:
            a3[i] = zfxy
    
    y_rec = dequentize_linear_fixed_point(a3, sfxy, zfxy, Q)


    return y_rec

def qlinearrelu_quant_fixed_point_c(x: np.array, sfxx: int, zx: int, sfxy: int, zy: int, Q: int):
    xq = quantize_linear_fixed_point(x, sfxx, zx, Q)

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
    NC_RELU_FXS8(xq,yq,{size},{sfxx},{zx},{sfxy},{zy},{acctype});

    NC_OUTTNS("{TEST_TMP_PATH}/{outname}.txt",yq,{size},"%d");

    return 0;
}}
"""
    
    
    clear_compilation_folder()
    generate_main_c(main, cname)
    run_bash_command(f"gcc -o {TEST_TMP_PATH}/{exename} {TEST_TMP_PATH}/{cname}.c -I {LIB_DIR}")
    run_bash_command(f"{TEST_TMP_PATH}/{exename}")
    cq = read_output(f"{TEST_TMP_PATH}/{outname}.txt")
    c_rec = dequentize_linear_fixed_point(cq, sfxy, zy, Q)
    return c_rec