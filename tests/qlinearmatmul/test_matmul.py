import numpy as np
from tests.common.common import compute_s_z, compute_sfx_zfx, quantize_linear, dequantize_linear, quantize_linear_fixed_point, dequentize_linear_fixed_point
from tests.common.common import clear_compilation_folder, generate_main_c, run_bash_command, read_output
from config.config import TEST_TMP_PATH, LIB_DIR

def test_matmul():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    # compute floating point model
    y_float = matmul_float_model(a, b)
    
    # quantization data
    Q = 31
    QS = 20
    sa, za = compute_s_z(a)
    sb, zb = compute_s_z(b)
    sy, zy = compute_s_z(y_float)
    sfxa, zfxa = compute_sfx_zfx(sa, za, Q)
    sfxb, zfxb = compute_sfx_zfx(sb, zb, Q)
    sfxy, zfxy = compute_sfx_zfx(sy, zy, Q)

    # compute quantized model
    y_rec1 = matmul_quant_float(a, b, sa, za, sb, zb, sy, zy)
    assert np.allclose(y_rec1, y_float, atol=1e-1)

    # compute quantized model fixed point
    y_rec2 = matmul_quant_fixed_point(a, b, sfxa, zfxa, sfxb, zfxb, sfxy, zfxy, Q, QS)
    assert np.allclose(y_rec2, y_float, atol=3e-1)

    # compute quantized model fixed point c
    y_rec3 = matmul_quant_fixed_point_c(a, b, 2, 2, 2, sfxa, za, sfxb, zb, sfxy, zy, Q, QS)
    assert np.allclose(y_rec3, y_float, atol=3e-1)

def matmul_float_model(a: np.array, b: np.array) -> np.array:
    y = np.matmul(a, b)
    return y

def matmul_quant_float(a: np.array, b: np.array, sa: float, za: int, sb: float, zb: int, sy: float, zy: int) -> np.array:
    a_quant = quantize_linear(a, sa, za)
    b_quant = quantize_linear(b, sb, zb)

    a0 = (sa * sb) / sy
    a1 = a_quant - za
    a2 = b_quant - zb
    a3 = np.round(a0 * np.matmul(a1, a2) + zy)
    y_quant = np.clip(a3, -128, 127).astype(np.int8)

    y_rec = dequantize_linear(y_quant, sy, zy)

    return y_rec

def matmul_quant_fixed_point(a: np.array, b: np.array, sfxa: int, zfxa: int, sfxb: int, zfxb: int, sfxy: int, zfxy: int, Q: int, QS: int) -> np.array:
    a_quant = quantize_linear_fixed_point(a, sfxa, zfxa, Q)
    b_quant = quantize_linear_fixed_point(b, sfxb, zfxb, Q)

    a0 = np.int64(sfxa) * np.int64(sfxb)
    a1 = np.int64(a_quant) - np.int64(zfxa)
    a2 = np.int64(b_quant) - np.int64(zfxb)
    a3 = np.int64(sfxy) * (2**Q)
    a4 = np.matmul(a1, a2).astype(np.int64)
    
    a0 = a0 // (2**QS)
    a3 = a3 // (2**QS)
    a5 = (a0*a4) / a3
    
    a6 = np.round(a5)
    a7 = a6 + zfxy
    a8 = np.clip(a7, -128, 127).astype(np.int8)

    y_rec = dequentize_linear_fixed_point(a8, sfxy, zfxy, Q)

    return y_rec

def matmul_quant_fixed_point_c(a: np.array, b: np.array, M: int, N: int, K: int, sfxa: int, za: int, sfxb: int, zb: int, sfxc: int, zc: int, Q: int, QS: int):
    aq = quantize_linear_fixed_point(a, sfxa, za, Q)
    bq = quantize_linear_fixed_point(b, sfxb, zb, Q)

    aq = aq.reshape(M*K)
    bq = bq.reshape(K*N)
    aq_str = ",".join(map(str, aq))
    bq_str = ",".join(map(str, bq))

    size = len(aq)

    cname = "main"
    exename = "test"
    outname = "out"

    main = f"""
#include "ncast_lib.h"
#include <stdio.h>

int main() {{ 
    int8_t aq[{size}] = {{ {aq_str} }};
    int8_t bq[{size}] = {{ {bq_str} }};

    int8_t cq[{size}];
    NC_QLINMATMUL(aq,bq,cq,{M},{N},{K},{sfxa},{sfxb},{za},{zb},{sfxc},{zc},{Q},{QS})

    NC_OUTTNS("{TEST_TMP_PATH}/{outname}.txt",cq,{size},"%d");

    return 0;
}}
"""
    
    
    clear_compilation_folder()
    generate_main_c(main, cname)
    run_bash_command(f"gcc -o {TEST_TMP_PATH}/{exename} {TEST_TMP_PATH}/{cname}.c -I {LIB_DIR}")
    run_bash_command(f"{TEST_TMP_PATH}/{exename}")
    cq = read_output(f"{TEST_TMP_PATH}/{outname}.txt")
    c_rec = dequentize_linear_fixed_point(cq, sfxc, zc, Q)
    c_rec = c_rec.reshape(M,N)
    return c_rec
