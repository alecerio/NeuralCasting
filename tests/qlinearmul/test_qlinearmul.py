import numpy as np
from tests.common.common import compute_s_z, compute_sfx_zfx, quantize_linear, dequantize_linear, quantize_linear_fixed_point, dequentize_linear_fixed_point
from tests.common.common import clear_compilation_folder, generate_main_c, run_bash_command, read_output
from config.config import TEST_TMP_PATH, LIB_DIR

def test_qlinearmul():
    # compute floating point model
    a = np.array([-1.2, -1.1, 1.0, -0.1, 1.2], dtype=np.float32)
    b = np.array([-2.6, -1.3, 0.0, 3.3, 0.6], dtype=np.float32)
    c = linearmul_float(a, b)

    # compute quantization data
    Q = 15
    sa, za = compute_s_z(a)
    sb, zb = compute_s_z(b)
    sc, zc = compute_s_z(c)
    sfxa, zfxa = compute_sfx_zfx(sa, za, Q)
    sfxb, zfxb = compute_sfx_zfx(sb, zb, Q)
    sfxc, zfxc = compute_sfx_zfx(sc, zc, Q)

    # compute quantized model
    c_rec1 = linearmul_quant_float(a, b, sa, za, sb, zb, sc, zc)
    assert np.allclose(c_rec1, c, atol=2e0)

    # compute quantized model fixed point
    c_rec2 = linearmul_quant_fixed_point(a, b, sfxa, zfxa, sfxb, zfxb, sfxc, zfxc, Q)
    assert np.allclose(c_rec2, c, atol=2e0)

    #compute quantized model fixed point c
    c_rec3 = linearmul_quant_fixed_point_c(a, b, sfxa, zfxa, sfxb, zfxb, sfxc, zfxc, Q)
    assert np.allclose(c_rec3, c, atol=2e0)

def linearmul_float(a: np.array, b: np.array) -> np.array:
    c = np.multiply(a, b)
    return c

def linearmul_quant_float(a: np.array, b: np.array, sa: float, za: int, sb: float, zb: int, sc: float, zc: int) -> np.array:
    a_quant = quantize_linear(a, sa, za)
    b_quant = quantize_linear(b, sb, zb)

    a0 = sa * sb / sc
    a1 = a_quant - za
    a2 = b_quant - zb
    a3 = np.round(a0 * a1 * a2 + zc).astype(np.int8)
    c_quant = np.clip(a3, -128, 127).astype(np.int8)

    c_rec = dequantize_linear(c_quant, sc, zc)

    return c_rec

def linearmul_quant_fixed_point(a: np.array, b: np.array, sfxa: int, zfxa: int, sfxb: int, zfxb: int, sfxc: int, zfxc: int, Q: int) -> np.array:
    a_quant = quantize_linear_fixed_point(a, sfxa, zfxa, Q)
    b_quant = quantize_linear_fixed_point(b, sfxb, zfxb, Q)

    a0 = np.int64(sfxa) * np.int64(sfxb) * ((np.int64(a_quant) - np.int64(zfxa)) * (np.int64(b_quant) - np.int64(zfxb)))
    a1 = np.int64(sfxc) * (2**Q)
    a2 = np.round(a0 / a1)
    a3 = a2 + zfxc

    c_rec = dequentize_linear_fixed_point(a3, sfxc, zfxc, Q)
    return c_rec

def linearmul_quant_fixed_point_c(a: np.array, b: np.array, sfxa: int, za: int, sfxb: int, zb: int, sfxc: int, zc: int, Q: int):
    aq = quantize_linear_fixed_point(a, sfxa, za, Q)
    bq = quantize_linear_fixed_point(b, sfxb, zb, Q)

    size = len(aq)
    aq_str = ",".join(map(str, aq))
    bq_str = ",".join(map(str, bq))

    cname = "main"
    exename = "test"
    outname = "out"
    acctype = "int32_t"

    main = f"""
#include "ncast_lib.h"
#include <stdio.h>

int main() {{ 
    int8_t aq[{size}] = {{ {aq_str} }};
    int8_t bq[{size}] = {{ {bq_str} }};

    int8_t cq[{size}];
    NC_QLMUL_FXS8(aq,bq,cq,{size},{sfxa},{za},{sfxb},{zb},{sfxc},{zc},{Q},{acctype});

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
    return c_rec
