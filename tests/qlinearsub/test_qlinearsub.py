import numpy as np
from tests.common.common import compute_s_z, compute_sfx_zfx, quantize_linear, dequantize_linear, quantize_linear_fixed_point, dequentize_linear_fixed_point
from tests.common.common import clear_compilation_folder, generate_main_c, run_bash_command, read_output
from config.config import TEST_TMP_PATH, LIB_DIR

def test_linearsub():
    # compute floating point model
    a_float = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.8], dtype=np.float32)
    b_float = np.array([-2.1, -0.6, -0.3, 0.0, 0.3, 0.6], dtype=np.float32)
    c_float = linearsub_float(a_float, b_float)
    
    # compute quantization data
    Q = 31
    sa, za = compute_s_z(a_float)
    sb, zb = compute_s_z(b_float)
    sc, zc = compute_s_z(c_float)
    sfxa, zfxa = compute_sfx_zfx(sa, za, Q)
    sfxb, zfxb = compute_sfx_zfx(sb, zb, Q)
    sfxc, zfxc = compute_sfx_zfx(sc, zc, Q)

    # compute quantized model
    c_rec1 = linearsub_quant_float(a_float, b_float, sa, za, sb, zb, sc, zc)
    assert np.allclose(c_rec1, c_float, atol=1e-1)

    # compute quantized model fixed point
    c_rec2 = linearsub_quant_fixed_point(a_float, b_float, sfxa, zfxa, sfxb, zfxb, sfxc, zfxc, Q)
    assert np.allclose(c_rec2, c_float, atol=1e-1)

    # compute quantized model fixed point c
    c_rec3 = linearsub_quant_fixed_point_c(a_float, b_float, sfxa, zfxa, sfxb, zfxb, sfxc, zfxc, Q)
    assert np.allclose(c_rec3, c_float, atol=1e-1)

def linearsub_float(a: np.array, b: np.array) -> np.array:
    c = np.subtract(a, b)
    return c

def linearsub_quant_float(a: np.array, b: np.array, sa: float, za: int, sb: float, zb: int, sc: float, zc: int) -> np.array:
    a_quant = quantize_linear(a, sa, za)
    b_quant = quantize_linear(b, sb, zb)
    
    a0 = sa / sc
    a1 = sb / sc
    a2 = a_quant-za
    a3 = b_quant-zb
    c_quant = np.round(a0 * a2 - a1 * a3 + zc).astype(np.int8)

    c_rec = dequantize_linear(c_quant, sc, zc)
    
    return c_rec

def linearsub_quant_fixed_point(a: np.array, b: np.array, sfxa: int, zfxa: int, sfxb: int, zfxb: int, sfxc: int, zfxc: int, Q: int) -> np.array:
    a_quant = quantize_linear_fixed_point(a, sfxa, zfxa, Q)
    b_quant = quantize_linear_fixed_point(b, sfxb, zfxb, Q)

    a0 = a_quant - zfxa
    a1 = b_quant - zfxb
    a2 = np.int64(sfxa) * a0
    a3 = np.int64(sfxb) * a1
    a4 = np.int64(sfxc) * zfxc
    a5 = a2 - a3 + a4
    a6 = np.floor(a5 / sfxc).astype(np.int8)

    c_rec = dequentize_linear_fixed_point(a6, sfxc, zfxc, Q)

    return c_rec

def linearsub_quant_fixed_point_c(a: np.array, b: np.array, sfxa: int, za: int, sfxb: int, zb: int, sfxc: int, zc: int, Q: int):
    aq = quantize_linear_fixed_point(a, sfxa, za, Q)
    bq = quantize_linear_fixed_point(b, sfxb, zb, Q)

    size = len(aq)
    aq_str = ",".join(map(str, aq))
    bq_str = ",".join(map(str, bq))

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
    NC_QLSUB_FXS8(aq,bq,cq,{size},{sfxa},{za},{sfxb},{zb},{sfxc},{zc});

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
