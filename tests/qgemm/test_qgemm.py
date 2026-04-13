import numpy as np
import torch
import torch.nn as nn
from tests.common.common import compute_s_z, compute_sfx_zfx, quantize_linear, dequantize_linear, quantize_linear_fixed_point, dequentize_linear_fixed_point
from tests.common.common import clear_compilation_folder, generate_main_c, run_bash_command, read_output
from config.config import TEST_TMP_PATH, LIB_DIR

def test_qgemm():
    # seed
    torch.manual_seed(42)
    
    # gemm parameters
    IN = 4
    OUT = 3
    gemm = nn.Linear(in_features=IN, out_features=OUT, bias=True)
    W = gemm.weight.detach().cpu().numpy()
    B = gemm.bias.detach().cpu().numpy()

    # input
    x = torch.randn(IN)
    xnp = x.detach().numpy()

    M = W.shape[-2]
    K = W.shape[-1]
    N = 1

    # torch model
    y_torch = gemm_torch_float(gemm, x)
    
    # floating point model
    y_float = gemm_float(xnp, W, B)
    assert np.allclose(y_torch.detach().numpy(), y_float, atol=1e-6)

    # quantization data
    Q = 15
    QS = 0
    sx, zx = compute_s_z(xnp)
    sw, zw = compute_s_z(W)
    sb, zb = compute_s_z(B)
    sy, zy = compute_s_z(y_float)
    sfx, zfx = compute_sfx_zfx(sx, zx, Q)
    sfxw, zfxw = compute_sfx_zfx(sw, zw, Q)
    sfxb, zfxb = compute_sfx_zfx(sb, zb, Q)
    sfxy, zfxy = compute_sfx_zfx(sy, zy, Q)

    # quant model (float)
    y_rec = gemm_quant_float(xnp, W, B, sx, zx, sw, zw, sb, zb, sy, zy)
    assert np.allclose(y_rec, y_float, atol=1e-1)

    # quant model (fixed point)
    y_rec1 = gemm_quant_fixed_point(xnp, W, B, sfx, zfx, sfxw, zfxw, sfxb, zfxb, sfxy, zfxy, Q)
    assert np.allclose(y_rec1, y_float, atol=1e-1)

    # quant model c (fixed point)
    y_rec2 = gemm_quant_fixed_point_c(xnp, W, B, M, N, K, sfx, zx, sfxw, zw, sfxb, zb, sfxy, zy, Q)
    assert np.allclose(y_rec2, y_float, atol=1e-1)

def gemm_torch_float(gemm, x):
    return gemm(x)

def gemm_float(x: np.array, W: np.array, B: np.array) -> np.array:
    return W @ x + B

def gemm_quant_float(x: np.array, W: np.array, B: np.array, sx: float, zx: int, sw: float, zw: int, sb: float, zb: int, sy: float, zy: int) -> np.array:
    W_quant = quantize_linear(W, sw, zw)
    x_quant = quantize_linear(x, sx, zx)
    B_quant = quantize_linear(B, sb, zb)

    a0 = (sw * sx) / sy
    a1 = W_quant - zw
    a2 = x_quant - zx
    a3 = (sb / sy)
    a4 = B_quant - zb
    a5 = np.matmul(a1, a2)
    a6 = a0 * a5
    a7 = a3 * a4
    a3 = np.round(a6 + a7 + zy)
    y_quant = np.clip(a3, -128, 127).astype(np.int8)

    y_rec = dequantize_linear(y_quant, sy, zy)

    return y_rec

def gemm_quant_fixed_point(x: np.array, W: np.array, B: np.array,
                           sfxx: int, zfxx: int,
                           sfxw: int, zfxw: int,
                           sfxb: int, zfxb: int,
                           sfxy: int, zfxy: int,
                           Q: int) -> np.array:
    W_quant = quantize_linear_fixed_point(W, sfxw, zfxw, Q)
    x_quant = quantize_linear_fixed_point(x, sfxx, zfxx, Q)
    B_quant = quantize_linear_fixed_point(B, sfxb, zfxb, Q)

    a0 = np.int32(sfxw) * np.int32(sfxx)
    a1 = np.int32(W_quant) - np.int32(zfxw)
    a2 = np.int32(x_quant) - np.int32(zfxx)
    a3 = np.int32(B_quant) - np.int32(zfxb)

    a4 = np.matmul(a1, a2).astype(np.int32)

    a5 = np.int32(sfxy) * (2**Q)
    a6 = (a0 * a4) // a5
    
    a7 = (np.int32(sfxb) * a3) // np.int32(sfxy)

    a8 = a6 + a7 + np.int32(zfxy)
    a9 = np.clip(a8, -128, 127).astype(np.int8)

    y_rec = dequentize_linear_fixed_point(a9, sfxy, zfxy, Q)

    return y_rec

def gemm_quant_fixed_point_c(
        x: np.array, W: np.array, B: np.array, 
        M: int, N: int, K: int, 
        sfxx: int, zx: int, 
        sfxw: int, zw: int, 
        sfxb: int, zb: int, 
        sfxy: int, zy: int,
        Q: int):
    xq = quantize_linear_fixed_point(x, sfxx, zx, Q)
    Wq = quantize_linear_fixed_point(W, sfxw, zw, Q)
    Bq = quantize_linear_fixed_point(B, sfxb, zb, Q)

    Wq = Wq.reshape(M*K)
    xq = xq.reshape(K*N)
    Bq = Bq.reshape(M)
    Wq_str = ",".join(map(str, Wq))
    xq_str = ",".join(map(str, xq))
    Bq_str = ",".join(map(str, Bq))

    size_w = M*K
    size_x = K*N
    size_b = M
    size_y = M*N

    cname = "main"
    exename = "test"
    outname = "out"
    acctype = "int32_t"

    main = f"""
#include "ncast_lib.h"
#include <stdio.h>

int main() {{ 
    int8_t Wq[{size_w}] = {{ {Wq_str} }};
    int8_t Bq[{size_b}] = {{ {Bq_str} }};
    int8_t xq[{size_x}] = {{ {xq_str} }};

    int8_t yq[{size_y}];
    NC_QGEMM_FXS8(Wq,xq,yq,Bq,{M},{N},{K},{sfxw},{sfxx},{sfxb},{zw},{zx},{zb},{sfxy},{zy},{Q},{acctype})

    NC_OUTTNS("{TEST_TMP_PATH}/{outname}.txt",yq,{size_y},"%d");

    return 0;
}}
"""
    
    
    clear_compilation_folder()
    generate_main_c(main, cname)
    run_bash_command(f"gcc -o {TEST_TMP_PATH}/{exename} {TEST_TMP_PATH}/{cname}.c -I {LIB_DIR}")
    run_bash_command(f"{TEST_TMP_PATH}/{exename}")
    yq = read_output(f"{TEST_TMP_PATH}/{outname}.txt")
    y_rec = dequentize_linear_fixed_point(yq, sfxy, zy, Q)
    y_rec = y_rec.reshape(M,N)
    return y_rec