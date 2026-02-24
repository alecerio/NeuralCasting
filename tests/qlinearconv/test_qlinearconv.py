import torch
import torch.nn as nn
import numpy as np
from tests.common.common import compute_s_z, compute_sfx_zfx, quantize_linear, dequantize_linear, quantize_linear_fixed_point, dequentize_linear_fixed_point
from tests.common.common import clear_compilation_folder, generate_main_c, run_bash_command, read_output
from config.config import TEST_TMP_PATH, LIB_DIR

def test_qlinearconv():
    # seed
    torch.manual_seed(42)

    # convoltion parameters
    Cin = 4
    Cout = 3
    Ks = 1
    conv = nn.Conv1d(Cin, Cout, kernel_size=Ks, bias=True)
    W = conv.weight.detach().numpy()
    B = conv.bias.detach().numpy()
    P = 0 # padding
    D = 1 # dilation
    S = 1 # stride

    x = torch.randn(1, Cin, 5)
    xnp = x.detach().numpy()

    # convolution torch model
    y_torch = convolution_torch_float(x, conv)

    # convolution floating point model
    y_float = convolution_float(x, W, B, P, D, S)
    assert np.allclose(y_torch.detach().numpy(), y_float, atol=1e-6)

    # compute quantization data
    Q = 31
    sx, zx = compute_s_z(xnp)
    sw, zw = compute_s_z(W)
    sb, zb = compute_s_z(B)
    sy, zy = compute_s_z(y_float)
    sfx, zfx = compute_sfx_zfx(sx, zx, Q)
    sfxw, zfxw = compute_sfx_zfx(sw, zw, Q)
    sfxb, zfxb = compute_sfx_zfx(sb, zb, Q)
    sfxy, zfxy = compute_sfx_zfx(sy, zy, Q)

    # convolution quantized model
    y_rec1 = convolution_quant_float(xnp, W, B, sx, zx, sw, zw, sb, zb, sy, zy, P, D, S)
    assert np.allclose(y_torch.detach().numpy(), y_rec1, atol=1e-1)

    # convolution quantized model fixed point
    y_rec2 = convolution_quant_fixed_point(xnp, W, B, sfx, zfx, sfxw, zfxw, sfxb, zfxb, sfxy, zfxy, P, D, S, Q)
    assert np.allclose(y_torch.detach().numpy(), y_rec2, atol=1e-1)

    # convolution quantized model fixed point c
    y_rec3 = convolution_quant_fixed_point_c(xnp, W, B, sfx, zfx, sfxw, zw, sfxb, zb, sfxy, zy, P, D, S, Q)
    assert np.allclose(y_torch.detach().numpy(), y_rec3, atol=1e-1)

def convolution_torch_float(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    y = conv(x)
    return y

def convolution_float(x: np.array, W: np.array, B: np.array, P: int, D: int, S: int) -> np.array:
    Cin = W.shape[1]
    Cout = W.shape[0]
    Ks = W.shape[2]
    Lin = x.shape[2]

    Lout = np.floor((Lin + 2*P - D*(Ks-1) - 1) / S + 1).astype(int)

    y = np.zeros((Cout, Lout), dtype=np.float32)

    for o in range(Cout):
        for t in range(Lout):   
            acc = 0.0
            for c in range(Cin):
                for k in range(Ks):
                    acc += W[o,c,k] * x[0,c,t+k-P]
            y[o,t] = acc + B[o]

    return y

def convolution_quant_float(x: np.array, W: np.array, B: np.array, sx: float, zx: int, sw: float, zw: int, sb: float, zb: int, sy: float, zy: int, P: int, D: int, S: int) -> np.array:
    xq = quantize_linear(x, sx, zx)
    Wq = quantize_linear(W, sw, zw)
    Bq = quantize_linear(B, sb, zb)

    Cout, Cin, Ks = W.shape
    Lin = x.shape[2]
    Lout = np.floor((Lin + 2*P - D*(Ks-1) - 1) / S + 1).astype(int)

    yq = np.zeros((Cout, Lout), dtype=np.int8)

    for o in range(Cout):
        for t in range(Lout):
            acc = 0.0
            for c in range(Cin):
                for k in range(Ks):
                    acc += (Wq[o,c,k]-zw) * (xq[0,c,t+k-P]-zx)
            yq[o,t] = acc * (sx * sw) / sy + (sb / sy) * (Bq[o] - zb) + zy
    y_rec = dequantize_linear(yq, sy, zy)
    return y_rec

def convolution_quant_fixed_point(x: np.array, W: np.array, B: np.array, sfxx: int, zfxx: int, sfxw: int, zfxw: int, sfxb: int, zfxb: int, sfxy: int, zfxy: int, P: int, D: int, S: int, Q: int) -> np.array:
    xq = quantize_linear_fixed_point(x, sfxx, zfxx, Q)
    Wq = quantize_linear_fixed_point(W, sfxw, zfxw, Q)
    Bq = quantize_linear_fixed_point(B, sfxb, zfxb, Q)

    Cout, Cin, Ks = W.shape
    Lin = x.shape[2]
    Lout = np.floor((Lin + 2*P - D*(Ks-1) - 1) / S + 1).astype(int)

    yq = np.zeros((Cout, Lout), dtype=np.int8)

    for o in range(Cout):
        for t in range(Lout):
            acc = 0
            for c in range(Cin):
                for k in range(Ks):
                    acc += (Wq[o,c,k]-zfxw) * (xq[0,c,t+k-P]-zfxx)
            a0 = (sfxb * (Bq[o] - zfxb)) / sfxy
            a1 = (sfxw * sfxx * acc) / (sfxy * (2**Q))
            a2 = (a0 + a1 + zfxy)
            yq[o,t] = np.clip(a2, -128, 127).astype(np.int8)

    y_rec = dequentize_linear_fixed_point(yq, sfxy, zfxy, Q)
    return y_rec

def convolution_quant_fixed_point_c(x: np.array, W: np.array, B: np.array, sfxx: int, zfxx: int, sfxw: int, zfxw: int, sfxb: int, zfxb: int, sfxy: int, zfxy: int, P: int, D: int, S: int, Q: int):
    xq = quantize_linear_fixed_point(x, sfxx, zfxx, Q)
    Wq = quantize_linear_fixed_point(W, sfxw, zfxw, Q)
    Bq = quantize_linear_fixed_point(B, sfxb, zfxb, Q)

    Cout, Cin, Ks = W.shape
    Lin = x.shape[2]
    Lout = np.floor((Lin + 2*P - D*(Ks-1) - 1) / S + 1).astype(int)

    yq = np.zeros((Cout, Lout), dtype=np.int8)

    xq = xq.reshape(Ks*Cin*Lin)
    Wq = Wq.reshape(Cout*Cin*Ks)
    xq_str = ",".join(map(str, xq))
    Wq_str = ",".join(map(str, Wq))
    Bq_str = ",".join(map(str, Bq))

    size_x = len(xq)
    size_w = len(Wq)

    Lout = int(((Lin + 2*P - D*(Ks-1) - 1) / S + 1))

    cname = "main"
    exename = "test"
    outname = "out"

    main = f"""
#include "ncast_lib.h"
#include <stdio.h>

int main() {{ 
    int8_t xq[{size_x}] = {{ {xq_str} }};
    int8_t Wq[{size_w}] = {{ {Wq_str} }};
    int8_t Bq[{Cout}] = {{ {Bq_str} }};

    int8_t yq[{Cout*Lout}];
    NC_QLINCONV_FXS8(xq,Wq,Bq,yq,{Ks},{Cin},{Lin},{Cout},{Lout},{P},{D},{S},{sfxx},{zfxx},{sfxw},{zfxw},{sfxy},{zfxy},{sfxb},{zfxb},{Q})

    NC_OUTTNS("{TEST_TMP_PATH}/{outname}.txt",yq,{Cout*Lout},"%d");

    return 0;
}}
"""
    
    
    clear_compilation_folder()
    generate_main_c(main, cname)
    run_bash_command(f"gcc -o {TEST_TMP_PATH}/{exename} {TEST_TMP_PATH}/{cname}.c -I {LIB_DIR}")
    run_bash_command(f"{TEST_TMP_PATH}/{exename}")
    yq = read_output(f"{TEST_TMP_PATH}/{outname}.txt")
    y_rec = dequentize_linear_fixed_point(yq, sfxy, zfxy, Q)
    y_rec = y_rec.reshape(Cout,Lout)
    return y_rec
