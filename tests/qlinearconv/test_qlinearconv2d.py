import torch
import torch.nn as nn
import numpy as np
from typing import List
from tests.common.common import compute_s_z, compute_sfx_zfx, quantize_linear, dequantize_linear, quantize_linear_fixed_point, dequentize_linear_fixed_point
from tests.common.common import clear_compilation_folder, generate_main_c, run_bash_command, read_output
from config.config import TEST_TMP_PATH, LIB_DIR

def test_qlinearconv():
    # seed
    torch.manual_seed(42)

    # convoltion parameters
    Cin = 4
    Cout = 3
    Ks = (3, 3)
    stride = (1, 1)
    padding = (0, 0)
    dilation = (1, 1)
    groups = 1
    conv = nn.Conv2d(
        Cin,
        Cout,
        kernel_size=Ks,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=True
    )

    W = conv.weight.detach().numpy()
    B = conv.bias.detach().numpy()

    LH = 4
    LW = 4
    x = torch.randn(1, 4, LH, LW)
    xnp = x.detach().numpy()

    # convolution torch model
    y_torch = convolution_torch_float(x, conv)

    # convolution float
    y_float = convolution_float(xnp, W, B, padding, dilation, stride)
    assert np.allclose(y_torch.detach().numpy(), y_float, atol=1e-6)

    # compute quantization data
    Q = 15
    sx, zx = compute_s_z(xnp)
    sw, zw = compute_s_z(W)
    sb, zb = compute_s_z(B)
    sy, zy = compute_s_z(y_float)
    sfx, zfx = compute_sfx_zfx(sx, zx, Q)
    sfxw, zfxw = compute_sfx_zfx(sw, zw, Q)
    sfxb, zfxb = compute_sfx_zfx(sb, zb, Q)
    sfxy, zfxy = compute_sfx_zfx(sy, zy, Q)

    # convolution quantized model (fixed)
    y_rec = convolution_quant_float(xnp, W, B, sx, zx, sw, zw, sb, zb, sy, zy, padding, dilation, stride)
    assert np.allclose(y_torch.detach().numpy(), y_rec, atol=1e-1)

    # convolution quantized model (float)
    y_rec1 = convolution_quant_fixed_point(xnp, W, B, sfx, zfx, sfxw, zfxw, sfxb, zfxb, sfxy, zfxy, padding, dilation, stride, Q)
    assert np.allclose(y_torch.detach().numpy(), y_rec1, atol=1e-1)

    # convolution quantized model (fixed c)
    y_rec2 = convolution_quant_fixed_point_c(xnp, W, B, sfx, zfx, sfxw, zfxw, sfxb, zfxb, sfxy, zfxy, padding, dilation, stride, Q)
    assert np.allclose(y_torch.detach().numpy(), y_rec2, atol=1e-1)

def convolution_torch_float(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    y = conv(x)
    return y

def convolution_float(x: np.array, W: np.array, B: np.array, P: List[int], D: List[int], S: List[int]) -> np.array:
    pad_h, pad_w = P
    dil_h, dil_w = D
    str_h, str_w = S

    N, Cin, Hin, Win = x.shape
    Cout, _, Kh, Kw = W.shape

    Hout = (Hin + 2 * pad_h - dil_h * (Kh - 1) - 1) // str_h + 1
    Wout = (Win + 2 * pad_w - dil_w * (Kw - 1) - 1) // str_w + 1

    xpad = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    y = np.zeros((N, Cout, Hout, Wout), dtype=np.float32)

    for n in range(N):
        for o in range(Cout):
            for oh in range(Hout):
                for ow in range(Wout):
                    acc = B[o]
                    for c in range(Cin):
                        for kh in range(Kh):
                            for kw in range(Kw):
                                ih = oh * str_h + kh * dil_h
                                iw = ow * str_w + kw * dil_w
                                acc += xpad[n, c, ih, iw] * W[o, c, kh, kw]
                    y[n, o, oh, ow] = acc

    return y

def convolution_quant_float( x: np.array, W: np.array, B: np.array, sx: float, zx: int, sw: float, zw: int, sb: float, zb: int, sy: float, zy: int, P, D, S) -> np.array:
    xq = quantize_linear(x, sx, zx)
    Wq = quantize_linear(W, sw, zw)
    Bq = quantize_linear(B, sb, zb)

    if isinstance(P, int):
        Ph, Pw = P, P
    else:
        Ph, Pw = P

    if isinstance(D, int):
        Dh, Dw = D, D
    else:
        Dh, Dw = D

    if isinstance(S, int):
        Sh, Sw = S, S
    else:
        Sh, Sw = S

    Cout, Cin, Kh, Kw = W.shape
    Hin = x.shape[2]
    Win = x.shape[3]

    Hout = int(np.floor((Hin + 2 * Ph - Dh * (Kh - 1) - 1) / Sh + 1))
    Wout = int(np.floor((Win + 2 * Pw - Dw * (Kw - 1) - 1) / Sw + 1))

    yq = np.zeros((Cout, Hout, Wout), dtype=np.int8)

    for o in range(Cout):
        for oh in range(Hout):
            for ow in range(Wout):
                acc = 0.0
                for c in range(Cin):
                    for kh in range(Kh):
                        for kw in range(Kw):
                            ih = oh * Sh + kh * Dh - Ph
                            iw = ow * Sw + kw * Dw - Pw
                            if 0 <= ih < Hin and 0 <= iw < Win:
                                acc += (Wq[o, c, kh, kw] - zw) * (xq[0, c, ih, iw] - zx)

                yq[o, oh, ow] = np.clip(
                    acc * (sx * sw) / sy + (sb / sy) * (Bq[o] - zb) + zy,
                    -128,
                    127
                ).astype(np.int8)

    y_rec = dequantize_linear(yq, sy, zy)
    return y_rec

def convolution_quant_fixed_point( x: np.array, W: np.array, B: np.array, sfxx: int, zfxx: int, sfxw: int, zfxw: int, sfxb: int, zfxb: int, sfxy: int, zfxy: int, P, D, S, Q: int) -> np.array:
    xq = quantize_linear_fixed_point(x, sfxx, zfxx, Q)
    Wq = quantize_linear_fixed_point(W, sfxw, zfxw, Q)
    Bq = quantize_linear_fixed_point(B, sfxb, zfxb, Q)

    if isinstance(P, int):
        Ph, Pw = P, P
    else:
        Ph, Pw = P

    if isinstance(D, int):
        Dh, Dw = D, D
    else:
        Dh, Dw = D

    if isinstance(S, int):
        Sh, Sw = S, S
    else:
        Sh, Sw = S

    Cout, Cin, Kh, Kw = W.shape
    Hin = x.shape[2]
    Win = x.shape[3]

    Hout = int(np.floor((Hin + 2 * Ph - Dh * (Kh - 1) - 1) / Sh + 1))
    Wout = int(np.floor((Win + 2 * Pw - Dw * (Kw - 1) - 1) / Sw + 1))

    yq = np.zeros((Cout, Hout, Wout), dtype=np.int8)

    for o in range(Cout):
        a0 = (sfxb * (Bq[o] - zfxb)) / sfxy

        for oh in range(Hout):
            for ow in range(Wout):
                acc = 0

                for c in range(Cin):
                    for kh in range(Kh):
                        for kw in range(Kw):
                            ih = oh * Sh + kh * Dh - Ph
                            iw = ow * Sw + kw * Dw - Pw

                            if 0 <= ih < Hin and 0 <= iw < Win:
                                acc += (Wq[o, c, kh, kw] - zfxw) * (xq[0, c, ih, iw] - zfxx)

                a1 = (sfxw * sfxx * acc) / (sfxy * (2 ** Q))
                a2 = a0 + a1 + zfxy
                yq[o, oh, ow] = np.clip(a2, -128, 127).astype(np.int8)

    y_rec = dequentize_linear_fixed_point(yq, sfxy, zfxy, Q)
    return y_rec

def convolution_quant_fixed_point_c(
    x: np.array,
    W: np.array,
    B: np.array,
    sfxx: int,
    zfxx: int,
    sfxw: int,
    zfxw: int,
    sfxb: int,
    zfxb: int,
    sfxy: int,
    zfxy: int,
    P,
    D,
    S,
    Q: int,
):
    xq = quantize_linear_fixed_point(x, sfxx, zfxx, Q)
    Wq = quantize_linear_fixed_point(W, sfxw, zfxw, Q)
    Bq = quantize_linear_fixed_point(B, sfxb, zfxb, Q)

    if isinstance(P, int):
        Ph, Pw = P, P
    else:
        Ph, Pw = P

    if isinstance(D, int):
        Dh, Dw = D, D
    else:
        Dh, Dw = D

    if isinstance(S, int):
        Sh, Sw = S, S
    else:
        Sh, Sw = S

    Cout, Cin, Kh, Kw = W.shape
    Hin = x.shape[2]
    Win = x.shape[3]

    Hout = int(np.floor((Hin + 2 * Ph - Dh * (Kh - 1) - 1) / Sh + 1))
    Wout = int(np.floor((Win + 2 * Pw - Dw * (Kw - 1) - 1) / Sw + 1))

    yq = np.zeros((Cout, Hout, Wout), dtype=np.int8)

    # La macro C lavora su XQ come [CIN][HIN][WIN], quindi togliamo la batch dim.
    xq_flat = xq[0].reshape(Cin * Hin * Win)
    Wq_flat = Wq.reshape(Cout * Cin * Kh * Kw)

    xq_str = ",".join(map(str, xq_flat))
    Wq_str = ",".join(map(str, Wq_flat))
    Bq_str = ",".join(map(str, Bq))

    size_x = len(xq_flat)
    size_w = len(Wq_flat)

    cname = "main"
    exename = "test"
    outname = "out"
    acctype = "int32_t"

    main = f"""
#include "ncast_lib.h"
#include <stdio.h>
#include <stdint.h>

int main() {{
    int8_t xq[{size_x}] = {{ {xq_str} }};
    int8_t Wq[{size_w}] = {{ {Wq_str} }};
    int8_t Bq[{Cout}] = {{ {Bq_str} }};

    int8_t yq[{Cout * Hout * Wout}];

    NC_QLINCONV2D_FXS8(
        xq, Wq, Bq, yq,
        {Kh}, {Kw},
        {Cin}, {Hin}, {Win},
        {Cout}, {Hout}, {Wout},
        {Ph}, {Pw},
        {Dh}, {Dw},
        {Sh}, {Sw},
        {sfxx}, {zfxx},
        {sfxw}, {zfxw},
        {sfxy}, {zfxy},
        {sfxb}, {zfxb},
        {Q}, {acctype}
    )

    NC_OUTTNS("{TEST_TMP_PATH}/{outname}.txt", yq, {Cout * Hout * Wout}, "%d");

    return 0;
}}
"""

    clear_compilation_folder()
    generate_main_c(main, cname)
    run_bash_command(f"gcc -o {TEST_TMP_PATH}/{exename} {TEST_TMP_PATH}/{cname}.c -I {LIB_DIR}")
    run_bash_command(f"{TEST_TMP_PATH}/{exename}")

    yq = read_output(f"{TEST_TMP_PATH}/{outname}.txt")
    y_rec = dequentize_linear_fixed_point(yq, sfxy, zfxy, Q)
    y_rec = y_rec.reshape(Cout, Hout, Wout)
    return y_rec