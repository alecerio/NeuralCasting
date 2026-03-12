from patmos.common.common import generate_patmos_analysis, generate_main_patmos_template
from wcet.common.common import generate_input_c_template, generate_input_h_template, generate_int16_str


def generate_input_c(KH, KW, CIN, HIN, WIN, COUT, HOUT, WOUT):
    generate_input_c_template([
        ['int8_t', 'xq', CIN * HIN * WIN, True],
        ['int8_t', 'wq', COUT * CIN * KH * KW, True],
        ['int8_t', 'bq', COUT, True],
        ['int8_t', 'yq', COUT * HOUT * WOUT, False],
    ])


def generate_input_h(KH, KW, CIN, HIN, WIN, COUT, HOUT, WOUT):
    generate_input_h_template([
        ['int8_t', 'xq', CIN * HIN * WIN],
        ['int8_t', 'wq', COUT * CIN * KH * KW],
        ['int8_t', 'bq', COUT],
        ['int8_t', 'yq', COUT * HOUT * WOUT],
    ])


def generate_main_patmos(KH, KW, CIN, HIN, WIN, COUT, HOUT, WOUT,
                         PADH, PADW, DILH, DILW, STRH, STRW, Q, acctype):
    f = generate_int16_str
    nn_statement = (
        f"NC_QLINCONV2D_FXS8("
        f"xq,wq,bq,yq,"
        f"{KH},{KW},{CIN},{HIN},{WIN},{COUT},{HOUT},{WOUT},"
        f"{PADH},{PADW},{DILH},{DILW},{STRH},{STRW},"
        f"{f()},{f()},{f()},{f()},{f()},{f()},{f()},{f()},{Q},{acctype})"
    )
    generate_main_patmos_template(nn_statement, 100)


def qlinearconv2d_patmos_analysis(name, KH, KW, CIN, HIN, WIN, COUT,
                                  PADH, PADW, DILH, DILW, STRH, STRW,
                                  Q, acctype):
    def generate_c_code():
        HOUT = int(((HIN + 2 * PADH - DILH * (KH - 1) - 1) / STRH + 1))
        WOUT = int(((WIN + 2 * PADW - DILW * (KW - 1) - 1) / STRW + 1))

        generate_main_patmos(
            KH, KW, CIN, HIN, WIN, COUT, HOUT, WOUT,
            PADH, PADW, DILH, DILW, STRH, STRW, Q, acctype
        )
        generate_input_c(KH, KW, CIN, HIN, WIN, COUT, HOUT, WOUT)
        generate_input_h(KH, KW, CIN, HIN, WIN, COUT, HOUT, WOUT)

    generate_patmos_analysis(name, generate_c_code)


if __name__ == '__main__':
    qlinearconv2d_patmos_analysis(
        name="qlinear_conv2d",
        KH=1,
        KW=1,
        CIN=4,
        HIN=5,
        WIN=5,
        COUT=3,
        PADH=0,
        PADW=0,
        DILH=1,
        DILW=1,
        STRH=1,
        STRW=1,
        Q=15,
        acctype="int32_t"
    )