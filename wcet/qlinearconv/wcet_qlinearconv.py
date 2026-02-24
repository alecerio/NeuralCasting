from wcet.common.common import generate_main_wcet_template, generate_input_c_template, generate_input_h_template, generate_wcet_analysis, generate_int32_str

def generate_main(KS, CIN, LIN, COUT, PAD, DIL, STR, LOUT, Q):
    f = generate_int32_str
    nn_statement = f"NC_QLINCONV_FXS8(xq,wq,bq,yq,{KS},{CIN},{LIN},{COUT},{LOUT},{PAD},{DIL},{STR},{f()},{f()},{f()},{f()},{f()},{f()},{f()},{f()},{Q})"
    generate_main_wcet_template(nn_statement)

def generate_input_c(KS, CIN, LIN, COUT, LOUT):
    generate_input_c_template([['int8_t', 'xq', KS*CIN*LIN, True], ['int8_t', 'wq', COUT*CIN*KS, True], ['int8_t', 'bq', COUT, True], ['int8_t', 'yq', COUT*LOUT, False]])

def generate_input_h(KS, CIN, LIN, COUT, LOUT):
    generate_input_h_template([ ['int8_t', 'xq', KS*CIN*LIN], ['int8_t', 'wq', COUT*CIN*KS], ['int8_t', 'bq', COUT], ['int8_t', 'yq', COUT*LOUT] ])

def qlinearconv_wcet_analysis(name, COUT, CIN, KS, LIN, PAD, DIL, STR, Q):
    def generate_c_code():
        LOUT = int(((LIN + 2*PAD - DIL*(KS-1) - 1) / STR + 1))
        generate_main(KS, CIN, LIN, COUT, PAD, DIL, STR, LOUT, Q)
        generate_input_c(KS, CIN, LIN, COUT, LOUT)
        generate_input_h(KS, CIN, LIN, COUT, LOUT)
    generate_wcet_analysis(name, generate_c_code)

if __name__ == '__main__':
    #qlinearconv_wcet_analysis(name="qlinearconv_test", COUT=3, CIN=4, KS=1, LIN=5, PAD=0, DIL=1, STR=1, Q=31)
    qlinearconv_wcet_analysis(name="qlinearconv_test", COUT=128, CIN=128, KS=1, LIN=5, PAD=0, DIL=1, STR=1, Q=31)