from patmos.common.common import generate_patmos_analysis, generate_main_patmos_template
from wcet.qlinearconv.wcet_qlinearconv import generate_input_c, generate_input_h, generate_int16_str

def generate_main_patmos(KS, CIN, LIN, COUT, LOUT, PAD, DIL, STR, Q, acctype):
    f = generate_int16_str
    nn_statement = f"NC_QLINCONV_FXS8(xq,wq,bq,yq,{KS},{CIN},{LIN},{COUT},{LOUT},{PAD},{DIL},{STR},{f()},{f()},{f()},{f()},{f()},{f()},{f()},{f()},{Q},{acctype})"
    generate_main_patmos_template(nn_statement, 100)

def qlinearconv_patmos_analysis(name, KS, CIN, LIN, COUT, PAD, DIL, STR, Q, acctype):
    def generate_c_code():
        LOUT = int(((LIN + 2*PAD - DIL*(KS-1) - 1) / STR + 1))
        generate_main_patmos(KS, CIN, LIN, COUT, LOUT, PAD, DIL, STR, Q, acctype)
        generate_input_c(KS, CIN, LIN, COUT, LOUT)
        generate_input_h(KS, CIN, LIN, COUT, LOUT)
    generate_patmos_analysis(name, generate_c_code)

if __name__ == '__main__':
    qlinearconv_patmos_analysis(name="qlinear_conv", KS=1, CIN=4, LIN=5, COUT=3, PAD=0, DIL=1, STR=1, Q=15, acctype="int32_t")
    
