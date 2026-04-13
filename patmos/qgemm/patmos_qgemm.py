from patmos.common.common import generate_patmos_analysis, generate_main_patmos_template
from wcet.qgemm.wcet_qgemm import generate_input_c, generate_input_h, generate_int16_str

def generate_main_patmos(M, K, N, Q, acctype):
    f = generate_int16_str
    nn_statement = f"NC_QGEMM_FXS8(aq,bq,cq,biasq,{M},{N},{K},{f()},{f()},{f()},{f()},{f()},{f()},{f()},{f()},{Q},{acctype})"
    generate_main_patmos_template(nn_statement, 100)

def qgemm_patmos_analysis(name, M, K, N, Q, acctype):
    def generate_c_code():
        generate_main_patmos(M, K, N, Q, acctype)
        generate_input_c(M, N, K)
        generate_input_h(M, N, K)
    generate_patmos_analysis(name, generate_c_code)

if __name__ == '__main__':
    qgemm_patmos_analysis(name="qgemm", M=10, K=10, N=10, Q=15, acctype="int32_t")
    
