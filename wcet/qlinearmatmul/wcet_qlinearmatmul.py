from wcet.common.common import generate_main_wcet_template, generate_input_c_template, generate_input_h_template, generate_wcet_analysis, generate_int16_str

def generate_main(M, N, K, Q, acctype):
    f = generate_int16_str
    nn_statement = f"NC_QLINMM_FXS8(aq,bq,cq,{M},{N},{K},{f()},{f()},{f()},{f()},{f()},{f()},{Q},{acctype})"
    generate_main_wcet_template(nn_statement)

def generate_input_c(M, N, K):
    generate_input_c_template([ ['int8_t', 'aq', M*K, True], ['int8_t', 'bq', K*N, True], ['int8_t', 'cq', M*N, False] ])

def generate_input_h(M, N, K):
    generate_input_h_template([ ['int8_t', 'aq', M*K], ['int8_t', 'bq', K*N], ['int8_t', 'cq', M*N] ])

def qlinearmatmul_wcet_analysis(name, M, N, K, Q, acctype):
    def generate_c_code():
        generate_main(M, N, K, Q, acctype)
        generate_input_c(M, N, K)
        generate_input_h(M, N, K)
    generate_wcet_analysis(name, generate_c_code)
    

if __name__ == '__main__':
    qlinearmatmul_wcet_analysis(name='qlinearmatmul_test', M=10, N=10, K=10, Q=15, acctype="int32_t")
