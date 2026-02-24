from patmos.common.common import generate_patmos_analysis, generate_main_patmos_template
from wcet.common.common import generate_int32_str
from wcet.qlinearmatmul.wcet_qlinearmatmul import generate_input_c, generate_input_h

def generate_main_patmos(M, K, N):
    f = generate_int32_str
    Q = 31
    Qs = 20
    nn_statement = f"NC_QLINMATMUL(aq,bq,cq,{M},{N},{K},{f()},{f()},{f()},{f()},{f()},{f()},{Q},{Qs})"
    generate_main_patmos_template(nn_statement, 100)

def qlinearmatmul_patmos_analysis(name, M, K, N):
    def generate_c_code():
        generate_main_patmos(M, K, N)
        generate_input_c(M, N, K)
        generate_input_h(M, N, K)
    generate_patmos_analysis(name, generate_c_code)

if __name__ == '__main__':
    qlinearmatmul_patmos_analysis(name="qlinear_matmul", M=10, K=10, N=10)
    
