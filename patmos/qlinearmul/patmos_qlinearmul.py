from patmos.common.common import generate_patmos_analysis, generate_main_patmos_template
from wcet.common.common import generate_int32_str
from wcet.qlinearmul.wcet_qlinearmul import generate_input_c, generate_input_h

def generate_main_patmos(size):
    f = generate_int32_str
    nn_statement = f"NC_QLMUL_FXS8(aq,bq,cq,{size},{f()},{f()},{f()},{f()},{f()},{f()},31)"
    generate_main_patmos_template(nn_statement, 100)

def qlinearmul_patmos_analysis(name, size):
    def generate_c_code():
        generate_main_patmos(size)
        generate_input_c(size)
        generate_input_h(size)
    generate_patmos_analysis(name, generate_c_code)

if __name__ == '__main__':
    qlinearmul_patmos_analysis(name="qlinear_mul", size=100)
    
