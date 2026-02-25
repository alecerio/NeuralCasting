from patmos.common.common import generate_patmos_analysis, generate_main_patmos_template
from wcet.common.common import generate_int16_str
from wcet.qlinearadd.wcet_qlinearadd import generate_input_c, generate_input_h

def generate_main_patmos(size):
    f = generate_int16_str
    nn_statement = f"NC_QLADD_FXS8(aq,bq,cq,{size},{f()},{f()},{f()},{f()},{f()},{f()},int32_t)"
    generate_main_patmos_template(nn_statement, 100)

def qlinearadd_patmos_analysis(name, size):
    def generate_c_code():
        generate_main_patmos(size)
        generate_input_c(size)
        generate_input_h(size)
    generate_patmos_analysis(name, generate_c_code)

if __name__ == '__main__':
    qlinearadd_patmos_analysis(name="qlinear_add", size=100)
    
