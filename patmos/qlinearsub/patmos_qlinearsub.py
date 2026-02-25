from patmos.common.common import generate_patmos_analysis, generate_main_patmos_template
from wcet.qlinearsub.wcet_qlinearsub import generate_input_c, generate_input_h, generate_int16_str

def generate_main_patmos(size, acctype):
    f = generate_int16_str
    nn_statement = f"NC_QLSUB_FXS8(aq,bq,cq,{size},{f()},{f()},{f()},{f()},{f()},{f()},{acctype})"
    generate_main_patmos_template(nn_statement, 100)

def qlinearsub_patmos_analysis(name, size, acctype):
    def generate_c_code():
        generate_main_patmos(size, acctype)
        generate_input_c(size)
        generate_input_h(size)
    generate_patmos_analysis(name, generate_c_code)

if __name__ == '__main__':
    qlinearsub_patmos_analysis(name="qlinear_sub", size=100, acctype="int32_t")
    
