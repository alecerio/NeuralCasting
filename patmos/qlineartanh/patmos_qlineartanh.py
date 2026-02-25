from patmos.common.common import generate_patmos_analysis, generate_main_patmos_template
from wcet.common.common import generate_int16_str
from wcet.qlineartanh.wcet_qlineartanh import generate_input_c, generate_input_h

def generate_main_patmos(size, acctype):
    f = generate_int16_str
    nn_statement = f"NC_QTANH_FXS8(xq,yq,{size},{f()},{f()},{acctype})"
    generate_main_patmos_template(nn_statement, 100)

def qlineartanh_patmos_analysis(name, size, acctype):
    def generate_c_code():
        generate_main_patmos(size, acctype)
        generate_input_c(size)
        generate_input_h(size)
    generate_patmos_analysis(name, generate_c_code)

if __name__ == '__main__':
    qlineartanh_patmos_analysis(name="qlinear_tanh", size=100, acctype="int32_t")
    
