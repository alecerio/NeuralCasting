from patmos.common.common import generate_patmos_analysis, generate_main_patmos_template
from wcet.common.common import generate_int32_str
from wcet.unsqueeze.wcet_unsqueeze import generate_input_c, generate_input_h

def generate_main_patmos(size):
    f = generate_int32_str
    nn_statement = f"NC_UNSQUEEZE(xq,yq,{size})"
    generate_main_patmos_template(nn_statement, 100)

def unsqueeze_patmos_analysis(name, size):
    def generate_c_code():
        generate_main_patmos(size)
        generate_input_c(size)
        generate_input_h(size)
    generate_patmos_analysis(name, generate_c_code)

if __name__ == '__main__':
    unsqueeze_patmos_analysis(name="unsqueeze", size=100)
    
