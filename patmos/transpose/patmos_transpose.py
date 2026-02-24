from patmos.common.common import generate_patmos_analysis, generate_main_patmos_template
from wcet.common.common import generate_int32_str
from wcet.transpose.wcet_transpose import generate_input_c, generate_input_h

def generate_main_patmos(cols, rows):
    f = generate_int32_str
    nn_statement = f"NC_TR2D(xq,yq,{cols},{rows})"
    generate_main_patmos_template(nn_statement, 100)

def transpose_patmos_analysis(name, cols, rows):
    def generate_c_code():
        generate_main_patmos(cols, rows)
        generate_input_c(cols, rows)
        generate_input_h(cols, rows)
    generate_patmos_analysis(name, generate_c_code)

if __name__ == '__main__':
    transpose_patmos_analysis(name="transpose", cols=5, rows=20)
    
