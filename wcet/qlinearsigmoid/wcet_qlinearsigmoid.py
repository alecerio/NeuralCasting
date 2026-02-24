from wcet.common.common import generate_main_wcet_template, generate_input_c_template, generate_input_h_template, generate_wcet_analysis, generate_int32_str

def generate_main(size):
    f = generate_int32_str
    nn_statement = f"NC_QSIGMOID_FXS8(xq,yq,{size},{f()},{f()})"
    generate_main_wcet_template(nn_statement)

def generate_input_c(size):
    generate_input_c_template([ ['int8_t', 'xq', size, True], ['int8_t', 'yq', size, False] ])

def generate_input_h(size):
    generate_input_h_template([ ['int8_t', 'xq', size], ['int8_t', 'yq', size] ])

def qlinearsigmoid_wcet_analysis(name, size):
    def generate_c_code():
        generate_main(size)
        generate_input_c(size)
        generate_input_h(size)
    generate_wcet_analysis(name, generate_c_code)
    

if __name__ == '__main__':
    qlinearsigmoid_wcet_analysis(name='qlinearsigmoid_test', size=100)
