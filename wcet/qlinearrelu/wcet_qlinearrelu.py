from wcet.common.common import generate_main_wcet_template, generate_input_c_template, generate_input_h_template, generate_wcet_analysis, generate_int16_str

def generate_main(size, acctype):
    f = generate_int16_str
    nn_statement = f"NC_RELU_FXS8(xq,yq,{size},{f()},{f()},{f()},{f()},{acctype})"
    generate_main_wcet_template(nn_statement)

def generate_input_c(size):
    generate_input_c_template([ ['int8_t', 'xq', size, True],  ['int8_t', 'yq', size, False] ])

def generate_input_h(size):
    generate_input_h_template([ ['int8_t', 'xq', size],  ['int8_t', 'yq', size] ])

def qlinearrelu_wcet_analysis(name, size, acctype):
    def generate_c_code():
        generate_main(size, acctype)
        generate_input_c(size)
        generate_input_h(size)
    generate_wcet_analysis(name, generate_c_code)

if __name__ == '__main__':
    qlinearrelu_wcet_analysis(name='qlinearrelu_test', size=100, acctype="int32_t")
