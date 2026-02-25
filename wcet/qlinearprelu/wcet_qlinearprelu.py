from wcet.common.common import generate_main_wcet_template, generate_input_c_template, generate_input_h_template, generate_wcet_analysis, generate_int16_str

def generate_main(size, Q, acctype):
    f = generate_int16_str
    nn_statement = f"NC_QLPRELU_FXS8(xq,{f()},yq,{size},{f()},{f()},{f()},{f()},{Q},{acctype})"
    generate_main_wcet_template(nn_statement)

def generate_input_c(size):
    generate_input_c_template([ ['int8_t', 'xq', size, True], ['int8_t', 'yq', size, False] ])

def generate_input_h(size):
    generate_input_h_template([ ['int8_t', 'xq', size], ['int8_t', 'yq', size] ])

def qlinearprelu_wcet_analysis(name, size, Q, acctype):
    def generate_c_code():
        generate_main(size, Q, acctype)
        generate_input_c(size)
        generate_input_h(size)
    generate_wcet_analysis(name, generate_c_code)

if __name__ == '__main__':
    qlinearprelu_wcet_analysis(name='qlinearprelu_test', size=100, Q=15, acctype="int32_t")
