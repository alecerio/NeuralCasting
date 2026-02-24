from wcet.common.common import generate_main_wcet_template, generate_input_c_template, generate_input_h_template, generate_wcet_analysis, generate_int32_str

def generate_main(size):
    f = generate_int32_str
    nn_statement = f"NC_QLMUL_FXS8(aq,bq,cq,{size},{f()},{f()},{f()},{f()},{f()},{f()},31)"
    generate_main_wcet_template(nn_statement)

def generate_input_c(size):
    generate_input_c_template([ ['int8_t', 'aq', size, True], ['int8_t', 'bq', size, True], ['int8_t', 'cq', size, False] ])

def generate_input_h(size):
    generate_input_h_template([ ['int8_t', 'aq', size], ['int8_t', 'bq', size], ['int8_t', 'cq', size] ])

def qlinearmul_wcet_analysis(name, size):
    def generate_c_code():
        generate_main(size)
        generate_input_c(size)
        generate_input_h(size)
    generate_wcet_analysis(name, generate_c_code)

if __name__ == '__main__':
    qlinearmul_wcet_analysis(name='qlinearmul_test', size=12850)
