from wcet.common.common import generate_main_wcet_template, generate_input_c_template, generate_input_h_template, generate_wcet_analysis, generate_int32_str

def generate_main(cols, rows):
    f = generate_int32_str
    nn_statement = f"NC_TR2D(xq,yq,{cols},{rows})"
    generate_main_wcet_template(nn_statement)

def generate_input_c(cols, rows):
    generate_input_c_template([ ['int8_t', 'xq', rows*cols, True], ['int8_t', 'yq', rows*cols, False] ])

def generate_input_h(cols, rows):
    generate_input_h_template([ ['int8_t', 'xq', rows*cols], ['int8_t', 'yq', rows*cols] ])

def transpose_wcet_analysis(name, cols, rows):
    def generate_c_code():
        generate_main(cols, rows)
        generate_input_c(cols, rows)
        generate_input_h(cols, rows)
    generate_wcet_analysis(name, generate_c_code)

if __name__ == '__main__':
    transpose_wcet_analysis(name='transpose_test', cols=5, rows=20)
