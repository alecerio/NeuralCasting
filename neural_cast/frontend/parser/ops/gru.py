from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.common.common import fix_identifier
import math
from neural_cast.frontend.parser.ops.common.common import node_shape
from neural_cast.frontend.common.common import onnx_type_to_c_dictionary
from neural_cast.frontend.parser.ops.common.common import gen_define_connected_output
from neural_cast.frontend.parser.ops.common.common import gen_for_loop_begin
from neural_cast.frontend.parser.ops.common.common import gen_for_loop_end
from neural_cast.frontend.parser.ops.common.common import gen_for_loop_index
from neural_cast.frontend.common.common import CompilerConfig

class GRU(OpNode):
    def __init__(self, name : str):
        super().__init__(name)
    
    def __str__(self):
        return super().__str__()
    
    def generate_code(self) -> str:
        parallel : str = CompilerConfig()['parallel']

        name : str = self.get_name()
        define_connected_output : str = gen_define_connected_output(self, 0)
        output_name : str = fix_identifier(self._output_varnames[0])
        in_shape : list[int] = node_shape(self._inputs[0])
        in_size : int = in_shape[0] * in_shape[1] * in_shape[2]
        define_connected_hidden : str = gen_define_connected_output(self, 1)
        output_hidden_name : str = fix_identifier(self._output_varnames[1])
        hidden_shape : list[int] = node_shape(self._inputs[4])
        hidden_size : int = hidden_shape[0] * hidden_shape[1] * hidden_shape[2]
        W_name : str = fix_identifier(self._input_varnames[1])
        input_name : str = fix_identifier(self._input_varnames[0])
        B_name : str = fix_identifier(self._input_varnames[3])
        R_name : str = fix_identifier(self._input_varnames[2])
        input_hidden_name : str = fix_identifier(self._input_varnames[4])

        if parallel == 'omp':
            omp_wir_x_bir : str = self._gen_omp_W_x_b(W_name, input_name, B_name)
            omp_reduction_a : str = self._gen_omp_reduction('a')
            omp_whr_h1_bhr : str = self._gen_omp_W_x_b(R_name, input_hidden_name, B_name)
            omp_reduction_b : str = self._gen_omp_reduction('b')
            omp_wiz_x_biz : str = self._gen_omp_W_x_b(W_name, input_name, B_name)
            omp_reduction_c : str = self._gen_omp_reduction('c')
            omp_whz_h1_bhz : str = self._gen_omp_W_x_b(R_name, input_hidden_name, B_name)
            omp_reduction_d : str = self._gen_omp_reduction('d')
            omp_win_x_bin : str = self._gen_omp_W_x_b(W_name, input_name, B_name)
            omp_reduction_e : str = self._gen_omp_reduction('e')
            omp_whn_h1_bhn : str = self._gen_omp_W_x_b(R_name, input_hidden_name, B_name)
            omp_reduction_f : str = self._gen_omp_reduction('f')
            omp_sigmoid_1 : str = self._gen_omp_rzn(['a', 'b', 'r'])
            omp_sigmoid_2 : str = self._gen_omp_rzn(['c', 'd', 'z'])
            omp_tanh : str = self._gen_omp_rzn(['n', 'e', 'r', 'f'])
            omp_hn : str = self._gen_omp_rzn(['tensor_'+output_hidden_name, 'z', 'n', 'tensor_'+input_hidden_name, 'tensor_'+output_name])
        else:
            omp_wir_x_bir : str = ""
            omp_reduction_a : str = ""
            omp_whr_h1_bhr : str = ""
            omp_reduction_b : str = ""
            omp_wiz_x_biz : str = ""
            omp_reduction_c : str = ""
            omp_whz_h1_bhz : str = ""
            omp_reduction_d : str = ""
            omp_win_x_bin : str = ""
            omp_reduction_e : str = ""
            omp_whn_h1_bhn : str = ""
            omp_reduction_f : str = ""
            omp_sigmoid_1 : str = ""
            omp_sigmoid_2 : str = ""
            omp_tanh : str = ""
            omp_hn : str = ""
        
        nflops_exp : int = 4
        nflops_tanh : int = 4
        nflops : int = hidden_size * ( 6 * ( in_size + hidden_size + 2 ) + nflops_tanh + 2 * ( 2 + nflops_exp ) )

        code : str = self._read_template_c("GRU.c")

        code = self._expand_pattern(code, "$NAME", name)
        code = self._expand_pattern(code, "$DEFINE_CONNECTED_OUTPUT", define_connected_output)
        code = self._expand_pattern(code, "$OUTPUT_NAME", output_name)
        code = self._expand_pattern(code, "$INPUT_SIZE", str(in_size))
        code = self._expand_pattern(code, "$DEFINE_CONNECTED_HIDDEN", define_connected_hidden)
        code = self._expand_pattern(code, "$OUTPUT_HIDDEN_NAME", output_hidden_name)
        code = self._expand_pattern(code, "$HIDDEN_SIZE", str(hidden_size))
        code = self._expand_pattern(code, "$W_NAME", W_name)
        code = self._expand_pattern(code, "$INPUT_NAME", input_name)
        code = self._expand_pattern(code, "$B_NAME", B_name)
        code = self._expand_pattern(code, "$R_NAME", R_name)
        code = self._expand_pattern(code, "$INPUT_HIDDEN_NAME", input_hidden_name)

        code = self._expand_pattern(code, "$OMP_WIR_X_BIR", omp_wir_x_bir)
        code = self._expand_pattern(code, "$OMP_REDUCTION_A", omp_reduction_a)
        code = self._expand_pattern(code, "$OMP_WHR_H1_BHR", omp_whr_h1_bhr)
        code = self._expand_pattern(code, "$OMP_REDUCTION_B", omp_reduction_b)
        code = self._expand_pattern(code, "$OMP_WIZ_X_BIZ", omp_wiz_x_biz)
        code = self._expand_pattern(code, "$OMP_REDUCTION_C", omp_reduction_c)
        code = self._expand_pattern(code, "$OMP_WHZ_H1_BHZ", omp_whz_h1_bhz)
        code = self._expand_pattern(code, "$OMP_REDUCTION_D", omp_reduction_d)
        code = self._expand_pattern(code, "$OMP_WIN_X_BIN", omp_win_x_bin)
        code = self._expand_pattern(code, "$OMP_REDUCTION_E", omp_reduction_e)
        code = self._expand_pattern(code, "$OMP_WHN_H1_BHN", omp_whn_h1_bhn)
        code = self._expand_pattern(code, "$OMP_REDUCTION_F", omp_reduction_f)
        code = self._expand_pattern(code, "$OMP_SIGMOID_1", omp_sigmoid_1)
        code = self._expand_pattern(code, "$OMP_SIGMOID_2", omp_sigmoid_2)
        code = self._expand_pattern(code, "$OMP_TANH", omp_tanh)
        code = self._expand_pattern(code, "$OMP_HN", omp_hn)

        code = self._expand_pattern(code, "$NFLOPS", str(nflops))

        return code
    
    def generate_includes_code_c(self) -> str:
        code : str = self._read_template_c("GRU_inc.c")
        return code

    def generate_declaration_code_c(self) -> str:
        return ""
    
    def infer_output_shape(self) -> list[list[int]]:
        input : Node = self._inputs[0]
        hidden : Node = self._inputs[4]
        shape : list[int] = node_shape(input)
        shape_hidden : list[int] = node_shape(hidden)

        batch_size : int = shape[1]
        sequence_lengh : int = shape[0]
        hidden_size : int = shape_hidden[2]
        shape_out = [1, batch_size, sequence_lengh, hidden_size]

        return shape_out
    
    def infer_output_type(self) -> int:
        input : Node = self._inputs[0]
        return input.infer_output_type()
    
    def get_op_type(self) -> str:
        return "GRU"
    
    def _gen_omp_W_x_b(self, W_name : str, x_name : str, b_name : str) -> str:
        code : str = '#pragma omp parallel for shared(tensor_' + W_name + ', tensor_' + x_name + ',  tensor_' + b_name + ') collapse(1)'
        return code
    
    def _gen_omp_reduction(self, acc_name : str) -> str:
        code : str = '#pragma omp reduction(' + acc_name + ': +)'
        return code
    
    def _gen_omp_rzn(self, dependencies : list[str]) -> str:
        dep_code : str = ''
        n_dep : int = len(dependencies)
        for index, dep in enumerate(dependencies):
            dep_code += dep
            if index < n_dep-1:
                dep_code += ', '
        return '#pragma omp parallel for shared(' + dep_code + ')'