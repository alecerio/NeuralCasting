from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.common.common import fix_identifier
import math
from neural_cast.frontend.parser.ops.common.common import node_type_binary_operation
from neural_cast.frontend.parser.ops.common.common import gen_for_loop_begin
from neural_cast.frontend.parser.ops.common.common import gen_for_loop_end
from neural_cast.frontend.parser.ops.common.common import gen_define_connected_output
from neural_cast.frontend.parser.ops.common.common import gen_element_wise_broadcasting_indices
from neural_cast.frontend.parser.ops.common.common import infer_output_shape_for_element_wise_binary_operators

class QLinearMul(OpNode):
    def __init__(self, name : str):
        super().__init__(name)

    def __str__(self):
        return super().__str__()

    def generate_code(self) -> str:
        #parallel : str = CompilerConfig()['parallel']
        name : str = fix_identifier(self.get_name())
        input1_name : str = fix_identifier(self._input_varnames[0])
        input_sa : str = fix_identifier(self._input_varnames[1])
        input_za : str = fix_identifier(self._input_varnames[2])
        input2_name : str = fix_identifier(self._input_varnames[3])
        input_sb : str = fix_identifier(self._input_varnames[4])
        input_zb : str = fix_identifier(self._input_varnames[5])
        input_sy : str = fix_identifier(self._input_varnames[6])
        input_zy : str = fix_identifier(self._input_varnames[7])
        output_name : str = fix_identifier(self._output_varnames[0])
        out_shape : int = self.infer_output_shape()
        for_loop_begin : str = gen_for_loop_begin(out_shape)
        for_loop_end : str = gen_for_loop_end(out_shape)
        input1 : Node = self._inputs[0]
        input2 : Node = self._inputs[3]
        [index_tot, index_1, index_2] = gen_element_wise_broadcasting_indices(input1, input2, out_shape, "QLinearMul")
        
        #if parallel == 'omp':
        #    for_loop_begin = gen_introduce_omp_in_for_loop_elemen_wise(for_loop_begin, input_name_1, input_name_2, output_name)

        code : str = self._read_template_c("QLinearMul.c")

        code = self._expand_pattern(code, "$(NAME)", name)
        code = self._expand_pattern(code, "$(INPUT_SA)", input_sa)
        code = self._expand_pattern(code, "$(INPUT_SY)", input_sy)
        code = self._expand_pattern(code, "$(INPUT_SB)", input_sb)
        code = self._expand_pattern(code, "$(FOR_LOOPS_BEGIN)", for_loop_begin)
        code = self._expand_pattern(code, "$(FOR_LOOPS_END)", for_loop_end)
        code = self._expand_pattern(code, "$(INPUT1_NAME)", input1_name)
        code = self._expand_pattern(code, "$(INPUT2_NAME)", input2_name)
        code = self._expand_pattern(code, "$(INDEX1)", str(index_1))
        code = self._expand_pattern(code, "$(INDEX2)", str(index_2))
        code = self._expand_pattern(code, "$(INPUT_ZA)", input_za)
        code = self._expand_pattern(code, "$(INPUT_ZB)", input_zb)
        code = self._expand_pattern(code, "$(INPUT_ZY)", input_zy)
        code = self._expand_pattern(code, "$(OUTPUT_NAME)", output_name)
        code = self._expand_pattern(code, "$(INDEX_TOT)", str(index_tot))

        return code
    
    def generate_includes_code_c(self) -> str:
        code : str = self._read_template_c("QLinearMul_inc.c")
        return code
    
    def generate_declaration_code_c(self) -> str:
        name : str = fix_identifier(self.get_name())
        out_shape : int = self.infer_output_shape()
        out_size : int = math.prod(out_shape)
        define_connected_output : str = gen_define_connected_output(self, 0)
        output_name : str = fix_identifier(self._output_varnames[0])

        code : str = self._read_template_c("QLinearMul_decl.c")

        code = self._expand_pattern(code, "$(NAME)", name)
        code = self._expand_pattern(code, "$(OUTPUT_SIZE)", str(out_size))
        code = self._expand_pattern(code, "$(DEFINE_CONNECTED_OUTPUT)", define_connected_output)
        code = self._expand_pattern(code, "$(OUTPUT_NAME)", output_name)

        return code
    
    def infer_output_shape(self) -> list[list[int]]:
        input1 : Node = self._inputs[0]
        input2 : Node = self._inputs[1]
        return infer_output_shape_for_element_wise_binary_operators(input1, input2, "QLinearMul")
    
    def get_op_type(self) -> str:
        return "QLinearMul"
    
    def infer_output_type(self) -> int:
        input1 : Node = self._inputs[0]
        input2 : Node = self._inputs[1]
        return node_type_binary_operation(input1, input2, "element-wise quantized multiplication")