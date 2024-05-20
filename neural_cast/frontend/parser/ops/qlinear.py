import math
from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.common.common import fix_identifier
from neural_cast.frontend.parser.ops.common.common import node_shape
from neural_cast.frontend.parser.ops.common.common import gen_define_connected_output
from neural_cast.frontend.parser.ops.common.common import gen_for_loop_begin
from neural_cast.frontend.parser.ops.common.common import gen_for_loop_end
from neural_cast.frontend.parser.ops.common.common import gen_for_loop_index
from neural_cast.frontend.common.common import onnx_type_to_c_dictionary
from neural_cast.frontend.common.common import CompilerConfig
from neural_cast.frontend.parser.ops.common.common import gen_introduce_omp_in_for_loop_elem_by_elem

class QLinear(OpNode):
    def __init__(self, name : str):
        super().__init__(name)

    def __str__(self):
        return super.__str__()

    def generate_code(self) -> str:
        parallel : str = CompilerConfig()['parallel']
        name : str = fix_identifier(self.get_name())
        input_name : str = fix_identifier(self._input_varnames[0])
        output_name : str = fix_identifier(self._output_varnames[0])
        y_scale_id : str = fix_identifier(self._input_varnames[1])
        zero_id : str = fix_identifier(self._input_varnames[2])


        #if parallel == 'omp':
        #    for_loop_begin = gen_introduce_omp_in_for_loop_elem_by_elem(for_loop_begin, input_name, output_name)

        code : str = self._read_template_c("QLinear.c")

        code = self._expand_pattern(code, "$(NAME)", name)
        code = self._expand_pattern(code, "$(INPUT_NAME)", input_name)
        code = self._expand_pattern(code, "$(OUTPUT_NAME)", output_name)
        code = self._expand_pattern(code, "$(SCALING_FACTOR_NAME)", y_scale_id)
        code = self._expand_pattern(code, "$(ZERO_NAME)", zero_id)

        return code

    def generate_declaration_code_c(self) -> str:
        define_connected_output : str = gen_define_connected_output(self, 0)
        output_name : str = fix_identifier(self._output_varnames[0])
        name : str = fix_identifier(self.get_name())
        in_shape : int = self.infer_output_shape()
        in_size : int = math.prod(in_shape)
        out_size : int = in_size

        code : str = self._read_template_c("QLinear_inc.c")

        code = self._expand_pattern(code, "$(DEFINE_CONNECTED_OUTPUT)", define_connected_output)
        code = self._expand_pattern(code, "$(OUTPUT_NAME)", output_name)
        code = self._expand_pattern(code, "$(NAME)", name)
        code = self._expand_pattern(code, "$(OUTPUT_SIZE)", str(out_size))

        return code

    def infer_output_shape(self) -> list[list[int]]:
        input : Node = self._inputs[0]
        shape : list[int] = node_shape(input)
        return shape

    def infer_output_type(self) -> int:
        input1 : Node = self._inputs[0]
        return input1.infer_output_type()

    def get_op_type(self) -> str:
        return "QLinear"

    def generate_includes_code_c(self) -> str:
        code : str = self._read_template_c("QLinear_inc.c")
        return code