from neural_cast.frontend.common.common import CompilerConfig
from neural_cast.frontend.parser.ops.common.common import gen_introduce_omp_in_for_loop_elem_by_elem
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

class Tanh(OpNode):
    def __init__(self, name : str):
        super().__init__(name)
    
    def __str__(self):
        return super().__str__()
    
    def generate_code(self) -> str:
        parallel : str = CompilerConfig()['parallel']

        name : str = self.get_name()
        define_connected_output : str = gen_define_connected_output(self, 0)
        output_name : str = fix_identifier(self._output_varnames[0])
        input_name : str = fix_identifier(self._input_varnames[0])
        out_shape : int = self.infer_output_shape()
        out_size : int = math.prod(out_shape)
        for_loop_begin : str = gen_for_loop_begin(out_shape)
        for_loop_end : str = gen_for_loop_end(out_shape)
        index : str = gen_for_loop_index(out_shape)
        output_type : int = self.infer_output_type()
        output_type_str : str = onnx_type_to_c_dictionary(output_type)
        nflops_exp : int = 4
        nflops : int = out_size * (2 * nflops_exp + 3)

        if parallel == 'omp':
            for_loop_begin = gen_introduce_omp_in_for_loop_elem_by_elem(for_loop_begin, input_name, output_name)

        code : str = self._read_template_c("Tanh.c")

        code = self._expand_pattern(code, "$NAME", name)
        code = self._expand_pattern(code, "$DEFINE_CONNECTED_OUTPUT", define_connected_output)
        code = self._expand_pattern(code, "$OUTPUT_NAME", output_name)
        code = self._expand_pattern(code, "$INPUT_NAME", input_name)
        code = self._expand_pattern(code, "$INPUT_SIZE", str(out_size))
        code = self._expand_pattern(code, "$FOR_LOOPS_BEGIN", for_loop_begin)
        code = self._expand_pattern(code, "$FOR_LOOPS_END", for_loop_end)
        code = self._expand_pattern(code, "$INDEX", index)
        code = self._expand_pattern(code, "$OUTPUT_TYPE", output_type_str)
        code = self._expand_pattern(code, "$NFLOPS", str(nflops))

        return code
    
    def generate_includes_code_c(self) -> str:
        code : str = self._read_template_c("Tanh_inc.c")
        return code

    def generate_declaration_code_c(self) -> str:
        return ""
    
    def infer_output_shape(self) -> list[list[int]]:
        input : Node = self._inputs[0]
        shape : list[int] = node_shape(input)
        return shape
    
    def infer_output_type(self) -> int:
        input : Node = self._inputs[0]
        return input.infer_output_type()
    
    def get_op_type(self) -> str:
        return "Tanh"