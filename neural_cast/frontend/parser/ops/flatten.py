import math
from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.common.common import fix_identifier
from neural_cast.frontend.parser.ops.common.common import gen_define_connected_output

class Flatten(OpNode):
    def __init__(self, name : str):
        super().__init__(name)

    def __str__(self):
        return super.__str__()

    def generate_code(self) -> str:
        #parallel : str = CompilerConfig()['parallel']
        name : str = fix_identifier(self.get_name())
        output_name : str = fix_identifier(self._output_varnames[0])
        input_name : str = fix_identifier(self._input_varnames[0])
        [_, out_size] = self.infer_output_shape()

        #if parallel == 'omp':
        #    for_loop_begin = gen_introduce_omp_in_for_loop_elem_by_elem(for_loop_begin, input_name, output_name)

        code : str = self._read_template_c("Flatten.c")

        code = self._expand_pattern(code, "$(NAME)", name)
        code = self._expand_pattern(code, "$(INPUT_NAME)", input_name)
        code = self._expand_pattern(code, "$(OUTPUT_NAME)", output_name)
        code = self._expand_pattern(code, "$(OUTPUT_SIZE)", str(out_size))

        return code
    
    def generate_declaration_code_c(self) -> str:
        define_connected_output : str = gen_define_connected_output(self, 0)
        output_name : str = fix_identifier(self._output_varnames[0])

        code : str = self._read_template_c("Flatten_decl.c")

        code = self._expand_pattern(code, "$(DEFINE_CONNECTED_OUTPUT)", define_connected_output)
        code = self._expand_pattern(code, "$(OUTPUT_NAME)", output_name)

        return code
    
    def infer_output_shape(self) -> list[list[int]]:
        in_shape = self._inputs[0].infer_output_shape()
        out_size = math.prod(in_shape)
        out_shape = [1, out_size]
        return out_shape
    
    def infer_output_type(self) -> int:
        input : Node = self._inputs[0]
        return input.infer_output_type()
    
    def get_op_type(self) -> str:
        return "Flatten"
    
    def generate_includes_code_c(self) -> str:
        return ""