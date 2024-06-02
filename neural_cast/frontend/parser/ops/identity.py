from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.common.common import fix_identifier
from neural_cast.frontend.parser.ops.common.common import node_shape
from neural_cast.frontend.parser.ops.common.common import gen_define_connected_output
import math

class Identity(OpNode):
    def __init__(self, name : str):
        super().__init__(name)
    
    def __str__(self):
        return super().__str__()
    
    def generate_code(self) -> str:
        name : str = self.get_name()
        out_shape = self.infer_output_shape()
        out_size : int = math.prod(out_shape)
        output_name : str = fix_identifier(self._output_varnames[0])
        input_name : str = fix_identifier(self._input_varnames[0])

        code : str = self._read_template_c("Identity.c")

        code = self._expand_pattern(code, "$(NAME)", name)
        code = self._expand_pattern(code, "$(OUTPUT_NAME)", output_name)
        code = self._expand_pattern(code, "$(INPUT_NAME)", input_name)
        code = self._expand_pattern(code, "$(OUTPUT_SIZE)", str(out_size))

        return code
    
    def generate_includes_code_c(self) -> str:
        return ""

    def generate_declaration_code_c(self) -> str:
        define_connected_output : str = gen_define_connected_output(self, 0)
        output_name : str = fix_identifier(self._output_varnames[0])

        code : str = self._read_template_c("Identity_decl.c")

        code = self._expand_pattern(code, "$(DEFINE_CONNECTED_OUTPUT)", define_connected_output)
        code = self._expand_pattern(code, "$(OUTPUT_NAME)", output_name)

        return code
    
    def infer_output_shape(self) -> list[list[int]]:
        input : Node = self._inputs[0]
        shape : list[int] = node_shape(input)
        return shape
    
    def infer_output_type(self) -> int:
        input : Node = self._inputs[0]
        return input.infer_output_type()
    
    def get_op_type(self) -> str:
        return "Identity"