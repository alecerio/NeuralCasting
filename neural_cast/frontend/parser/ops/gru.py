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

class GRU(OpNode):
    def __init__(self, name : str):
        super().__init__(name)
    
    def __str__(self):
        return super().__str__()
    
    def generate_code(self) -> str:
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