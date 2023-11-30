from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.output_node import OutputNode
from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.parser.node.input_node import InputNode
from neural_cast.frontend.parser.node.init_node import InitializerNode
from neural_cast.frontend.parser.node_types.node_type import NodeType
from neural_cast.frontend.parser.node_types.tensor_type import TensorType
from neural_cast.frontend.common.common import fix_identifier
from neural_cast.frontend.exceptions.CompilerException import CompilerException
import math
from neural_cast.frontend.parser.ops.common.common import node_shape
from neural_cast.frontend.parser.ops.common.common import gen_define_connected_output
from neural_cast.frontend.parser.ops.common.common import gen_for_loop_begin
from neural_cast.frontend.parser.ops.common.common import gen_for_loop_end
from neural_cast.frontend.parser.ops.common.common import gen_for_loop_index
from neural_cast.frontend.common.common import onnx_type_to_c_dictionary

class Sigmoid(OpNode):
    def __init__(self, name : str):
        super().__init__(name)

    def __str__(self):
        return super.__str__()

    def generate_code(self) -> str:
        name : str = fix_identifier(self.get_name())
        define_connected_output : str = gen_define_connected_output(self, 0)
        input_name : str = fix_identifier(self._input_varnames[0])
        output_name : str = fix_identifier(self._output_varnames[0])
        in_shape : int = self.infer_output_shape()
        in_size : int = math.prod(in_shape)
        for_loop_begin : str = gen_for_loop_begin(in_shape)
        for_loop_end : str = gen_for_loop_end(in_shape)
        index : str = gen_for_loop_index(in_shape)
        output_type : int = self.infer_output_type()
        output_type_str : str = onnx_type_to_c_dictionary(output_type)

        code : str = self._read_template_c("Sigmoid.c")

        code = self._expand_pattern(code, "$NAME", name)
        code = self._expand_pattern(code, "$DEFINE_CONNECTED_OUTPUT", define_connected_output)
        code = self._expand_pattern(code, "$INPUT_NAME", input_name)
        code = self._expand_pattern(code, "$OUTPUT_NAME", output_name)
        code = self._expand_pattern(code, "$INPUT_SIZE", str(in_size))
        code = self._expand_pattern(code, "$FOR_LOOPS_BEGIN", for_loop_begin)
        code = self._expand_pattern(code, "$FOR_LOOPS_END", for_loop_end)
        code = self._expand_pattern(code, "$INDEX", index)
        code = self._expand_pattern(code, "$OUTPUT_TYPE", output_type_str)

        return code
    
    def generate_declaration_code_c(self) -> str:
        return ""
    
    def infer_output_shape(self) -> list[list[int]]:
        input : Node = self._inputs[0]
        shape : list[int] = node_shape(input)
        return shape
    
    def infer_output_type(self) -> int:
        input1 : Node = self._inputs[0]
        return input1.infer_output_type()
    
    def get_op_type(self) -> str:
        return "Sigmoid"
    
    def generate_includes_code_c(self) -> str:
        optype : str = self.get_op_type()
        code : str = self._read_template_c("Sigmoid_inc.c")
        code = self._expand_pattern(code, "$TYPE", optype)
        return code