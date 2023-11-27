from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.output_node import OutputNode
from neural_cast.frontend.parser.node.input_node import InputNode
from neural_cast.frontend.parser.node.init_node import InitializerNode
from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.parser.node_types.node_type import NodeType
from neural_cast.frontend.parser.node_types.tensor_type import TensorType
from neural_cast.frontend.common.common import fix_identifier
from neural_cast.frontend.exceptions.CompilerException import CompilerException
import math
from neural_cast.frontend.parser.ops.common.common import node_shape

class Tanh(OpNode):
    def __init__(self, name : str):
        super().__init__(name)
    
    def __str__(self):
        return super().__str__()
    
    def generate_code(self) -> str:
        name : str = self.get_name()
        define_connected_output : str = self._gen_define_connected_output()
        output_name : str = fix_identifier(self._output_varnames[0])
        input_name : str = fix_identifier(self._input_varnames[0])
        in_shape : int = self.infer_output_shape()
        in_size : int = math.prod(in_shape)
        for_loop_begin : str = self._gen_for_loop_begin(in_shape)
        for_loop_end : str = self._gen_for_loop_end(in_shape)
        index : str = self._gen_for_loop_index(in_shape)

        code : str = self._read_template_c("Tanh.c")

        code = self._expand_pattern(code, "$NAME", name)
        code = self._expand_pattern(code, "$DEFINE_CONNECTED_OUTPUT", define_connected_output)
        code = self._expand_pattern(code, "$OUTPUT_NAME", output_name)
        code = self._expand_pattern(code, "$INPUT_NAME", input_name)
        code = self._expand_pattern(code, "$INPUT_SIZE", str(in_size))
        code = self._expand_pattern(code, "$FOR_LOOPS_BEGIN", for_loop_begin)
        code = self._expand_pattern(code, "$FOR_LOOPS_END", for_loop_end)
        code = self._expand_pattern(code, "$INDEX", index)

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
    
    def get_op_type(self) -> str:
        return "Tanh"
    
    def _gen_define_connected_output(self, ) -> str:
        connected_output : bool = isinstance(self._outputs[0], OutputNode)
        
        if connected_output:
            define_connected_output = ""
        else:
            define_connected_output = "#define CONNECTED_OUTPUT"
        
        return define_connected_output
    
    def _gen_for_loop_begin(self, shape : list[int]) -> str:
        code : str = ""
        n_dims = len(shape)
        for dim in range(n_dims):
            size : int = shape[dim]
            index : str = "i" + str(dim)
            code += "for(" + \
                    "int " + index + "=0; " + index + "<" + str(size) + "; " + index + "++) {\n"
        return code
    
    def _gen_for_loop_end(self, shape : list[int]) -> str:
        code : str = ""
        n_dims = len(shape)
        for _ in range(n_dims):
            code += "}\n"
        return code
    
    def _gen_for_loop_index(self, shape : list[int]) -> str:
        code : str = ""
        n_dims : int = len(shape)
        for i in range(n_dims):
            index : str = "i" + str(i)
            size : int = 1
            for j in range(i+1, n_dims):
                size *= shape[j]
            code += index + "*" + str(size)
            if i < n_dims-1:
                code += " + "
        return code