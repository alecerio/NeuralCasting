"""
Author: Alessandro Cerioli
Description:
    The class ReLu represents the ReLU (Rectified Linear Unit) activation function.
"""

from compiler.frontend.parser.node.op_node import OpNode
from compiler.frontend.parser.node.input_node import InputNode
from compiler.frontend.parser.node.node import Node
from compiler.frontend.parser.node.output_node import OutputNode
from compiler.frontend.parser.node_types.node_type import NodeType
from compiler.frontend.parser.node_types.tensor_type import TensorType
from compiler.frontend.common.common import fix_identifier
from compiler.frontend.exceptions.CompilerException import CompilerException

class ReLu(OpNode):
    def __init__(self, name : str):
        super().__init__(name)
    
    def __str__(self):
        return super().__str__()

    def generate_code(self) -> str:
        """
        Method to generate the code related to the ReLu function.

        Return:
            The code related to the ReLu function.
        """
        
        define_connected_output : str = self._gen_define_connected_output()
        shape : list[int] = self.infer_output_shape()
        name : str = fix_identifier(self._name)
        input_name : str = fix_identifier(self._input_varnames[0])
        output_name : str = fix_identifier(self._output_varnames[0])

        code : str = self._read_template_c("ReLu.c")

        code = self._expand_pattern(code, "$DEFINE_CONNECTED_OUTPUT", define_connected_output)
        code = self._expand_pattern(code, "$NAME", name)
        code = self._expand_pattern(code, "$INPUT_NAME", input_name)
        code = self._expand_pattern(code, "$OUTPUT_NAME", output_name)
        code = self._expand_pattern(code, "$INPUT_SIZE", str(shape[1]))

        return code
    
    def infer_output_shape(self) -> list[list[int]]:
        input : Node = self._inputs[0]
        if isinstance(input, InputNode):
            t : NodeType = input.get_node_type()
            if isinstance(t, TensorType):
                shape = t.get_shape()
            else:
                raise CompilerException("Error: input node type not supported")
        elif isinstance(input, OpNode):
            shape = input.infer_output_shape()
        else:
            raise CompilerException("Error: invalid ReLu input node")
        return shape
    
    def generate_declaration_code_c(self) -> str:
        return ""

    def _gen_define_connected_output(self, ) -> str:
        connected_output : bool = isinstance(self._outputs[0], OutputNode)
        
        if connected_output:
            define_connected_output = ""
        else:
            define_connected_output = "#define CONNECTED_OUTPUT"
        
        return define_connected_output

    def get_op_type(self) -> str:
        return "ReLu"
    
    def generate_includes_code_c(self) -> str:
        return ""