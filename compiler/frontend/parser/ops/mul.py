from compiler.frontend.parser.node.op_node import OpNode
from compiler.frontend.parser.node.node import Node
from compiler.frontend.parser.node.output_node import OutputNode
from compiler.frontend.parser.node.input_node import InputNode
from compiler.frontend.parser.node_types.node_type import NodeType
from compiler.frontend.parser.node_types.tensor_type import TensorType
from compiler.frontend.common.common import fix_identifier
from compiler.frontend.exceptions.CompilerException import CompilerException

class Mul(OpNode):
    def __init__(self, name : str):
        super().__init__(name)

    def __str__(self):
        return super().__str__()

    def generate_code(self) -> str:
        name : str = fix_identifier(self.get_name())
        define_connected_output : str = self._gen_define_connected_output()
        input_name_1 : str = fix_identifier(self._input_varnames[0])
        input_name_2 : str = fix_identifier(self._input_varnames[1])
        output_name : str = fix_identifier(self._output_varnames[0])
        in_size : int = self.infer_output_shape()[1]

        code : str = self._read_template_c("Mul.c")

        code = self._expand_pattern(code, "$NAME", name)
        code = self._expand_pattern(code, "$DEFINE_CONNECTED_OUTPUT", define_connected_output)
        code = self._expand_pattern(code, "$INPUT1_NAME", input_name_1)
        code = self._expand_pattern(code, "$INPUT2_NAME", input_name_2)
        code = self._expand_pattern(code, "$OUTPUT_NAME", output_name)
        code = self._expand_pattern(code, "$INPUT_SIZE", str(in_size))

        return code
    
    def generate_includes_code_c(self) -> str:
        return ""
    
    def generate_declaration_code_c(self) -> str:
        return ""
    
    def infer_output_shape(self) -> list[list[int]]:
        input1 : Node = self._inputs[0]
        shape1 : list[int] = self._node_shape(input1)

        input2 : Node = self._inputs[1]
        shape2 : list[int] = self._node_shape(input2)

        if shape1 != shape2:
            raise CompilerException("Error: inputs in Mul operator must have the same shape")
        
        return shape1
    
    def get_op_type(self) -> str:
        return "Mul"
    
    def _gen_define_connected_output(self, ) -> str:
        connected_output : bool = isinstance(self._outputs[0], OutputNode)
        
        if connected_output:
            define_connected_output = ""
        else:
            define_connected_output = "#define CONNECTED_OUTPUT"
        
        return define_connected_output

    def _node_shape(self, node : Node) -> list[int]:
        if isinstance(node, InputNode):
            t : NodeType = node.get_node_type()
            if isinstance(t, TensorType):
                shape = t.get_shape()
            else:
                raise CompilerException("Error: input node type not supported")
        elif isinstance(node, OpNode):
            shape = node.infer_output_shape()
        else:
            raise CompilerException("Error: invalid Mul input node")
        return shape