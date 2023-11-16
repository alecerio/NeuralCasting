from compiler.frontend.parser.node.op_node import OpNode
from compiler.frontend.parser.node.node import Node
from compiler.frontend.parser.node.output_node import OutputNode
from compiler.frontend.parser.node.input_node import InputNode
from compiler.frontend.parser.node.init_node import InitializerNode
from compiler.frontend.parser.node_types.node_type import NodeType
from compiler.frontend.parser.node_types.tensor_type import TensorType
from compiler.frontend.common.common import fix_identifier
from compiler.frontend.exceptions.CompilerException import CompilerException
import math

class Gather(OpNode):
    def __init__(self, name : str, axis : int):
        super().__init__(name)
        self._axis : int = axis

    def __str__(self):
        return super().__str__()
    
    def get_axis(self) -> int:
        return self._axis
    
    def set_axis(self, axis : int) -> None:
        if axis != 0 and axis != 1:
            raise CompilerException("Error: gather operator only supports axis 0 and 1")
        self._axis = axis

    def generate_code(self) -> str:
        name : str = fix_identifier(self.get_name())
        define_connected_output : str = self._gen_define_connected_output()
        output_shape : list[int] = self.infer_output_shape()
        output_size : int = math.prod(output_shape)
        indices_shape : list[int] = self._get_indices_shape()
        indices_size : int = math.prod(indices_shape)
        indices_name : str = fix_identifier(self._inputs[1].get_name())
        values_col : int = self._get_values_shape()[1]
        output_name : str = fix_identifier(self._output_varnames[0])
        values_name : str = fix_identifier(self._inputs[0].get_name())
        values_row : int = self._get_values_shape()[0]

        code : str = self._read_template_c("Gather.c")

        code = self._expand_pattern(code, "$NAME", name)
        code = self._expand_pattern(code, "$DEFINE_CONNECTED_OUTPUT", define_connected_output)
        code = self._expand_pattern(code, "$OUTPUT_SIZE", str(output_size))
        code = self._expand_pattern(code, "$AXIS", str(self._axis))
        code = self._expand_pattern(code, "$IND_SIZE", str(indices_size))
        code = self._expand_pattern(code, "$INDICES", str(indices_name))
        code = self._expand_pattern(code, "$VAL_SIZE_COLS", str(values_col))
        code = self._expand_pattern(code, "$OUTPUT_NAME", output_name)
        code = self._expand_pattern(code, "$VALUES", values_name)
        code = self._expand_pattern(code, "$VAL_SIZE_ROWS", str(values_row))

        return code
    
    def generate_includes_code_c(self) -> str:
        return ""
    
    def generate_declaration_code_c(self) -> str:
        return ""
    
    def infer_output_shape(self) -> list[list[int]]:
        val_shape : list[int] = self._get_values_shape()
        ind_shape : list[int] = self._get_indices_shape()
        if self._axis == 0:
            return [val_shape[1], ind_shape[0], ind_shape[1]]
        elif self._axis == 1:
            return [val_shape[0], ind_shape[0], ind_shape[1]]
        else:
            raise CompilerException("Error: gather operator only supports axis 0 and 1")
    
    def get_op_type(self) -> str:
        return "Gather"
    
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
        elif isinstance(node, InitializerNode):
            shape = node.get_tensor().shape
        else:
            raise CompilerException("Error: invalid MatMul input node")
        return shape
    
    def _get_values_shape(self) -> list[int]:
        values : Node = self._inputs[0]
        val_shape : list[int] = self._node_shape(values)
        return val_shape
    
    def _get_indices_shape(self) -> list[int]:
        indices : Node = self._inputs[1]
        ind_shape : list[int] = self._node_shape(indices)
        return ind_shape