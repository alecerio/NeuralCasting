from compiler.frontend.parser.node.op_node import OpNode
from compiler.frontend.parser.node.node import Node
from compiler.frontend.parser.node.output_node import OutputNode
from compiler.frontend.parser.node.input_node import InputNode
from compiler.frontend.parser.node.init_node import InitializerNode
from compiler.frontend.parser.node_types.node_type import NodeType
from compiler.frontend.parser.node_types.tensor_type import TensorType
from compiler.frontend.common.common import fix_identifier
from compiler.frontend.exceptions.CompilerException import CompilerException

class MatMul(OpNode):
    def __init__(self, name : str):
        super().__init__(name)

    def __str__(self):
        return super().__str__()

    def generate_code(self) -> str:
        name : str = fix_identifier(self.get_name())
        define_connected_output : str = self._gen_define_connected_output()
        output_name : str = fix_identifier(self._output_varnames[0])
        out_shape : list[int] = self.infer_output_shape()
        n_rows_left : int = out_shape[0]
        n_cols_right : int = out_shape[1]
        input_name_1 : str = fix_identifier(self._input_varnames[0])
        input_name_2 : str = fix_identifier(self._input_varnames[1])
        n_cols_left : int = self._infer_ncols_left()

        code : str = self._read_template_c("MatMul.c")

        code = self._expand_pattern(code, "$NAME", name)
        code = self._expand_pattern(code, "$DEFINE_CONNECTED_OUTPUT", define_connected_output)
        code = self._expand_pattern(code, "$INPUT_NAME_1", input_name_1)
        code = self._expand_pattern(code, "$INPUT_NAME_2", input_name_2)
        code = self._expand_pattern(code, "$OUTPUT_NAME", output_name)
        code = self._expand_pattern(code, "$N_ROWS_LEFT", str(n_rows_left))
        code = self._expand_pattern(code, "$N_COLS_RIGHT", str(n_cols_right))
        code = self._expand_pattern(code, "$N_COLS_LEFT", str(n_cols_left))

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

        if shape1[1] != shape2[0]:
            raise CompilerException("Error: in MatMul, the number of columns of the left matrix must be equal to the number of columns of the right matrix")
        
        shape = [shape1[0], shape2[1]]
        
        return shape
    
    def get_op_type(self) -> str:
        return "MatMul"
    
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
    
    def _infer_ncols_left(self) -> int:
        input_left : Node = self._inputs[0]
        shape_left : list[int] = self._node_shape(input_left)
        return shape_left[1]