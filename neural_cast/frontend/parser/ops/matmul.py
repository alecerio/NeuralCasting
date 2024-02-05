from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.common.common import fix_identifier
from neural_cast.frontend.exceptions.CompilerException import CompilerException
from neural_cast.frontend.parser.ops.common.common import node_shape
from neural_cast.frontend.parser.ops.common.common import node_type_binary_operation
from neural_cast.frontend.parser.ops.common.common import gen_define_connected_output
from neural_cast.frontend.common.common import onnx_type_to_c_dictionary

class MatMul(OpNode):
    def __init__(self, name : str):
        super().__init__(name)

    def __str__(self):
        return super().__str__()

    def generate_code(self) -> str:
        name : str = fix_identifier(self.get_name())
        define_connected_output : str = gen_define_connected_output(self, 0)
        output_name : str = fix_identifier(self._output_varnames[0])
        out_shape : list[int] = self.infer_output_shape()
        n_rows_left : int = out_shape[-2]
        n_cols_right : int = out_shape[-1]
        input_name_1 : str = fix_identifier(self._input_varnames[0])
        input_name_2 : str = fix_identifier(self._input_varnames[1])
        n_cols_left : int = self._infer_ncols_left()
        output_type : int = self.infer_output_type()
        output_type_str : str = onnx_type_to_c_dictionary(output_type)

        code : str = self._read_template_c("MatMul.c")

        code = self._expand_pattern(code, "$NAME", name)
        code = self._expand_pattern(code, "$DEFINE_CONNECTED_OUTPUT", define_connected_output)
        code = self._expand_pattern(code, "$INPUT_NAME_1", input_name_1)
        code = self._expand_pattern(code, "$INPUT_NAME_2", input_name_2)
        code = self._expand_pattern(code, "$OUTPUT_NAME", output_name)
        code = self._expand_pattern(code, "$N_ROWS_LEFT", str(n_rows_left))
        code = self._expand_pattern(code, "$N_COLS_RIGHT", str(n_cols_right))
        code = self._expand_pattern(code, "$N_COLS_LEFT", str(n_cols_left))
        code = self._expand_pattern(code, "$OUTPUT_TYPE", output_type_str)

        return code
    
    def generate_includes_code_c(self) -> str:
        return ""
    
    def generate_declaration_code_c(self) -> str:
        return ""
    
    def infer_output_shape(self) -> list[list[int]]:
        input1 : Node = self._inputs[0]
        shape1 : list[int] = node_shape(input1)

        input2 : Node = self._inputs[1]
        shape2 : list[int] = node_shape(input2)

        i1 : int = self._first_instance_non_one(shape1)
        i2 : int = self._first_instance_non_one(shape2)
        imax : int = max(i1, i2)
        shape1 = shape1[i1:]
        shape2 = shape2[i2:]

        if len(shape1) == 1 and len(shape2) == 1 and shape1[0] == shape2[0]:
            shape1 = [1, shape1[0]]
            shape2 = [shape2[0], 1]
        elif len(shape1) == 1 and shape1[0] == shape2[0]:
            shape1 = [1, shape1[0]]
        elif len(shape2) == 1 and shape2[0] == shape1[1]:
            shape2 = [shape2[0], 1]
        
        if shape1[-1] != shape2[-2]:
            raise CompilerException("Error: in MatMul, the number of columns of the left matrix must be equal to the number of columns of the right matrix")
        
        shape = [1] * imax + [shape1[-2], shape2[-1]]
        return shape
    
    def infer_output_type(self) -> int:
        input1 : Node = self._inputs[0]
        input2 : Node = self._inputs[1]
        return node_type_binary_operation(input1, input2, "matmul")

    def get_op_type(self) -> str:
        return "MatMul"
    
    def _infer_ncols_left(self) -> int:
        input_left : Node = self._inputs[0]
        shape_left : list[int] = node_shape(input_left)
        return shape_left[-1]
    
    def _first_instance_non_one(self, shape : list[int]) -> int:
        for i in range(len(shape)):
            if shape[i] != 1:
                return i
        return -1