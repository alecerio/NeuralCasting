from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.parser.node.input_node import InputNode
from neural_cast.frontend.parser.node.init_node import InitializerNode
from neural_cast.frontend.parser.node.output_node import OutputNode
from neural_cast.frontend.parser.node_types.node_type import NodeType
from neural_cast.frontend.parser.node_types.tensor_type import TensorType
from neural_cast.frontend.common.common import fix_identifier
from neural_cast.frontend.exceptions.CompilerException import CompilerException
from neural_cast.frontend.common.common import onnx_type_to_c_dictionary
import math

class Constant(OpNode):
    def __init__(self, name : str, tensor, data_type):
        super().__init__(name)
        self._tensor = tensor
        self.data_type = data_type

    def __str__(self):
        super_str : str = super().__str__()
        tensor_str : str = "tensor: " + str(self._tensor)
        data_type_str : str = "data type: " + str(onnx_type_to_c_dictionary(self.data_type))
        return super_str + "\n" + \
                tensor_str + "\n" + \
                data_type_str

    def generate_code(self) -> str:
        return ""
    
    def generate_includes_code_c(self) -> str:
        return ""
    
    def generate_declaration_code_c(self) -> str:
        name : str = fix_identifier(self.get_name())
        data_type : str = onnx_type_to_c_dictionary(self.data_type)
        output_name : str = fix_identifier(self._output_varnames[0])
        out_shape : list[int] = self.infer_output_shape()
        out_size : int = math.prod(out_shape)
        const_values : str = self._gen_const_values_code()

        code : str = self._read_template_c("Constant_decl.c")

        code = self._expand_pattern(code, "$NAME", name)
        code = self._expand_pattern(code, "$TYPE", data_type)
        code = self._expand_pattern(code, "$OUTPUT_NAME", output_name)
        code = self._expand_pattern(code, "$SIZE", str(out_size))
        code = self._expand_pattern(code, "$CONSTANT_VALUES", const_values)

        return code
    
    def infer_output_shape(self) -> list[list[int]]:
        return self._tensor.shape
    
    def infer_output_type(self) -> int:
        return self.data_type
    
    def get_op_type(self) -> str:
        return "Constant"
    
    def _gen_const_values_code(self) -> str:
        flat_tensor = self._tensor.flatten()
        code : str = ""
        for val in flat_tensor:
            code += str(val) + ", "
        return code
            