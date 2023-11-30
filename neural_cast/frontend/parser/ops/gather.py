from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.common.common import fix_identifier
from neural_cast.frontend.exceptions.CompilerException import CompilerException
from neural_cast.frontend.parser.ops.common.common import node_shape
from neural_cast.frontend.parser.ops.common.common import gen_define_connected_output
from neural_cast.frontend.common.common import onnx_type_to_c_dictionary
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
        define_connected_output : str = gen_define_connected_output(self, 0)
        output_shape : list[int] = self.infer_output_shape()
        output_size : int = math.prod(output_shape)
        indices_shape : list[int] = self._get_indices_shape()
        indices_size : int = math.prod(indices_shape)
        indices_name : str = fix_identifier(self._inputs[1].get_name())
        
        val_shape : list[int] = self._get_values_shape()
        if len(val_shape) == 2: values_col : int = val_shape[1]
        else: values_col = 1
        output_name : str = fix_identifier(self._output_varnames[0])
        values_name : str = fix_identifier(self._inputs[0].get_name())
        values_row : int = self._get_values_shape()[0]

        values_datatype : int = self._get_values_type()
        values_datatype_str : str = onnx_type_to_c_dictionary(values_datatype)

        indices_datatype : int = self._get_indices_type()
        indices_datatype_str : str = onnx_type_to_c_dictionary(indices_datatype)

        output_datatype : int = self.infer_output_type()
        output_datatype_str : str = onnx_type_to_c_dictionary(output_datatype)

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
        code = self._expand_pattern(code, "$VALUE_TYPE", str(values_datatype_str))
        code = self._expand_pattern(code, "$INDEX_TYPE", str(indices_datatype_str))
        code = self._expand_pattern(code, "$OUTPUT_TYPE", str(output_datatype_str))

        return code
    
    def generate_includes_code_c(self) -> str:
        return ""
    
    def generate_declaration_code_c(self) -> str:
        return ""
    
    def infer_output_shape(self) -> list[list[int]]:
        val_shape : list[int] = self._get_values_shape()
        if self._axis == 0: val_shape = [val_shape[1]]
        elif self._axis == 1: val_shape = [val_shape[0]]
        ind_shape : list[int] = self._get_indices_shape()
        shape = []
        if self._axis == 0:
            for i in range(len(ind_shape)): shape.append(ind_shape[i])
            for i in range(len(val_shape)): shape.append(val_shape[i])
        elif self._axis == 1:
            for i in range(len(val_shape)): shape.append(val_shape[i])
            for i in range(len(ind_shape)): shape.append(ind_shape[i])
        return shape
    
    def infer_output_type(self) -> int:
        return self._get_values_type()

    def get_op_type(self) -> str:
        return "Gather"
    
    def _get_values_shape(self) -> list[int]:
        values : Node = self._inputs[0]
        val_shape : list[int] = node_shape(values)
        return val_shape
    
    def _get_indices_shape(self) -> list[int]:
        indices : Node = self._inputs[1]
        ind_shape : list[int] = node_shape(indices)
        return ind_shape
    
    def _get_values_type(self) -> int:
        values : Node = self._inputs[0]
        values_datatype : int = values.infer_output_type()
        return values_datatype
    
    def _get_indices_type(self) -> int:
        indices : Node = self._inputs[1]
        indices_datatype : int = indices.infer_output_type()
        return indices_datatype
