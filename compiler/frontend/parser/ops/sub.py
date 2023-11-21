from compiler.frontend.parser.node.op_node import OpNode
from compiler.frontend.parser.node.node import Node
from compiler.frontend.parser.node.input_node import InputNode
from compiler.frontend.parser.node.init_node import InitializerNode
from compiler.frontend.parser.node_types.node_type import NodeType
from compiler.frontend.parser.node_types.tensor_type import TensorType
from compiler.frontend.parser.node.output_node import OutputNode
from compiler.frontend.exceptions.CompilerException import CompilerException
from compiler.frontend.common.common import fix_identifier
import math
from compiler.frontend.parser.ops.common.common import node_shape

class Sub(OpNode):
    def __init__(self, name : str):
        super().__init__(name)
    
    def __str__(self):
        return super.__str__()
    
    def generate_code(self) -> str:
        name : str = fix_identifier(self.get_name())
        define_connected_output : str = self._gen_define_connected_output()
        input_name_1 : str = fix_identifier(self._input_varnames[0])
        input_name_2 : str = fix_identifier(self._input_varnames[1])
        output_name : str = fix_identifier(self._output_varnames[0])
        in_shape : int = self.infer_output_shape()
        in_size : int = math.prod(in_shape)
        for_loop_begin : str = self._gen_for_loop_begin(in_shape)
        for_loop_end : str = self._gen_for_loop_end(in_shape)
        
        index_tot : str = self._gen_for_loop_index(in_shape)
        in_shape_1 = list(self._inputs[0].infer_output_shape())
        in_shape_2 = list(self._inputs[1].infer_output_shape())
        in_dims_1 = len(in_shape_1)
        in_dims_2 = len(in_shape_2)
        if in_dims_1 == in_dims_2 and in_shape_1 == in_shape_2:
            index_1 : str = index_tot
            index_2 : str = index_tot
        elif in_dims_1 > in_dims_2 and self._compatible_for_broadcasting(in_shape_1, in_shape_2):
            index_1 : str = index_tot
            if in_shape_2 == []:
                in_shape_2 = [0] * in_dims_1
            else:
                in_shape_2 = [0] * (in_dims_1 - in_dims_2) + in_shape_2
            index_2 : str = self._gen_for_loop_index(in_shape_2)
        elif in_dims_2 > in_dims_1 and self._compatible_for_broadcasting(in_shape_2, in_shape_1):
            index_2 : str = index_tot
            if in_shape_1 == []:
                in_shape_1 = [0] * in_dims_2
            else:
                in_shape_1 = [0] * (in_dims_2 - in_dims_1) + in_shape_1
            index_1 : str = self._gen_for_loop_index(in_shape_1)
        else:
            raise CompilerException("Error: incompatible input broadcasting in Sub operator")

        code : str = self._read_template_c("Sub.c")

        code = self._expand_pattern(code, "$NAME", name)
        code = self._expand_pattern(code, "$DEFINE_CONNECTED_OUTPUT", define_connected_output)
        code = self._expand_pattern(code, "$INPUT1_NAME", input_name_1)
        code = self._expand_pattern(code, "$INPUT2_NAME", input_name_2)
        code = self._expand_pattern(code, "$OUTPUT_NAME", output_name)
        code = self._expand_pattern(code, "$INPUT_SIZE", str(in_size))
        code = self._expand_pattern(code, "$FOR_LOOPS_BEGIN", for_loop_begin)
        code = self._expand_pattern(code, "$FOR_LOOPS_END", for_loop_end)
        code = self._expand_pattern(code, "$INDEX_TOT", index_tot)
        code = self._expand_pattern(code, "$INDEX_1", index_1)
        code = self._expand_pattern(code, "$INDEX_2", index_2)

        return code
    
    def generate_declaration_code_c(self) -> str:
        return ""
    
    def generate_includes_code_c(self) -> str:
        return ""
    
    def infer_output_shape(self) -> list[list[int]]:
        input1 : Node = self._inputs[0]
        shape1 : list[int] = node_shape(input1)

        input2 : Node = self._inputs[1]
        shape2 : list[int] = node_shape(input2)

        list_shape1 : list[int] = list(shape1)
        list_shape2 : list[int] = list(shape2)
        dims_shape1 : int = len(list_shape1)
        dims_shape2 : int = len(list_shape2)

        if dims_shape1 == dims_shape2 and list_shape1 == list_shape2:
            shape = list_shape1
        elif dims_shape1 > dims_shape2:
            compatible_broadcasting : bool = self._compatible_for_broadcasting(list_shape1, list_shape2)
            if compatible_broadcasting:
                shape = list_shape1
            else:
                raise CompilerException("Error: incompatible input broadcasting in Sub operator. Shape 2 does not fit in shape 1 for broadcasting.")
        elif dims_shape2 > dims_shape1:
            compatible_broadcasting : bool = self._compatible_for_broadcasting(list_shape2, list_shape1)
            if compatible_broadcasting:
                shape = list_shape2
            else:
                raise CompilerException("Error: incompatible input broadcasting in Sub operator. Shape 1 does not fit in shape 2 for broadcasting.")
        else:
            raise CompilerException("Error: incompatible input broadcasting in Sub operator.  Equal number of dimensions, but different shape.")

        
        return shape
    
    def get_op_type(self) -> str:
        return "Sub"
    
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
        first : int = n_dims
        for i in range(n_dims):
            if shape[i] != 0:
                first = i
                break
        if first == n_dims:
            return "0"
        for i in range(first, n_dims):
            index : str = "i" + str(i)
            size : int = 1
            for j in range(i+1, n_dims):
                size *= shape[j]
            code += index + "*" + str(size)
            if i < n_dims-1:
                code += " + "
        return code
    
    def _compatible_for_broadcasting(self, larger_shape : list[int], smaller_shape : list[int]) -> bool:
        if len(smaller_shape) > len(larger_shape):
            raise CompilerException("Error: in compatible broadcasting check, smaller shape cannot have more dimensions than larger shape")
        if smaller_shape == []:
            return True
        subshape : list[int] = larger_shape[-len(smaller_shape):]
        if subshape == smaller_shape:
            return True
        else:
            return False