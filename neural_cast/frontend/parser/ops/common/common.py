from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.parser.node.input_node import InputNode
from neural_cast.frontend.parser.node.init_node import InitializerNode
from neural_cast.frontend.parser.node.output_node import OutputNode
from neural_cast.frontend.parser.node_types.node_type import NodeType
from neural_cast.frontend.parser.node_types.tensor_type import TensorType
from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.init_node import InitializerNode
from neural_cast.frontend.exceptions.CompilerException import CompilerException
from neural_cast.frontend.common.common import signed_integer_types
from neural_cast.frontend.common.common import unsigned_integer_types
from neural_cast.frontend.common.common import floating_point_types
import re

def node_shape(node : Node) -> list[int]:
    shape = node.infer_output_shape()
    if len(shape) == 0:
        return []
    return shape

def node_type_binary_operation(node1 : Node, node2 : Node, op_name_for_error_message : str = None) -> int:
    node1_type : int = node1.infer_output_type()
    node2_type : int = node2.infer_output_type()

    ss : bool = node1_type in signed_integer_types and node2_type in signed_integer_types
    su : bool = node1_type in signed_integer_types and node2_type in unsigned_integer_types
    us : bool = node2_type in signed_integer_types and node1_type in unsigned_integer_types
    uu : bool = node2_type in unsigned_integer_types and node1_type in unsigned_integer_types
    fi : bool = node1_type in floating_point_types and node2_type in (signed_integer_types + unsigned_integer_types)
    i_f : bool = node2_type in floating_point_types and node1_type in (signed_integer_types + unsigned_integer_types)
    ff : bool = node1_type in floating_point_types and node2_type in floating_point_types

    if ss:
        return 6
    elif su or us or uu:
        return 12
    elif fi or i_f or ff:
        return 1
    else:
        if op_name_for_error_message == None:
            op_name_for_error_message = "binary operation"
        CompilerException("Error: unsupported type for " + op_name_for_error_message)

def compatible_for_broadcasting(larger_shape : list[int], smaller_shape : list[int]) -> bool:
        if len(smaller_shape) > len(larger_shape):
            raise CompilerException("Error: in compatible broadcasting check, smaller shape cannot have more dimensions than larger shape")
        if smaller_shape == []:
            return True
        subshape : list[int] = larger_shape[-len(smaller_shape):]
        if subshape == smaller_shape:
            return True
        else:
            return False

def gen_define_connected_output(node : Node, output_index : int) -> str:
        
    if isinstance(node, OpNode):
         outputs = node.get_output_nodes_list()
    elif isinstance(node, InputNode):
        outputs = node.get_output_nodes_list()
    elif isinstance(node, InitializerNode):
         outputs = node.get_output_nodes_list()
    else:
         raise CompilerException("Error: impossible to define connected outputs for an output node")

    connected_output : bool = isinstance(outputs[output_index], OutputNode)
        
    if connected_output:
        define_connected_output = ""
    else:
        define_connected_output = "#define CONNECTED_OUTPUT"
    
    return define_connected_output

def gen_for_loop_begin(shape : list[int]) -> str:
    code : str = ""
    n_dims = len(shape)
    for dim in range(n_dims):
        size : int = shape[dim]
        index : str = "i" + str(dim)
        code += "for(" + \
                "int " + index + "=0; " + index + "<" + str(size) + "; " + index + "++) {\n"
    return code

def gen_for_loop_end(shape : list[int]) -> str:
    code : str = ""
    n_dims = len(shape)
    for _ in range(n_dims):
        code += "}\n"
    return code

def gen_for_loop_index(shape : list[int]) -> str:
    code : str = ""
    n_dims : int = len(shape)
    first : int = n_dims
    for i in range(n_dims):
        if shape[i] != 0:
            first = i
            break
    if first == n_dims:
        return "0"
    for i in range(first+1, n_dims):
        index : str = "i" + str(i)
        size : int = 1
        for j in range(i+1, n_dims):
            size *= shape[j]
        code += index + "*" + str(size)
        if i < n_dims-1:
            code += " + "
    return code

def infer_output_shape_for_element_wise_binary_operators(input1 : Node, input2 : Node, op_name_for_error_message : str = None) -> list[list[int]]:
    if op_name_for_error_message == None:
        op_name_for_error_message = "binary"
    
    shape1 : list[int] = node_shape(input1)
    shape2 : list[int] = node_shape(input2)

    list_shape1 : list[int] = list(shape1)
    list_shape2 : list[int] = list(shape2)
    dims_shape1 : int = len(list_shape1)
    dims_shape2 : int = len(list_shape2)
    
    old_list_shape1 = list_shape1
    old_list_shape2 = list_shape2
    i1 : int = _first_instance_non_one(list_shape1)
    i2 : int = _first_instance_non_one(list_shape2)
    list_shape1 = list_shape1[i1:]
    dims_shape1 = len(list_shape1)
    list_shape2 = list_shape2[i2:]
    dims_shape2 = len(list_shape2)

    if dims_shape1 == dims_shape2 and list_shape1 == list_shape2:
        shape = list_shape1
    elif dims_shape1 > dims_shape2:
        compatible_broadcasting : bool = compatible_for_broadcasting(list_shape1, list_shape2)
        if compatible_broadcasting:
            shape = list_shape1
        else:
            raise CompilerException("Error: incompatible input broadcasting in " + op_name_for_error_message + " operator. Shape 2 does not fit in shape 1 for broadcasting.")
    elif dims_shape2 > dims_shape1:
        compatible_broadcasting : bool = compatible_for_broadcasting(list_shape2, list_shape1)
        if compatible_broadcasting:
            shape = list_shape2
        else:
            raise CompilerException("Error: incompatible input broadcasting in " + op_name_for_error_message + " operator. Shape 1 does not fit in shape 2 for broadcasting.")
    else:
        raise CompilerException("Error: incompatible input broadcasting in " + op_name_for_error_message + " operator.  Equal number of dimensions, but different shape.")

    if len(old_list_shape1) > len(old_list_shape2):
        shape = old_list_shape1
    else:
        shape = old_list_shape2

    return shape

def gen_element_wise_broadcasting_indices(input1 : Node, input2 : Node, output_shape : list[int], op_name_for_error_message = None) -> list[int]:
    if op_name_for_error_message == None:
        op_name_for_error_message = "binary"

    index_tot : str = gen_for_loop_index(output_shape)
    in_shape_1 = list(input1.infer_output_shape())
    in_shape_2 = list(input2.infer_output_shape())
    in_dims_1 = len(in_shape_1)
    in_dims_2 = len(in_shape_2)
    if in_dims_1 == in_dims_2 and in_shape_1 == in_shape_2:
        index_1 : str = index_tot
        index_2 : str = index_tot
    elif in_dims_1 > in_dims_2:
        index_1 : str = index_tot
        if in_shape_2 == []:
            in_shape_2 = [0] * in_dims_1
        else:
            in_shape_2 = [0] * (in_dims_1 - in_dims_2) + in_shape_2
        index_2 : str = gen_for_loop_index(in_shape_2)
    elif in_dims_2 > in_dims_1:
        index_2 : str = index_tot
        if in_shape_1 == []:
            in_shape_1 = [0] * in_dims_2
        else:
            in_shape_1 = [0] * (in_dims_2 - in_dims_1) + in_shape_1
        index_1 : str = gen_for_loop_index(in_shape_1)
    else:
        raise CompilerException("Error: incompatible input broadcasting in " + op_name_for_error_message + " operator")
    
    return [index_tot, index_1, index_2]

def gen_const_values_code(tensor) -> str:
    flat_tensor = tensor.flatten()
    code : str = ""
    for val in flat_tensor:
        code += str(val) + ", "
    return code

def gen_introduce_omp_in_for_loop_elemen_wise(for_loop_begin : str, A_name : str, B_name : str, C_name : str) -> str:
    for_loops : list[str] = for_loop_begin.split('\n')
    for_loops = [x for x in for_loops if x != '']
    bounds : list[int] = []
    for loop in for_loops:
        match = re.search(r'<(.*?);', loop)
        if match:
            bound_str : str = match.group(1)
            bounds.append(int(bound_str))
    max_bound = max(bounds)
    max_index = bounds.index(max_bound)
    
    private_indices : str = ''
    for n_index in range(0, max_index):
        index : str = 'i' + str(n_index)
        private_indices += index
        if n_index < max_index-1:
            private_indices += ', '
    omp_directive : str = '#pragma omp parallel for shared(tensor_' + A_name + ', tensor_' + B_name + ', tensor_' + C_name + ') private(' + private_indices + ')'
    
    parall_for_loops : str = ''
    for index, loop in enumerate(for_loops):
        if index == max_index:
            parall_for_loops += omp_directive + '\n'
        parall_for_loops += loop + '\n'
    
    return parall_for_loops

def _first_instance_non_one(shape : list[int]) -> int:
    for i in range(len(shape)):
        if shape[i] != 1:
            return i
    return -1