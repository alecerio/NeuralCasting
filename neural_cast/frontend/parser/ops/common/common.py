from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.parser.node.input_node import InputNode
from neural_cast.frontend.parser.node_types.node_type import NodeType
from neural_cast.frontend.parser.node_types.tensor_type import TensorType
from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.init_node import InitializerNode
from neural_cast.frontend.exceptions.CompilerException import CompilerException
from neural_cast.frontend.common.common import signed_integer_types
from neural_cast.frontend.common.common import unsigned_integer_types
from neural_cast.frontend.common.common import floating_point_types

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