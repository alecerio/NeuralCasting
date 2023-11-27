from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.parser.node.input_node import InputNode
from neural_cast.frontend.parser.node_types.node_type import NodeType
from neural_cast.frontend.parser.node_types.tensor_type import TensorType
from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.init_node import InitializerNode
from neural_cast.frontend.exceptions.CompilerException import CompilerException

def node_shape(node : Node) -> list[int]:
    shape = node.infer_output_shape()
    if len(shape) == 0:
        return []
    #while shape[0] == 1:
    #    shape = shape[1:]
    #    if len(shape) == 0:
    #        return []
    return shape