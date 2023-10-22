from compiler.frontend.parser.node.node import Node
from compiler.frontend.parser.node_types.node_type import NodeType

class InputNode(Node):
    def __init__(self, name : str, type : NodeType):
        super().__init__(name=name)
        self._type = type
    
    def get_node_type(self) ->  NodeType:
        return self._type
    
    def set_node_type(self, type : NodeType):
        self._type = type