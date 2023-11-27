from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.parser.node_types.node_type import NodeType
from neural_cast.frontend.exceptions.CompilerException import CompilerException

class OutputNode(Node):
    def __init__(self, name : str, type : NodeType):
        super().__init__(name=name)
        self._type = type
        self._inputs : list[Node] = []
    
    def __str__(self):
        super_str : str = super().__str__()
        output_node_type : str = "output node type:\n" + str(self._type)
        inputs_name : str = "input nodes: "
        for input in self._inputs:
            inputs_name = inputs_name + input.get_name() + ", " 
        return  super_str + "\n" + \
                output_node_type + "\n" + \
                inputs_name

    def get_node_type(self) ->  NodeType:
        return self._type
    
    def set_node_type(self, type : NodeType):
        self._type = type

    def append_input_node(self, node : Node):
        if isinstance(node, OutputNode):
            raise CompilerException("Error: an output node can't be an input node")
        self._inputs.append(node)

    def remove_input_node_by_name(self, name : str):
        index : int = self._get_input_node_index_by_name(name)
        if index == -1:
            raise CompilerException("Error: input not found in output node")
        self._inputs.pop(index)
    
    def remove_input_node_by_index(self, index : int):
        if index == -1:
            raise CompilerException("Error: input not found in output node")
        self._inputs.pop(index)

    def get_input_node_by_name(self, name : str) -> Node:
        index : int = self._get_input_node_index_by_name(name)
        if index == -1:
            raise CompilerException("Error: input not found in output node")
        return self._inputs[index]

    def get_input_node_by_index(self, index : int) -> Node:
        if index == -1:
            raise CompilerException("Error: input not found in output node")
        return self._inputs[index]
    
    def get_input_nodes_name(self) -> list[str]:
        names : list[str] = []
        for input in self._inputs:
            name = input.get_name()
            names.append(name)
    
    def num_input_nodes(self) -> int:
        return len(self._inputs)
    
    def get_input_nodes_list(self) -> list[Node]:
        return self._inputs

    def generate_code(self) -> str:
        return ""

    def _get_input_node_index_by_name(self, name : str) -> int:
        for i in range(len(self._inputs)):
            node : Node = self._inputs[i]
            if node.get_name() == name:
                return i
        return -1
    
    def generate_includes_code_c(self) -> str:
        return ""

    def generate_declaration_code_c(self) -> str:
        return ""

    def generate_code(self) -> str:
        return ""
    
    def infer_output_shape(self) -> list[list[int]]:
        return self._inputs[0].infer_output_shape()