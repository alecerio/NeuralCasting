from compiler.frontend.parser.node.node import Node
from compiler.frontend.parser.node_types.node_type import NodeType

class InputNode(Node):
    def __init__(self, name : str, type : NodeType):
        super().__init__(name=name)
        self._type = type
        self._outputs : list[Node] = []
    
    def __str__(self):
        super_str : str = super().__str__()
        input_node_type : str = "input node type:\n" + str(self._type)
        outputs_name : str = "output nodes: "
        for output in self._outputs:
            outputs_name = outputs_name + output.get_name() + ", " 
        return  super_str + "\n" + \
                input_node_type + "\n" + \
                outputs_name
                

    def get_node_type(self) ->  NodeType:
        return self._type
    
    def set_node_type(self, type : NodeType):
        self._type = type
    
    def append_output_node(self, node : Node):
        if isinstance(node, InputNode):
            raise Exception("Error: an input node can't be an output node")
        self._outputs.append(node)
    
    def remove_output_node_by_name(self, name : str):
        index : int = self._get_output_node_index_by_name(name)
        if index == -1:
            raise Exception("Error: output not found in input node")
        self._outputs.pop(index)

    def remove_output_node_by_index(self, index : int):
        if index == -1:
            raise Exception("Error: output not found in input node")
        self._outputs.pop(index)

    def get_output_node_by_name(self, name : str) -> Node:
        index : int = self._get_output_node_index_by_name(name)
        if index == -1:
            raise Exception("Error: output not found in input node")
        return self._outputs[index]

    def get_output_node_by_index(self, index : int) -> Node:
        if index == -1:
            raise Exception("Error: output not found in input node")
        return self._outputs[index]

    def get_output_nodes_name(self) -> list[str]:
        names : list[str] = []
        for output in self._outputs:
            name = output.get_name()
            names.append(name)

    def num_output_nodes(self) -> int:
        return len(self._outputs)

    def _get_output_node_index_by_name(self, name : str) -> int:
        for i in range(len(self._outputs)):
            node : Node = self._outputs[i]
            if node.get_name() == name:
                return i
        return -1
    

            
        