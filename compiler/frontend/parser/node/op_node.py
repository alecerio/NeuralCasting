import abc
from compiler.frontend.parser.node.node import Node
from compiler.frontend.parser.node.output_node import OutputNode
from compiler.frontend.parser.node.input_node import InputNode

class OpNode(Node, abc.ABC):
    def __init__(self, name : str):
        super().__init__(name)
        self._inputs : list[Node] = []
        self._outputs : list[Node] = []

    def __str__(self):
        super_str : str = super().__str__()

        op_type : str = "op type: " + self.get_op_type()
        
        inputs : str = "inputs name: "
        for node in self._inputs:
            inputs = inputs + node.name + ", "

        outputs : str = "outputs name: "
        for node in self._outputs:
            outputs = outputs + node.name + ", "

        return super_str + "\n" + \
                op_type + "\n" + \
                inputs + "\n" + \
                outputs + "\n"


    def append_input(self, node : Node):
        if isinstance(node, OutputNode):
            raise Exception("Error: output node can't be the input for an op node")
        self._inputs.append(node)
    
    def remove_input_by_name(self, name : str):
        i : int = self.__get_index_by_name__(self._inputs, name)
        if i == -1:
            raise Exception("Error: input node to remove not found")
        else:
            self._inputs.pop(i)
    
    def remove_input_by_index(self, index : int):
        if index < 0 or index >= len(self._inputs):
            raise Exception("Error: invalid input node index")
        self._inputs.pop(index)
    
    def get_input_by_name(self, name : str) -> Node:
        i : int = self.__get_index_by_name__(self._inputs, name)
        if i == -1:
            raise Exception("Error: input node not found")
        else:
            return self._inputs[i]
        
    def get_input_by_index(self, index : int) -> Node:
        if index < 0 or index >= len(self._inputs):
            raise Exception("Error: invalid input node index")
    
    def append_output(self, node : Node):
        if isinstance(node, InputNode):
            raise Exception("Error: input node can't be the output for an op node")
        self._outputs.append(node)

    def remove_output_by_name(self, name : str):
        i : int = self.__get_index_by_name__(self._outputs, name)
        if i == -1:
            raise Exception("Error: output node to remove not found")
        else:
            self._outputs.pop(i)
    
    def remove_output_by_index(self, index : int):
        if index < 0 or index >= len(self._outputs):
            raise Exception("Error: invalid output node index")
        self._outputs.pop(index)
    
    def get_output_by_name(self, name : str) -> Node:
        i : int = self.__get_index_by_name__(self._outputs, name)
        if i == -1:
            raise Exception("Error: output node not found")
        else:
            return self._outputs[i]
        
    def get_output_by_index(self, index : int) -> Node:
        if index < 0 or index >= len(self._outputs):
            raise Exception("Error: invalid output node index")

    def __get_index_by_name__(self, node_list : list[Node], name : str) -> int:
        i : int = 0
        for node in node_list:
            if node.get_name() == name:
                return i
            i = i+1
        return -1

    @abc.abstractmethod
    def generate_code(self) -> str:
        pass

    @abc.abstractmethod
    def get_op_type(self) -> str:
        pass