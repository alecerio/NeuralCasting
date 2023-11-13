from compiler.frontend.parser.node.node import Node
from compiler.frontend.parser.node.input_node import InputNode
from compiler.frontend.exceptions.CompilerException import CompilerException
import math

class InitializerNode(Node):
    def __init__(self, name : str, tensor, data_type):
        super().__init__(name=name)
        self._tensor = tensor
        self._data_type = data_type
        self._outputs : list[Node] = []
    
    def __str__(self):
        super_str : str = super().__str__()
        tensor_str = "tensor:\n" + str(self._tensor)
        data_type_str = "data type: " + str(self._data_type)
        outputs_name : str = "output nodes: "
        for output in self._outputs:
            outputs_name = outputs_name + output.get_name() + ", " 
        return  super_str + "\n" + \
                tensor_str + "\n" + \
                data_type_str + "\n" + \
                outputs_name
    
    def set_tensor(self, tensor):
        self._tensor = tensor

    def get_tensor(self):
        return self._tensor
    
    def set_data_type(self, data_type):
        self._data_type = data_type

    def get_data_type(self):
        return self._data_type

    def append_output_node(self, node : Node):
        if isinstance(node, InputNode):
            raise CompilerException("Error: an input node can't be an output node")
        self._outputs.append(node)
    
    def remove_output_node_by_name(self, name : str):
        index : int = self._get_output_node_index_by_name(name)
        if index == -1:
            raise CompilerException("Error: output not found in input node")
        self._outputs.pop(index)

    def remove_output_node_by_index(self, index : int):
        if index == -1:
            raise CompilerException("Error: output not found in input node")
        self._outputs.pop(index)

    def get_output_node_by_name(self, name : str) -> Node:
        index : int = self._get_output_node_index_by_name(name)
        if index == -1:
            raise CompilerException("Error: output not found in input node")
        return self._outputs[index]

    def get_output_node_by_index(self, index : int) -> Node:
        if index == -1:
            raise CompilerException("Error: output not found in input node")
        return self._outputs[index]

    def get_output_nodes_name(self) -> list[str]:
        names : list[str] = []
        for output in self._outputs:
            name = output.get_name()
            names.append(name)

    def num_output_nodes(self) -> int:
        return len(self._outputs)

    def get_output_nodes_list(self) -> list[Node]:
        return self._outputs

    def _get_output_node_index_by_name(self, name : str) -> int:
        for i in range(len(self._outputs)):
            node : Node = self._outputs[i]
            if node.get_name() == name:
                return i
        return -1
    
    def generate_code(self) -> str:
        return ""

    def generate_includes_code_c(self) -> str:
        return ""

    def generate_declaration_code_c(self) -> str:
        tensor_flat = self._tensor.flatten()
        size : int = len(tensor_flat)
        code : str = "float32 tensor_" + self.get_name() + "[" + str(size) + "] = {"
        for i in range(size):
            code += str(tensor_flat[i]) + ", "
        code += "};"