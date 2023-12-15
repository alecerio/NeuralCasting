from .graph_node import GraphNode
import onnx

class OutputNode(GraphNode):
    def __init__(self, name : str, output : onnx.onnx_ml_pb2.ValueInfoProto):
        super().__init__(name)
        self._output = output
        self._inputs : list[GraphNode] = []
    
    def __str__(self):
        return super().__str__() + "\n" + \
                "output: " + str(self._output)
    
    def set_output(self, output : onnx.onnx_ml_pb2.ValueInfoProto) -> None:
        self._output = output
    
    def get_output(self) -> onnx.onnx_ml_pb2.ValueInfoProto:
        return self._output
    
    def add_input(self, node : GraphNode) -> bool:
        if self._is_node_in_list(node):
            return False
        self._inputs.append(node)
        return True

    def remove_input(self, node_name : str) -> bool:
        index : int = self._get_index_by_name(node_name)
        if index < 0:
            return False
        self._inputs.pop(index)
        return True

    def remove_input(self, index : int) -> bool:
        if not self._is_valid_input_index(index):
            return False
        self._inputs.pop(index)
        return True

    def get_input(self, node_name : str) -> GraphNode:
        index : int = self._get_index_by_name(node_name)
        if index < 0:
            return None
        else:
            return self._inputs[index]

    def get_input(self, index : int) -> GraphNode:
        if not self._is_valid_input_index(index):
            return None
        else:
            return self._inputs[index]

    def get_input_names(self) -> list[str]:
        names : list[str] = []
        for input in self._inputs:
            name : str = input.get_name()
            names.append(name)
        return names

    def n_inputs(self) -> int:
        return len(self._inputs)
    
    def clear_inputs(self) -> None:
        self._inputs.clear()

    def _is_node_in_list(self, node : GraphNode) -> bool:
        name : str = node.get_name()
        names : list[str] = self.get_input_names()
        if name in names:
            return True
        else:
            return False

    def _get_index_by_name(self, name : str) -> int:
        node_names : list[str] = self.get_input_names()
        for i, n in enumerate(node_names):
            if n == name:
                return i
        return -1
    
    def _is_valid_input_index(self, index : int) -> bool:
        if index < 0 or index >= self.n_inputs():
            return False
        else:
            return True