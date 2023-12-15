from .graph_node import GraphNode
import onnx

class InputNode(GraphNode):
    def __init__(self, name : str, input : onnx.onnx_ml_pb2.ValueInfoProto):
        super().__init__(name)
        self._input : onnx.onnx_ml_pb2.ValueInfoProto = input
        self._outputs : list[GraphNode] = []
    
    def __str__(self):
        return super().__str__() + "\n" + \
                "input: " + str(self._input)
    
    def set_input(self, input : onnx.onnx_ml_pb2.ValueInfoProto) -> None:
        self._input = input
    
    def get_input(self) -> onnx.onnx_ml_pb2.ValueInfoProto:
        return self._input
    
    def add_output(self, node : GraphNode) -> bool:
        if self._is_node_in_list(node):
            return False
        self._outputs.append(node)
        return True

    def remove_output(self, node_name : str) -> bool:
        index : int = self._get_index_by_name(node_name)
        if index < 0:
            return False
        self._outputs.pop(index)
        return True

    def remove_output(self, index : int) -> bool:
        if not self._is_valid_output_index(index):
            return False
        self._outputs.pop(index)
        return True

    def get_output(self, node_name : str) -> GraphNode:
        index : int = self._get_index_by_name(node_name)
        if index < 0:
            return None
        else:
            return self._outputs[index]

    def get_output(self, index : int) -> GraphNode:
        if not self._is_valid_output_index(index):
            return None
        else:
            return self._outputs[index]

    def get_output_names(self) -> list[str]:
        names : list[str] = []
        for output in self._outputs:
            name : str = output.get_name()
            names.append(name)
        return names

    def n_outputs(self) -> int:
        return len(self._outputs)
    
    def clear_outputs(self) -> None:
        self._outputs.clear()

    def _is_node_in_list(self, node : GraphNode) -> bool:
        name : str = node.get_name()
        names : list[str] = self.get_output_names()
        if name in names:
            return True
        else:
            return False

    def _get_index_by_name(self, name : str) -> int:
        node_names : list[str] = self.get_output_names()
        for i, n in enumerate(node_names):
            if n == name:
                return i
        return -1
    
    def _is_valid_output_index(self, index : int) -> bool:
        if index < 0 or index >= self.n_outputs():
            return False
        else:
            return True
