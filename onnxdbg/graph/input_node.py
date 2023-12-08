from graph_node import GraphNode
import onnx

class InputNode(GraphNode):
    def __init__(self, name : str, input : model.graph.input):
        super().__init__(name)
        self._input = input
    
    def __str__(self):
        return super().__str__() + "\n" + \
                "input: " + str(self._input)
    
    def set_input(self, input : model.graph.input) -> None:
        self._input = input
    
    def get_input(self) -> model.graph.input:
        return self._input
