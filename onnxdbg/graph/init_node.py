from graph_node import GraphNode 
import onnx

class InitNode(GraphNode):
    def __init__(self, initializer : model.graph.initializer):
        self._initializer = initializer

    def __str__(self):
        return super().__str__() + "\n" + \
                "initializer: " + str(self._initializer)
    
    def set_initializer(self, initializer : model.graph.initializer):
        self._initializer = initializer
    
    def get_initializer(self) -> model.graph.initializer:
        return self._initializer
                         
