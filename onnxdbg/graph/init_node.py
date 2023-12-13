from .graph_node import GraphNode 
import onnx

class InitNode(GraphNode):
    def __init__(self, name : str, initializer : onnx.onnx_ml_pb2.TensorProto):
        super().__init__(name)
        self._initializer : onnx.onnx_ml_pb2.TensorProto = initializer
        self._outputs : list[GraphNode] = []

    def __str__(self):
        return super().__str__() + "\n" + \
                "initializer: " + str(self._initializer)
    
    def set_initializer(self, initializer : onnx.onnx_ml_pb2.TensorProto):
        self._initializer = initializer
    
    def get_initializer(self) -> onnx.onnx_ml_pb2.TensorProto:
        return self._initializer
                         
