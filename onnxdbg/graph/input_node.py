from graph_node import GraphNode
import onnx

class InputNode(GraphNode):
    def __init__(self, name : str, input : onnx.onnx_ml_pb2.ValueInfoProto):
        super().__init__(name)
        self._input : onnx.onnx_ml_pb2.ValueInfoProto = input
    
    def __str__(self):
        return super().__str__() + "\n" + \
                "input: " + str(self._input)
    
    def set_input(self, input : onnx.onnx_ml_pb2.ValueInfoProto) -> None:
        self._input = input
    
    def get_input(self) -> onnx.onnx_ml_pb2.ValueInfoProto:
        return self._input
