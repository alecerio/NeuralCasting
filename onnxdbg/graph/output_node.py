from graph_node import GraphNode
import onnx

class OutputNode(GraphNode):
    def __init__(self, name : str, output : onnx.onnx_ml_pb2.ValueInfoProto):
        super().__init__(name)
        self._output = output
    
    def __str__(self):
        return super().__str__() + "\n" + \
                "output: " + str(self._output)
    
    def set_output(self, output : onnx.onnx_ml_pb2.ValueInfoProto) -> None:
        self._output = output
    
    def get_output(self) -> onnx.onnx_ml_pb2.ValueInfoProto:
        return self._output