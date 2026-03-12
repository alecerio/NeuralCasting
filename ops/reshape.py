from ops.ncast_op import NCastOp, NCastOutputDict
from config.config import NCastConfig
from typing import List
from common.common import retrieve_input, is_tensor_in_output, set_valid_tensor_identifier
import onnx

class Reshape(NCastOp):
    def __init__(self, onnx_unit):
        super().__init__(onnx_unit)
    
    def update_output_dict(self, graph, ops):
        result = retrieve_input(graph, ops, self.onnx_unit.input[0])
        shape: List[int] = result[0]
        #dtype: int = result[1]
        #self.out_dict[self.onnx_unit.output[0]] = NCastOutputDict(shape, dtype)

    def emit_includes(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return []

    def emit_activations_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return []
        #output = self.onnx_unit.output[0]
        #if is_tensor_in_output(output, graph):
        #    return []
        #else:
        #    shape = self.out_dict[output].shape
        #    size = 1
        #    for dim in shape:
        #        size *= dim
        #    output_id = set_valid_tensor_identifier(output)
        #    return [f"float {output_id}[{size}];\n"]

    def emit_attributes_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return []

    def emit_run_ops(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> str:
        return ""
        #x = set_valid_tensor_identifier(self.onnx_unit.input[0])
        #y = set_valid_tensor_identifier(self.onnx_unit.output[0])
        #shape = self.out_dict[self.onnx_unit.output[0]].shape
        #size = 1
        #for dim in shape:
        #    size *= dim
        #slope = set_valid_tensor_identifier(self.onnx_unit.input[1])
        #return f"NCAST_PRELU({x}, {y}, {size}, {slope}[0]);"

