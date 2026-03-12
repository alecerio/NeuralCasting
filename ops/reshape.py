from ops.ncast_op import NCastOp, NCastOutputDict
from config.config import NCastConfig
from typing import List
from common.common import retrieve_input, get_init_data
import onnx

class Reshape(NCastOp):
    def __init__(self, onnx_unit):
        super().__init__(onnx_unit)
    
    def update_output_dict(self, graph, ops):
        result = retrieve_input(graph, ops, self.onnx_unit.input[0])
        shape: List[int] = result[0]
        dtype: int = result[1]

        new_shape = get_init_data(graph, self.onnx_unit.input[1])
        if new_shape is None:
            raise Exception("In Reshape, supported only shape an initializer.")
        
        self.out_dict[self.onnx_unit.output[0]] = NCastOutputDict(new_shape, dtype)


    def emit_includes(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return []

    def emit_activations_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return []

    def emit_attributes_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return []

    def emit_run_ops(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> str:
        return ""

