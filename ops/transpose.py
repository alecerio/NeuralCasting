
from ops.ncast_op import NCastOp, NCastOutputDict
from config.config import NCastConfig
from typing import List
from common.common import retrieve_input
import onnx
from common.common import not_implemented_feature_exception

class Transpose(NCastOp):
    def __init__(self, onnx_unit):
        super().__init__(onnx_unit)
    
    def update_output_dict(self, graph, ops):
        result = retrieve_input(graph, ops, self.onnx_unit.input[0])
        shape: List[int] = result[0]
        dtype: int = result[1]

        perm = None
        for attr in self.onnx_unit.attribute:
            if attr.name == "perm":
                perm = onnx.helper.get_attribute_value(attr)
        
        new_shape = []
        for p in perm:
            new_shape.append(shape[p])
        
        self.out_dict[self.onnx_unit.output[0]] = NCastOutputDict(new_shape, dtype)

    def emit_includes(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        not_implemented_feature_exception("Gather", "emit includes not supported")

    def emit_activations_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        not_implemented_feature_exception("Gather", "emit activation initialization not supported")

    def emit_attributes_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        not_implemented_feature_exception("Gather", "emit attributes initialization not supported")

    def emit_run_ops(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> str:
        not_implemented_feature_exception("Gather", "emit run ops not supported")

