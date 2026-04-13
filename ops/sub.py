
from ops.ncast_op import NCastOp, NCastOutputDict
from config.config import NCastConfig
from typing import List
from common.common import retrieve_input
import onnx
from common.common import not_implemented_feature_exception

class Sub(NCastOp):
    def __init__(self, onnx_unit):
        super().__init__(onnx_unit)
    
    def update_output_dict(self, graph, ops):
        resultA = retrieve_input(graph, ops, self.onnx_unit.input[0])
        shapeA: List[int] = resultA[0]
        dtypeA: int = resultA[1]

        resultB = retrieve_input(graph, ops, self.onnx_unit.input[1])
        shapeB: List[int] = resultB[0]
        dtypeB: int = resultB[1]

        if len(shapeA) > len(shapeB):
            new_shape = shapeA
        else:
            new_shape = shapeB
        
        self.out_dict[self.onnx_unit.output[0]] = NCastOutputDict(new_shape, dtypeA)

    def emit_includes(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        not_implemented_feature_exception("Sub", "emit includes not supported")

    def emit_activations_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        not_implemented_feature_exception("Sub", "emit activation initialization not supported")

    def emit_attributes_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        not_implemented_feature_exception("Sub", "emit attributes initialization not supported")

    def emit_run_ops(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> str:
        not_implemented_feature_exception("Sub", "emit run ops not supported")

