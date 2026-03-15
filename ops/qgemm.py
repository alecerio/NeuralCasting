
from ops.ncast_op import NCastOp, NCastOutputDict
from config.config import NCastConfig
from typing import List
from common.common import retrieve_input
import onnx
from common.common import not_implemented_feature_exception

class QGemm(NCastOp):
    def __init__(self, onnx_unit):
        super().__init__(onnx_unit)
    
    def update_output_dict(self, graph, ops):
        resultA = retrieve_input(graph, ops, self.onnx_unit.input[0])
        shapeA: List[int] = resultA[0]
        dtypeA: int = resultA[1]

        resultB = retrieve_input(graph, ops, self.onnx_unit.input[3])
        shapeB: List[int] = resultB[0]
        dtypeB: int = resultB[1]

        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in self.onnx_unit.attribute}
        transA = attrs.get("transA", 0)
        transB = attrs.get("transB", 0)


        dimsA = len(shapeA)
        dimsB = len(shapeB)
        if dimsA > 1 and dimsB > 1:
            idxA = -2 if transA == 0 else -1
            idxB = -1 if transB == 0 else -2
            shapeC = [shapeA[idxA], shapeB[idxB]]
        elif dimsA > 1 and dimsB == 1:
            idxA = -2 if transA == 0 else -1
            shapeC = [shapeA[idxA]]
        elif dimsA == 1 and dimsB > 1:
            idxB = -1 if transB == 0 else -2
            shapeC = [shapeB[idxB]]
        else:
            shapeC = [1]

        self.out_dict[self.onnx_unit.output[0]] = NCastOutputDict(shapeC, dtypeA)

    def emit_includes(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        not_implemented_feature_exception("QGemm", "emit includes not supported")

    def emit_activations_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        not_implemented_feature_exception("QGemm", "emit activation initialization not supported")

    def emit_attributes_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        not_implemented_feature_exception("QGemm", "emit attributes initialization not supported")

    def emit_run_ops(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> str:
        not_implemented_feature_exception("QGemm", "emit run ops not supported")

