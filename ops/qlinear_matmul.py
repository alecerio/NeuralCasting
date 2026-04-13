
from ops.ncast_op import NCastOp, NCastOutputDict
from config.config import NCastConfig
from typing import List
from common.common import retrieve_input
import onnx
from common.common import not_implemented_feature_exception

class QLinearMatmul(NCastOp):
    def __init__(self, onnx_unit):
        super().__init__(onnx_unit)
    
    def update_output_dict(self, graph, ops):
        resultA = retrieve_input(graph, ops, self.onnx_unit.input[0])
        shapeA: List[int] = resultA[0]
        dtypeA: int = resultA[1]

        resultB = retrieve_input(graph, ops, self.onnx_unit.input[3])
        shapeB: List[int] = resultB[0]
        dtypeB: int = resultB[1]

        dimsA = len(shapeA)
        dimsB = len(shapeB)
        if dimsA > 1 and dimsB > 1:
            shapeC = [shapeA[-2], shapeB[-1]]
        elif dimsA > 1 and dimsB == 1:
            shapeC = [shapeA[-2]]
        elif dimsA == 1 and dimsB > 1:
            shapeC = [shapeB[-1]]
        else:
            shapeC = [1]
        
        lenA = len(shapeA)
        lenB = len(shapeB)
        lenC = len(shapeC)

        if lenA >= lenB:
            if lenA > lenC:
                diff = lenA - lenC
                shapeC = shapeA[:diff] + shapeC
        else:
            if lenB > lenC:
                diff = lenB - lenC
                shapeC = shapeB[:diff] + shapeC

        self.out_dict[self.onnx_unit.output[0]] = NCastOutputDict(shapeC, dtypeA)

    def emit_includes(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        not_implemented_feature_exception("QLinearMatmul", "emit includes not supported")

    def emit_activations_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        not_implemented_feature_exception("QLinearMatmul", "emit activation initialization not supported")

    def emit_attributes_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        not_implemented_feature_exception("QLinearMatmul", "emit attributes initialization not supported")

    def emit_run_ops(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> str:
        not_implemented_feature_exception("QLinearMatmul", "emit run ops not supported")

