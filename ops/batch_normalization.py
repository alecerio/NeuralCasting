from ops.ncast_op import NCastOp, NCastOutputDict
from config.config import NCastConfig
from typing import List
from common.common import retrieve_input, set_valid_tensor_identifier, is_tensor_in_output
import onnx

class BatchNormalization(NCastOp):
    def __init__(self, onnx_unit):
        super().__init__(onnx_unit)

    def update_output_dict(self, graph, ops):
        result = retrieve_input(graph, ops, self.onnx_unit.input[0])
        shape: List[int] = result[0]
        dtype: int = result[1]
        self.out_dict[self.onnx_unit.output[0]] = NCastOutputDict(shape, dtype)

    def emit_includes(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return ["#include <math.h>"]

    def emit_activations_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        output_name = set_valid_tensor_identifier(self.onnx_unit.output[0])
        if is_tensor_in_output(self.onnx_unit.output[0], graph):
            return []
        else:
            for op in ops:
                if op.onnx_unit.name == self.onnx_unit.name:
                    op_output = op.out_dict[self.onnx_unit.output[0]]
                    out_shape = op_output.shape
                    size = 1
                    for dim in out_shape:
                        size *= dim
                    break
        return [f"float {output_name}[{size}];\n"]

    def emit_attributes_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return []

    def emit_run_ops(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> str:
        x = set_valid_tensor_identifier(self.onnx_unit.input[0])
        y = set_valid_tensor_identifier(self.onnx_unit.output[0])
        shape = self.out_dict[self.onnx_unit.output[0]].shape
        size = 1
        for dim in shape:
            size *= dim
        cout = shape[1]
        scale = set_valid_tensor_identifier(self.onnx_unit.input[1])
        b = set_valid_tensor_identifier(self.onnx_unit.input[2])
        mean = set_valid_tensor_identifier(self.onnx_unit.input[3])
        var = set_valid_tensor_identifier(self.onnx_unit.input[4])
        return f"NCAST_BATCH_NORM({x}, {y}, {size}, {cout}, {scale}, {b}, {mean}, {var}, 1e-5);"

