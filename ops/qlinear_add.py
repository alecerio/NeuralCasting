from ops.ncast_op import NCastOp, NCastOutputDict
from config.config import NCastConfig
from typing import List
from common.common import retrieve_input, is_tensor_in_output, set_valid_tensor_identifier, set_onnx_data_type_to_string
import onnx

class QLinearAdd(NCastOp):
    def __init__(self, onnx_unit):
        super().__init__(onnx_unit)
    
    def update_output_dict(self, graph, ops):
        result = retrieve_input(graph, ops, self.onnx_unit.input[0])
        shape: List[int] = result[0]
        dtype: int = result[1]
        self.out_dict[self.onnx_unit.output[0]] = NCastOutputDict(shape, dtype)

    def emit_includes(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return ["#include <math.h>", "#include <stdint.h>"]

    def emit_activations_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        output = self.onnx_unit.output[0]
        if is_tensor_in_output(output, graph):
            return []
        else:
            shape = self.out_dict[output].shape
            dtype = self.out_dict[output].data_type
            dtype_str = set_onnx_data_type_to_string(dtype)
            size = 1
            for dim in shape:
                size *= dim
            output_id = set_valid_tensor_identifier(output)
            return [f"{dtype_str} {output_id}[{size}];\n"]

    def emit_attributes_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return []

    def emit_run_ops(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> str:
        a = set_valid_tensor_identifier(self.onnx_unit.input[0])
        b = set_valid_tensor_identifier(self.onnx_unit.input[3])
        c = set_valid_tensor_identifier(self.onnx_unit.output[0])
        shape = self.out_dict[self.onnx_unit.output[0]].shape
        size = 1
        for dim in shape:
            size *= dim
        scale_a = set_valid_tensor_identifier(self.onnx_unit.input[1])
        zero_a = set_valid_tensor_identifier(self.onnx_unit.input[2])
        scale_b = set_valid_tensor_identifier(self.onnx_unit.input[4])
        zero_b = set_valid_tensor_identifier(self.onnx_unit.input[5])
        scale_c = set_valid_tensor_identifier(self.onnx_unit.input[6])
        zero_c = set_valid_tensor_identifier(self.onnx_unit.input[7])
        return f"NCAST_QLINEAR_ADD({a}, {b}, {c}, {size}, {scale_a}[0], {zero_a}[0], {scale_b}[0], {zero_b}[0], {scale_c}[0], {zero_c}[0]);"
