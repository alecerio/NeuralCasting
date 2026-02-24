from ops.ncast_op import NCastOp, NCastOutputDict
from config.config import NCastConfig
from typing import List
from common.common import set_valid_tensor_identifier, is_tensor_in_output, retrieve_input, set_onnx_data_type_to_string
import onnx

class QuantizeLinear(NCastOp):
    def __init__(self, onnx_unit):
        super().__init__(onnx_unit)
    
    def update_output_dict(self, graph, ops):
        input = self.onnx_unit.input[0]
        result = retrieve_input(graph, ops, input)
        shape: List[int] = result[0]
        dtype: int = onnx.TensorProto.UINT8
        self.out_dict[self.onnx_unit.output[0]] = NCastOutputDict(shape, dtype)

    def emit_includes(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return ["#include <math.h>", "#include <stdint.h>"]

    def emit_activations_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        output = self.onnx_unit.output[0]
        if is_tensor_in_output(self.onnx_unit.output[0], graph):
            return []
        else:
            shape = self.out_dict[self.onnx_unit.output[0]].shape
            dype = self.out_dict[self.onnx_unit.output[0]].data_type
            dtype_str = set_onnx_data_type_to_string(dype)
            size = 1
            for dim in shape:
                size *= dim
            output_id = set_valid_tensor_identifier(output)
            return [f"{dtype_str} {output_id}[{size}];\n"]

    def emit_attributes_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return []

    def emit_run_ops(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> str:
        x = set_valid_tensor_identifier(self.onnx_unit.input[0])
        y = set_valid_tensor_identifier(self.onnx_unit.output[0])
        shape = self.out_dict[self.onnx_unit.output[0]].shape
        size = 1
        for dim in shape:
            size *= dim
        scale_y = set_valid_tensor_identifier(self.onnx_unit.input[1])
        zero_y = set_valid_tensor_identifier(self.onnx_unit.input[2]) if len(self.onnx_unit.input) > 2 else "0"
        return f"NCAST_QUANTIZE_LINEAR({x}, {y}, {size}, {scale_y}[0], {zero_y}[0]);"