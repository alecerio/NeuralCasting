from ops.ncast_op import NCastOp, NCastOutputDict
from config.config import NCastConfig
from typing import List
from common.common import set_valid_tensor_identifier
from typing import Dict
import onnx
from common.common import is_tensor_in_output, retrieve_input

class Abs(NCastOp):
    def __init__(self, onnx_unit):
        super().__init__(onnx_unit)

    def update_output_dict(self, graph, ops: List[NCastOp]):
        input = self.onnx_unit.input[0]
        result = retrieve_input(graph, ops, input)
        
        shape: List[int] = result[0]
        dtype: int = result[1]
        self.out_dict[self.onnx_unit.output[0]] = NCastOutputDict(shape, dtype)

    def emit_includes(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return ["#include <stddef.h>"]

    def emit_activations_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        output_name = set_valid_tensor_identifier(self.onnx_unit.output[0])
        if is_tensor_in_output(self.onnx_unit.output[0], graph):
            return []
        else:
            size = self._get_output_size(graph)
        return [f"float {output_name}[{size}];\n"]

    def emit_attributes_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return []

    def emit_run_ops(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> str:
        input_name = set_valid_tensor_identifier(self.onnx_unit.input[0])
        output_name = set_valid_tensor_identifier(self.onnx_unit.output[0])
        size = self._get_output_size(graph)
        return f"NCAST_ABS({input_name}, {output_name}, {size});"
    
    def _get_output_size(self, graph: onnx.onnx_ml_pb2.GraphProto) -> int:
        for tensor in graph.value_info:
            if tensor.name == self.onnx_unit.output[0]:
                size = 1
                for dim in tensor.type.tensor_type.shape.dim:
                    size *= dim.dim_value
                return size
        return 0

    

