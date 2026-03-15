from ops.ncast_op import NCastOp, NCastOutputDict
from config.config import NCastConfig
from typing import List
from common.common import retrieve_input, set_valid_tensor_identifier, set_onnx_data_type_to_string, get_init_data
import onnx
from onnx import numpy_helper

class Unsqueeze(NCastOp):
    def __init__(self, onnx_unit):
        super().__init__(onnx_unit)
    
    def update_output_dict(self, graph, ops):
        result = retrieve_input(graph, ops, self.onnx_unit.input[0])
        shape: List[int] = result[0]
        dtype: int = result[1]

        # find axes
        axes = get_init_data(graph, self.onnx_unit.input[1])
        for a in axes:
            shape.insert(axes[a], 1)
        self.out_dict[self.onnx_unit.output[0]] = NCastOutputDict(shape, dtype)


    def emit_includes(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return []

    def emit_activations_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        output_id = set_valid_tensor_identifier(self.onnx_unit.output[0])
        dtype = self.out_dict[self.onnx_unit.output[0]].data_type
        dtype_str = set_onnx_data_type_to_string(dtype)
        return [f"{dtype_str}* {output_id};"]

    def emit_attributes_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return []

    def emit_run_ops(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> str:
        output_id = set_valid_tensor_identifier(self.onnx_unit.output[0])
        input_id = set_valid_tensor_identifier(self.onnx_unit.input[0])
        return f"{output_id} = {input_id};"
