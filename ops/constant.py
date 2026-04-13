from ops.ncast_op import NCastOp, NCastOutputDict
from config.config import NCastConfig
from typing import List
from common.common import set_valid_tensor_identifier, set_onnx_data_type_to_string
import onnx

class Constant(NCastOp):
    def __init__(self, onnx_unit):
        super().__init__(onnx_unit)
    
    def update_output_dict(self, graph, ops):
        attr = next(a for a in self.onnx_unit.attribute if a.name == "value")
        tensor = attr.t
        dtype: int = tensor.data_type
        shape: List[int] = list(tensor.dims)
        self.out_dict[self.onnx_unit.output[0]] = NCastOutputDict(shape, dtype)

    def emit_includes(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return []

    def emit_activations_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return []

    def emit_attributes_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        attr = next(a for a in self.onnx_unit.attribute if a.name == "value")
        tensor = attr.t
        dtype = tensor.data_type
        dtype_str = set_onnx_data_type_to_string(dtype)
        shape = list(tensor.dims)
        size = 1
        for dim in shape:
            size *= dim
        value = onnx.numpy_helper.to_array(tensor)
        
        lst = value.tolist()
        lst2 = " ".join(map(str, lst))
        return [f"{dtype_str} {set_valid_tensor_identifier(self.onnx_unit.output[0])}[{size}] = {{ {lst2} }};\n"]

    def emit_run_ops(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> str:
        return ""
