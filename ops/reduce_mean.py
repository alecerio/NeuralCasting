from ops.ncast_op import NCastOp, NCastOutputDict
from config.config import NCastConfig
from typing import List
from common.common import retrieve_input, get_init_data
import onnx
from onnx import helper

class ReduceMean(NCastOp):
    def __init__(self, onnx_unit):
        super().__init__(onnx_unit)
    
    def update_output_dict(self, graph, ops):
        result = retrieve_input(graph, ops, self.onnx_unit.input[0])
        shape: List[int] = result[0]
        dtype : int = result[1]

        axes = get_init_data(graph, self.onnx_unit.input[1])
        if axes is None:
            raise Exception("In ReduceMean, supported only axes an initializer.")

        ndim = len(shape)

        attrs = {a.name: helper.get_attribute_value(a) for a in self.onnx_unit.attribute}
        keepdims = attrs.get("keepdims", 1)
        
        if axes is None:
            axes = list(range(ndim))

        axes = [(a + ndim) % ndim for a in axes]

        if keepdims:
            out_shape = [
                1 if i in axes else shape[i]
                for i in range(ndim)
            ]
        else:
            out_shape = [
                shape[i]
                for i in range(ndim)
                if i not in axes
            ]

        self.out_dict[self.onnx_unit.output[0]] = NCastOutputDict(shape, dtype)

    def emit_includes(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return []

    def emit_activations_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return []

    def emit_attributes_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return []

    def emit_run_ops(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> str:
        return ""

