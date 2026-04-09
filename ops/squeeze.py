
from ops.ncast_op import NCastOp, NCastOutputDict
from config.config import NCastConfig
from typing import List
from common.common import retrieve_input, get_init_data
import onnx
from common.common import not_implemented_feature_exception
import numpy as np

class Squeeze(NCastOp):
    def __init__(self, onnx_unit):
        super().__init__(onnx_unit)
    
    def update_output_dict(self, graph, ops):
        result = retrieve_input(graph, ops, self.onnx_unit.input[0])
        shape: List[int] = result[0]
        dtype: int = result[1]
        
        if len(self.onnx_unit.input) == 1:
            index = 0
            while shape[index] == 1 and index <  len(shape):
                index = index + 1
            new_shape = shape[index:]
        else:
            axes = get_init_data(graph, self.onnx_unit.input[1])
            if axes != None:
                shape_np = np.array(shape)
                axes_np = np.array(axes)
                mask = np.ones(len(shape_np), dtype=bool)
                mask[axes_np] = False
                new_shape_np = shape_np[mask]
                new_shape = new_shape_np.astype(int).tolist()
            else:
                i = 0
                if shape != []:
                    while shape[i] == 1 and i < len(shape):
                        i = i+1
                    new_shape = shape[i:]
                else:
                    new_shape = []
        self.out_dict[self.onnx_unit.output[0]] = NCastOutputDict(new_shape, dtype)

    def emit_includes(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        not_implemented_feature_exception("Squeeze", "emit includes not supported")

    def emit_activations_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        not_implemented_feature_exception("Squeeze", "emit activation initialization not supported")

    def emit_attributes_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        not_implemented_feature_exception("Squeeze", "emit attributes initialization not supported")

    def emit_run_ops(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> str:
        not_implemented_feature_exception("Squeeze", "emit run ops not supported")

