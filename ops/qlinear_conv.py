from ops.ncast_op import NCastOp, NCastOutputDict
from config.config import NCastConfig
from typing import List
from common.common import retrieve_input, is_tensor_in_output, set_valid_tensor_identifier, set_onnx_data_type_to_string
import onnx
from onnx import helper

class QLinearConv(NCastOp):
    def __init__(self, onnx_unit):
        super().__init__(onnx_unit)
    
    def update_output_dict(self, graph, ops):
        result = retrieve_input(graph, ops, self.onnx_unit.input[0])
        shape: List[int] = result[0]
        dtype: int = result[1]

        attrs = {a.name: helper.get_attribute_value(a) for a in self.onnx_unit.attribute}
        strides = attrs.get("strides")
        pads = attrs.get("pads")
        dilations = attrs.get("dilations")
        kernel_shape = attrs.get("kernel_shape")

        weight_name = self.onnx_unit.input[3]
        W = None
        for init in graph.initializer:
            if init.name == weight_name:
                W = init
                break

        Cout = W.dims[0]
        Lout = (shape[2] + pads[0] + pads[1] - dilations[0] * (kernel_shape[0] - 1) - 1) // strides[0] + 1
        self.out_dict[self.onnx_unit.output[0]] = NCastOutputDict([1, Cout, Lout], dtype)

    def emit_includes(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops: List[NCastOp]) -> List[str]:
        return ["#include <stdint.h>", "#include <math.h>"]

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
        #NCAST_QLINEAR_CONV(X, W, B, Y, CIN, COUT, LIN, SCALE_X, ZERO_X, SCALE_W, ZERO_W, SCALE_Y, ZERO_Y, DIL, GROUP, KERNEL, PADS, STRIDES)
        x = set_valid_tensor_identifier(self.onnx_unit.input[0])
        w = set_valid_tensor_identifier(self.onnx_unit.input[3])
        if len(self.onnx_unit.input) > 8:
            b = set_valid_tensor_identifier(self.onnx_unit.input[8])
        else:
            b = "0"
        y = set_valid_tensor_identifier(self.onnx_unit.output[0])

        result_x = retrieve_input(graph, ops, self.onnx_unit.input[0])
        shape_x = result_x[0]
        cin = shape_x[1]
        lin = shape_x[2]

        result_w = retrieve_input(graph, ops, self.onnx_unit.input[3])
        shape_w = result_w[0]
        cout = shape_w[0]

        scale_x = set_valid_tensor_identifier(self.onnx_unit.input[1])
        zero_x = set_valid_tensor_identifier(self.onnx_unit.input[2])
        scale_w = set_valid_tensor_identifier(self.onnx_unit.input[4])
        zero_w = set_valid_tensor_identifier(self.onnx_unit.input[5])
        scale_y = set_valid_tensor_identifier(self.onnx_unit.input[6])
        zero_y = set_valid_tensor_identifier(self.onnx_unit.input[7])

        attrs = {a.name: helper.get_attribute_value(a) for a in self.onnx_unit.attribute}
        dilations = attrs.get("dilations")[0]
        group = attrs.get("group")
        kernel_shape = attrs.get("kernel_shape")[0]
        pads = attrs.get("pads")
        strides = attrs.get("strides")[0]

        return f"NCAST_QLINEAR_CONV({x}, {w}, {b}, {y}, {cin}, {cout}, {lin}, {scale_x}, {zero_x}, {scale_w}, {zero_w}, {scale_y}, {zero_y}, {dilations}, {group}, {kernel_shape}, {{{pads[0]}, {pads[1]}}}, {strides});"
