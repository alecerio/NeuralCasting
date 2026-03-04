import onnx
from typing import List
from ops.abs import Abs
from ops.constant import Constant
from ops.quantize_linear import QuantizeLinear
from ops.qlinear_conv import QLinearConv
from ops.dequantize_linear import DequantizeLinear
from ops.qlinear_add import QLinearAdd
from ops.qlinear_sigmoid import QLinearSigmoid
from ops.unsqueeze import Unsqueeze
from ops.qlinear_mul import QLinearMul
from ops.prelu import PRelu
from ops.batch_normalization import BatchNormalization
from ops.squeeze import Squeeze
from ops.qlinear_matmul import QLinearMatmul
from ops.sub import Sub
from ops.tanh import Tanh
from ops.ncast_op import NCastOp
from config.config import NCastConfig, TEMPLATES_DIR
from common.common import set_valid_tensor_identifier, set_onnx_data_type_to_string

op_dict = {
    "Abs": Abs,
    "BatchNormalization": BatchNormalization,
    "Constant": Constant,
    "DequantizeLinear": DequantizeLinear,
    "QuantizeLinear": QuantizeLinear,
    "QLinearConv": QLinearConv,
    "QLinearAdd": QLinearAdd,
    "QLinearSigmoid": QLinearSigmoid,
    "Unsqueeze": Unsqueeze,
    "QLinearMul": QLinearMul,
    "PRelu": PRelu,
    "Squeeze": Squeeze,
    "QLinearMatMul": QLinearMatmul,
    "Sub": Sub,
    "Tanh": Tanh
}

class NCastGraph:
    def __init__(self, model_path: str):
        model = onnx.load(model_path)
        self.graph = model.graph
        self.ops: List[NCastOp] = []
        self.parse()
        self.update_output_tensors_info()


    def parse(self) -> None:
        for idx, node in enumerate(self.graph.node):
            op_type = node.op_type
            name = node.name if node.name else "<unnamed>"

            if op_type in op_dict.keys():
                OpType = op_dict[op_type]
                op = OpType(node)
                self.ops.append(op)
            else:
                print(f"Warning: Operator '{op_type}' not implemented in NCastGraph. Skipping.")
                #raise NotImplementedError(f"Operator '{op_type}' not implemented in NCastGraph.")

    def update_output_tensors_info(self):
        for op in self.ops:
            op.update_output_dict(self.graph, self.ops)
        
        #for op in self.ops:
        #    print(f"Operator: {op.onnx_unit.op_type}")
        #    for output_name, output_info in op.out_dict.items():
        #        print(f"  Output: {output_name} -> {output_info}")

    def compile(self, config: NCastConfig) -> List[str]:
        template_file_c = self._read_file(f"{TEMPLATES_DIR}/file_c/file_c.c")
        template_file_h = self._read_file(f"{TEMPLATES_DIR}/file_h/file_h.h")
        
        includes : List[str] = []
        activations_initialization : List[str] = []
        attributes_initialization : List[str] = []
        run_ops : List[str] = []

        for op in self.ops:
            self._append_string_list(includes, op.emit_includes(config, self.graph, self.ops))
            self._append_string_list(activations_initialization, op.emit_activations_initialization(config, self.graph, self.ops))
            self._append_string_list(attributes_initialization, op.emit_attributes_initialization(config, self.graph, self.ops))
            self._append_string_list(run_ops, [op.emit_run_ops(config, self.graph, self.ops)])

        template_file_h = self._replace_includes(template_file_h, includes)
        
        template_file_h = self._replace_inputs(template_file_h)
        template_file_h = self._replace_outputs(template_file_h)
        template_file_h = self._replace_model_name(template_file_h)

        template_file_c = self._replace_activations_initialization(template_file_c, activations_initialization)
        template_file_c = self._replace_attributes_initialization(template_file_c, attributes_initialization)
        template_file_c = self._replace_weights_initialization(template_file_c)
        template_file_c = self._replace_run_ops(template_file_c, run_ops)
        template_file_c = self._replace_inputs(template_file_c)
        template_file_c = self._replace_outputs(template_file_c)
        template_file_c = self._replace_model_name(template_file_c)

        return [template_file_c, template_file_h]

    def get_tensor_shape(self, tensor_name: str):
        for op in self.ops:
            out_names = list(op.out_dict.keys())
            if tensor_name in out_names:
                return op.out_dict[tensor_name].shape
        for input in self.graph.input:
            if tensor_name == input:
                shape = [d.dim_value for d in input.type.tensor_type.shape.dim]
                return shape
        for init in self.graph.initializer:
            if tensor_name == init.name:
                return init.dims

        raise Exception(f"tensor {tensor_name} not found")

    def _read_file(self, path: str) -> str:
        file = open(path, "r")
        content = file.read()
        file.close()
        return content
    
    def _replace_includes(self, template: str, includes: List[str]) -> str:
        includes = list(set(includes)) # remove duplicates
        includes_str = "\n".join(includes)
        template = template.replace("$INCLUDES", includes_str)
        return template
    
    def _replace_activations_initialization(self, template: str, activations_initialization: List[str]) -> str:
        activations_initialization_str = "\n".join(activations_initialization)
        template = template.replace("$ACTIVATIONS_INITIALIZATION", activations_initialization_str)
        return template
    
    def _replace_attributes_initialization(self, template: str, attributes_initialization: List[str]) -> str:
        attributes_initialization_str = "\n".join(attributes_initialization)
        template = template.replace("$ATTRIBUTES_INITIALIZATION", attributes_initialization_str)
        return template
    
    def _replace_run_ops(self, template: str, run_ops: List[str]) -> str:
        run_ops_str = "\n\n".join(run_ops)
        template = template.replace("$RUN_OPS", run_ops_str)
        return template
    
    def _replace_inputs(self, template: str) -> str:
        inputs = []
        for input in self.graph.input:
            input_id = set_valid_tensor_identifier(input.name)
            dtype = input.type.tensor_type.elem_type
            dtype_str = set_onnx_data_type_to_string(dtype)
            inputs.append(f"{dtype_str}* {input_id}")
        inputs_str = ", ".join(inputs)
        template = template.replace("$INPUTS", inputs_str)
        return template
    
    def _replace_outputs(self, template: str) -> str:
        outputs = []
        for output in self.graph.output:
            output_id = set_valid_tensor_identifier(output.name)
            dtype = output.type.tensor_type.elem_type
            dtype_str = set_onnx_data_type_to_string(dtype)
            outputs.append(f"{dtype_str}* {output_id}")
        outputs_str = ", ".join(outputs)
        template = template.replace("$OUTPUTS", outputs_str)
        return template
    
    def _replace_model_name(self, template: str) -> str:
        template = template.replace("$MODEL_NAME", self.graph.name)
        return template
    
    def _replace_weights_initialization(self, template: str) -> str:
        weights_initialization: List[str] = []
        for initializer in self.graph.initializer:
            name = set_valid_tensor_identifier(initializer.name)
            dtype = initializer.data_type
            dtype_str = set_onnx_data_type_to_string(dtype)
            values = initializer.float_data if initializer.data_type == 1 else []
            if not values:
                import numpy as np
                tensor_array = onnx.numpy_helper.to_array(initializer)
                values = tensor_array.flatten().tolist()
            size = len(values)
            values_str = ", ".join([str(v) for v in values])
            weights_initialization.append(f"{dtype_str} {name}[{size}] = {{{values_str}}};\n")
        initializers_str = "".join(weights_initialization)
        template = template.replace("$WEIGHTS_INITIALIZATION", initializers_str)
        return template

    def _append_string_list(self, dst: list[str], src: list[str]) -> None:
        dst.extend(src)
