
from ops.ncast_op import NCastOp
from config.config import NCastConfig
from typing import List
from common.common import set_valid_tensor_identifier

class PRelu(NCastOp):
    def __init__(self, onnx_unit):
        super().__init__(onnx_unit)

    def emit_includes(self, config: NCastConfig) -> List[str]:
        return []
    
    def emit_weights_initialization(self, config: NCastConfig) -> List[str]:
        return []

    def emit_activations_initialization(self, config: NCastConfig) -> List[str]:
        return []

    def emit_attributes_initialization(self, config: NCastConfig) -> List[str]:
        return []

    def emit_run_ops(self, config: NCastConfig) -> str:
        return ""


