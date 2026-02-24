from abc import ABC, abstractmethod
from typing import Dict, List
import onnx
from config.config import NCastConfig

class NCastOutputDict:
    def __init__(self, shape: List[int], data_type: int):
        self.shape: List[int] = shape
        self.data_type: int = data_type

    def __str__(self):
        return f"Shape: {self.shape}, Data Type: {self.data_type}"
    
    def __repr__(self):
        return f"Shape: {self.shape}, Data Type: {self.data_type}"

class NCastOp(ABC):
    def __init__(self, onnx_unit: onnx.onnx_ml_pb2.NodeProto):
        super().__init__()
        self.onnx_unit = onnx_unit
        self.out_dict = {}

    def __str__(self):
        return super().__str__() + f" (ONNX op: {self.onnx_unit.op_type})"

    @abstractmethod
    def update_output_dict(self, graph: onnx.onnx_ml_pb2.GraphProto, ops) -> None:
        pass

    @abstractmethod
    def emit_includes(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops) -> List[str]:
        pass

    @abstractmethod
    def emit_activations_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops) -> List[str]:
        pass

    @abstractmethod
    def emit_attributes_initialization(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops) -> List[str]:
        pass

    @abstractmethod
    def emit_run_ops(self, config: NCastConfig, graph: onnx.onnx_ml_pb2.GraphProto, ops) -> str:
        pass
