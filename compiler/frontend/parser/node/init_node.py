from compiler.frontend.parser.node.op_node import OpNode
from compiler.frontend.common.common import fix_identifier
import math

class InitializerNode(OpNode):
    def __init__(self, name: str, tensor):
        super().__init__(name)
        self._tensor = tensor
    
    def __str__(self):
        return super().__str__() + "\n" + \
                "tensor: " + str(self._tensor)
    
    def set_tensor(self, tensor):
        self._tensor = tensor

    def get_tensor(self):
        return self._tensor
    
    def generate_code(self) -> str:
        return ""
    
    def generate_declaration_code_c(self) -> str:
        name : str = self.get_name()
        flatten_tensor = self._tensor.flatten()
        size : int = len(flatten_tensor)
        result : str = "float32_t " + fix_identifier(name) + "[" + str(size) + "] = {"
        for i in range(size):
            result += str(flatten_tensor[i]) + ", "
        result += "};"
    
    def generate_includes_code_c(self) -> str:
        return ""
    
    def get_op_type(self) -> str:
        return "InitializerNode"
    
    def infer_output_shape(self) -> list[list[int]]:
        return self._tensor.shape
