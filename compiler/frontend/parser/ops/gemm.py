import numpy as np
from compiler.frontend.parser.node.op_node import OpNode

class Gemm(OpNode):
    def __init__(self, name : str, n : int, m : int):
        super().__init__(name)
        self._weights : np.ndarray = np.zeros((n, m), dtype=float)
        self._bias : np.ndarray = np.zeros((n,), dtype=float)
    
    def set_weights(self, weights : np.ndarray):
        n : int = weights.shape[0]
        m : int = weights.shape[1]

        if n != self._weights.shape[0] or m != self._weights.shape[1]:
            raise Exception("Error: invalid new weights shape")
        
        self._weights = weights
    
    def set_weight(self, weight : float, i : int, j : int):
        if i < 0 or i >= self._weights.shape[0] or j < 0 or j >= self._weights.shape[1]:
            raise Exception("Error: invalid weight indices")
        self._weights[i][j] = weight
    
    def get_weights(self) -> np.ndarray:
        return self._weights
    
    def get_weight(self, i : int, j : int) -> float:
        if i < 0 or i >= self._weights.shape[0] or j < 0 or j >= self._weights.shape[1]:
            raise Exception("Error: invalid weight indices")
        return self._weights[i][j]

    def set_bias(self, bias: np.ndarray):
        n : int = bias.shape[0]
        
        if n != self._bias.shape[0]:
            raise Exception("Error: invalid new bias shape")
        
        self._bias = bias
    
    def set_bias_elem(self, bias_elem : float, i : int):
        if i < 0 or i >= self._bias.shape[0]:
            raise Exception("Error: invalid bias index")
        self._bias[i] = bias_elem
    
    def get_bias(self):
        return self._bias

    def get_bias_elem(self, i : int):
        if i < 0 or i >= self._bias.shape[0]:
            raise Exception("Error: invalid bias index")
        return self._bias[i]
    
    def generate_code(self) -> str:
        # TO DO
        return ""