"""
Author: Alessandro Cerioli
Description: 
    The Gemm class represents the gemm (General Matrix Multiplication) operation.
    Useful for dense and fully connected layers.
"""

import numpy as np
from compiler.frontend.parser.node.op_node import OpNode

class Gemm(OpNode):
    def __init__(self, name : str, n : int = 1, m : int = 1):
        super().__init__(name)
        self._weights : np.ndarray = np.zeros((n, m), dtype=float)
        self._bias : np.ndarray = np.zeros((n,), dtype=float)
    
    def __str__(self):
        return super().__str__() + "\n" + \
                "weights: " + str(self._weights) + "\n" + \
                "bias: " + str(self._bias)

    def set_weights_and_bias(self, weights : np.ndarray, bias : np.ndarray):
        w_shape = weights.shape
        b_shape = bias.shape

        if len(w_shape) != 2: 
            raise Exception("Error: weights must be a 2D matrix")
        if len(b_shape) != 1:
            raise Exception("Error: bias must be a vector")
        if w_shape[1] != b_shape[0]:
            raise Exception("Error: number of weights columns must be equal to number of bias rows")
        
        self._weights = weights
        self._bias = bias


    def set_weights(self, weights : np.ndarray):
        """
        Method to set all the weights of the gemm.
        The shape must be the same as the current weights matrix.

        Args:
            weights: The new weights of the gemm
        """
        n : int = weights.shape[0]
        m : int = weights.shape[1]

        if n != self._weights.shape[0] or m != self._weights.shape[1]:
            raise Exception("Error: invalid new weights shape")
        
        self._weights = weights
    
    def set_weight(self, weight : float, i : int, j : int):
        """
        Method to set a single weight of the gemm.
        The indices must be in the weight matrix shape.
        
        Arg:
            weight: new weight
            i: row index
            j: col index
        """
        if i < 0 or i >= self._weights.shape[0] or j < 0 or j >= self._weights.shape[1]:
            raise Exception("Error: invalid weight indices")
        self._weights[i][j] = weight
    
    def get_weights(self) -> np.ndarray:
        """
        Method to get all the weights of the gemm.
        
        Return:
            The weights matrix.
        """
        return self._weights
    
    def get_weight(self, i : int, j : int) -> float:
        """
        Method to get a single weight of the weights matrix.

        Arg:
            i: row index
            j: col index
        
        Return:
            The weight in position (i,j).
        """
        if i < 0 or i >= self._weights.shape[0] or j < 0 or j >= self._weights.shape[1]:
            raise Exception("Error: invalid weight indices")
        return self._weights[i][j]

    def set_bias(self, bias: np.ndarray):
        """
        Method to set all the bias vector of the gemm.

        Arg:
            bias: The new bias for the gemm.
        """
        n : int = bias.shape[0]
        m : int = bias.shape[1]
        
        if n != self._bias.shape[0]:
            raise Exception("Error: invalid new bias shape")
        
        if m != 1:
            raise Exception("Error: bias must be a vector and not a matrix")
        
        if len(bias.shape) != 2:
            raise Exception("Error: bias must be in two dimensions [n,1]")

        self._bias = bias
    
    def set_bias_elem(self, bias_elem : float, i : int):
        """
        Method to set a single element of the bias.

        Arg:
            bias_elem: The new bias element.
            i: The position of the element in the bias vector.
        """
        if i < 0 or i >= self._bias.shape[0]:
            raise Exception("Error: invalid bias index")
        self._bias[i] = bias_elem
    
    def get_bias(self) -> np.ndarray:
        """
        Method to get all the bias vector of the gemm.

        Return:
            The bias vector of the gemm.
        """
        return self._bias

    def get_bias_elem(self, i : int) -> float:
        """
        Method to get a single element of the bias vector.

        Arg:
            i: The position of the element in the bias vector.
        
        Return:
            The bias element in position i-th.
        """
        if i < 0 or i >= self._bias.shape[0]:
            raise Exception("Error: invalid bias index")
        return self._bias[i]
    
    def generate_code(self) -> str:
        """
        Method to generate the code related to gemm operation.

        Return:
            The code related to gemm operation.
        """
        # TO DO
        return ""
    
    def get_op_type(self) -> str:
        return "Gemm"