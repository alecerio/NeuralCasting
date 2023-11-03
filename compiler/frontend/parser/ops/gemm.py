"""
Author: Alessandro Cerioli
Description: 
    The Gemm class represents the gemm (General Matrix Multiplication) operation.
    Useful for dense and fully connected layers.
"""

import numpy as np
from compiler.frontend.parser.node.op_node import OpNode
from compiler.frontend.parser.node.output_node import OutputNode
from compiler.frontend.common.common import is_valid_onnx_data_type
from compiler.frontend.common.common import onnx_type_to_c_dictionary

class Gemm(OpNode):
    def __init__(self, name : str, n : int = 1, m : int = 1, weight_data_type = 1, bias_data_type = 1):
        super().__init__(name)
        self._weights : np.ndarray = np.zeros((n, m), dtype=float)
        self._bias : np.ndarray = np.zeros((n,), dtype=float)
        self._weight_data_type : int = weight_data_type
        self._bias_data_type : int = bias_data_type
    
    def __str__(self):
        return super().__str__() + "\n" + \
                "weights: " + str(self._weights) + "\n" + \
                "bias: " + str(self._bias)

    def set_weight_data_type(self, weight_data_type) -> None:
        if is_valid_onnx_data_type(weight_data_type):
            self._weight_data_type = weight_data_type
        else:
            raise Exception("Error: invalid onnx data type")

    def get_weight_data_type(self) -> int:
        return self._weight_data_type
    
    def set_bias_data_type(self, bias_data_type) -> None:
        if is_valid_onnx_data_type(bias_data_type):
            self._bias_data_type = bias_data_type
        else:
            raise Exception("Error: invalid onnx data type")
    
    def get_bias_data_type(self) -> int:
        return self._bias_data_type

    def set_weights_and_bias(self, weights : np.ndarray, bias : np.ndarray):
        w_shape = weights.shape
        b_shape = bias.shape

        if len(w_shape) != 2: 
            raise Exception("Error: weights must be a 2D matrix")
        if len(b_shape) != 1:
            raise Exception("Error: bias must be a vector")
        if w_shape[0] != b_shape[0]:
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

        # node identifier
        name : str = self._name.replace("/", "").replace(":", "")

        # input identifier
        input_name : str = self._input_varnames[0].replace("/", "").replace(":", "")

        # output identifier
        output_name : str = self._output_varnames[0].replace("/", "").replace(":", "")

        # weights size
        [out_size, in_size] = self._weights.shape

        # batch size
        if len(self._bias.shape) > 1: [_, batch_size] = self._bias.shape
        else: batch_size = 1

        # define connected output
        define_connected_output : str = self._gen_define_connected_output()

        # generate code blocks
        output_init_code = self._gen_output_init_code(out_size)
        
        # read template c code
        code : str = self._read_template_c("Gemm.c")

        code = self._expand_pattern(code, "$DEFINE_CONNECTED_OUTPUT", define_connected_output)
        code = self._expand_pattern(code, "$INPUT_SIZE", str(in_size))
        code = self._expand_pattern(code, "$OUTPUT_SIZE", str(out_size))
        code = self._expand_pattern(code, "$BATCH_SIZE", str(batch_size))
        code = self._expand_pattern(code, "$NAME", name)
        code = self._expand_pattern(code, "$INPUT_NAME", input_name)
        code = self._expand_pattern(code, "$OUTPUT_NAME", output_name)
        code = self._expand_pattern(code, "$OUTPUT_INIT", output_init_code)

        return code
    
    def infer_output_shape(self) -> list[list[int]]:
        # batch size
        if len(self._bias.shape) > 1: [_, batch_size] = self._bias.shape
        else: batch_size = 1

        # output size
        [out_size, _] = self._weights.shape

        return [out_size, batch_size]

    def _gen_weights_code(self, out_size : int, in_size : int) -> str:
        weights_code = ""
        
        for i in range(out_size):
            for j in range(in_size):
                weights_code = weights_code + str(self._weights[i][j]) + ", "
        
        return weights_code
    
    def _gen_bias_code(self, out_size : int, batch_size : int) -> str:
        bias_code = ""

        for o in range(out_size):
            bias_code = bias_code + str(self._bias[o]) + ", "
        
        return bias_code
    
    def _gen_output_init_code(self, out_size : int) -> str:
        output_init_code = ""

        for _ in range(out_size):
            output_init_code = output_init_code + "0.0f, "
        
        return output_init_code
    
    def _gen_define_connected_output(self, ) -> str:
        connected_output : bool = isinstance(self._outputs[0], OutputNode)
        
        if connected_output:
            define_connected_output = ""
        else:
            define_connected_output = "#define CONNECTED_OUTPUT"
        
        return define_connected_output
    
    def get_op_type(self) -> str:
        return "Gemm"
    
    def generate_declaration_code_c(self) -> str:
        # node identifier
        name : str = self._name.replace("/", "").replace(":", "")

        # weight type
        w_type : str = onnx_type_to_c_dictionary(self._weight_data_type)

        # bias type
        b_type : str = onnx_type_to_c_dictionary(self._bias_data_type)

        # weights size
        [out_size, in_size] = self._weights.shape

        # weights and bias code
        weights_code = self._gen_weights_code(out_size, in_size)
        bias_code = self._gen_bias_code(out_size, 1)

        # read template c code
        code : str = self._read_template_c("Gemm_decl.c")

        # expand
        code = self._expand_pattern(code, "$NAME", name)
        code = self._expand_pattern(code, "$TYPE_WEIGHTS", w_type)
        code = self._expand_pattern(code, "$TYPE_BIAS", b_type)
        code = self._expand_pattern(code, "$INPUT_SIZE", str(in_size))
        code = self._expand_pattern(code, "$OUTPUT_SIZE", str(out_size))
        code = self._expand_pattern(code, "$WEIGHTS", weights_code)
        code = self._expand_pattern(code, "$BIAS", bias_code)

        return code