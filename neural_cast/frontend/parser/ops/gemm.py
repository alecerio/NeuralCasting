"""
Author: Alessandro Cerioli
Description: 
    The Gemm class represents the gemm (General Matrix Multiplication) operation.
    Useful for dense and fully connected layers.
"""

from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.parser.node.input_node import InputNode
from neural_cast.frontend.parser.node.init_node import InitializerNode
from neural_cast.frontend.parser.node_types.node_type import NodeType
from neural_cast.frontend.parser.node_types.tensor_type import TensorType
from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.common.common import onnx_type_to_c_dictionary
from neural_cast.frontend.common.common import fix_identifier
from neural_cast.frontend.exceptions.CompilerException import CompilerException
from neural_cast.frontend.parser.ops.common.common import node_type_binary_operation
from neural_cast.frontend.parser.ops.common.common import gen_define_connected_output
from neural_cast.frontend.common.common import CompilerConfig

class Gemm(OpNode):
    def __init__(self, name : str):
        super().__init__(name)
    
    def __str__(self):
        return super().__str__() + "\n"

    def generate_code(self) -> str:
        """
        Method to generate the code related to gemm operation.

        Return:
            The code related to gemm operation.
        """

        parallel : str = CompilerConfig()['parallel']

        # node identifier
        name : str = fix_identifier(self._name)

        # input identifier
        input_name : str = fix_identifier(self._input_varnames[0])
        input_name_w : str = fix_identifier(self._input_varnames[1])
        input_name_b : str = fix_identifier(self._input_varnames[2])

        # output identifier
        output_name : str = fix_identifier(self._output_varnames[0])

        # weights size
        [out_size, in_size] = self.get_weights_shape()

        # define connected output
        define_connected_output : str = gen_define_connected_output(self, 0)

        # output type
        output_type : int = self.infer_output_type()
        output_type_str : str = onnx_type_to_c_dictionary(output_type)

        # nflops
        nflops : int = in_size * out_size * 2 + out_size

        # omp directives
        if parallel == 'omp':
            omp_parallel_for : str = self.gen_omp_parallel_for(input_name_w, input_name, output_name, input_name_b)
            omp_reduction : str = '#pragma omp reduction(temp: +)'
        
        # read template c code
        code : str = self._read_template_c("Gemm.c")

        code = self._expand_pattern(code, "$DEFINE_CONNECTED_OUTPUT", define_connected_output)
        code = self._expand_pattern(code, "$INPUT_SIZE", str(in_size))
        code = self._expand_pattern(code, "$OUTPUT_SIZE", str(out_size))
        code = self._expand_pattern(code, "$NAME", name)
        code = self._expand_pattern(code, "$INPUT_NAME_X", input_name)
        code = self._expand_pattern(code, "$INPUT_NAME_W", input_name_w)
        code = self._expand_pattern(code, "$INPUT_NAME_B", input_name_b)
        code = self._expand_pattern(code, "$OUTPUT_NAME", output_name)
        code = self._expand_pattern(code, "$OUTPUT_TYPE", output_type_str)
        if parallel == 'omp':
            code = self._expand_pattern(code, "$OMP_PARALLEL_FOR", omp_parallel_for)
            code = self._expand_pattern(code, "$OMP_REDUCTION", omp_reduction)
        else:
            code = self._expand_pattern(code, "$OMP_PARALLEL_FOR", "")
            code = self._expand_pattern(code, "$OMP_REDUCTION", "")
        code = self._expand_pattern(code, "$NFLOPS", str(nflops))

        return code
    
    def infer_output_shape(self) -> list[list[int]]:
        shape = self.get_weights_shape()
        return [1, shape[0]]

    def infer_output_type(self) -> int:
        input1 : Node = self._inputs[0]
        input2 : Node = self._inputs[1]
        return node_type_binary_operation(input1, input2, "Gemm")

    def get_weights_shape(self) -> list[list[int]]:
        input_w : Node = self._inputs[1]
        if isinstance(input_w, InputNode):
            t : NodeType = input_w.get_node_type()
            if isinstance(t, TensorType):
                shape = t.get_shape()
            else:
                raise CompilerException("Error: input node type not supported")
        elif isinstance(input_w, OpNode):
            shape = input_w.infer_output_shape()
        elif isinstance(input_w, InitializerNode):
            shape = input_w.get_tensor().shape
        else:
            raise CompilerException("Error: invalid Gemm input node")

        return shape
    
    def get_op_type(self) -> str:
        return "Gemm"
    
    def generate_declaration_code_c(self) -> str:
        return ""
    
    def generate_includes_code_c(self) -> str:
        return ""
    
    def gen_omp_parallel_for(self, W_name, x_name, y_name, b_name) -> str:
        code : str = '#pragma omp parallel for shared(tensor_' + W_name + ', tensor_' + x_name + ', tensor_' + y_name + ', tensor_' + b_name + ') collapse(1)'
        return code