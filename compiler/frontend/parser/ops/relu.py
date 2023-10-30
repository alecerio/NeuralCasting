"""
Author: Alessandro Cerioli
Description:
    The class ReLu represents the ReLU (Rectified Linear Unit) activation function.
"""

from compiler.frontend.parser.node.op_node import OpNode

class ReLu(OpNode):
    def __init__(self, name : str):
        super().__init__(name)
    
    def __str__(self):
        return super().__str__()

    def generate_code(self) -> str:
        """
        Method to generate the code related to the ReLu function.

        Return:
            The code related to the ReLu function.
        """
        
        input_name : str = self._input_varnames[0].replace("/", "")
        output_name : str = self._output_varnames[0].replace("/", "")

        code : str = self._read_template_c("ReLu.c")

        code = self._expand_pattern(code, "$INPUT_NAME", input_name)
        code = self._expand_pattern(code, "$OUTPUT_NAME", output_name)

        return code
    
    def get_op_type(self) -> str:
        return "ReLu"