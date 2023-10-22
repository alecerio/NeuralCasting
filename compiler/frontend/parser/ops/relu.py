"""
Author: Alessandro Cerioli
Description:
    The class ReLu represents the ReLU (Rectified Linear Unit) activation function.
"""

from compiler.frontend.parser.node.op_node import OpNode

class ReLu(OpNode):
    def __init__(self, name : str):
        super().__init__(name)
    
    def generate_code(self) -> str:
        """
        Method to generate the code related to the ReLu function.

        Return:
            The code related to the ReLu function.
        """
        # TO DO
        return ""