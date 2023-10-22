from compiler.frontend.parser.node.op_node import OpNode

class ReLu(OpNode):
    def __init__(self, name : str):
        super().__init__(name)
    
    def generate_code(self) -> str:
        # TO DO
        return ""