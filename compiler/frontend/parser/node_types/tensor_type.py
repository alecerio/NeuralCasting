from compiler.frontend.parser.node_types.node_type import NodeType

class TensorType(NodeType):
    def __init__(self, shape: list[int], elem_type : int):
        super().__init__()
        self._shape : list[int] = shape
        self._elem_type : int = elem_type
    
    def get_shape(self) -> list[int]:
        return self._shape
    
    def set_shape(self, shape : list[int]):
        self._shape = shape

    def get_elem_type(self) -> int:
        return self._elem_type
    
    def set_elem_type(self, elem_type : int):
        if elem_type < 0: elem_type = 0
        elif elem_type > 9: elem_type = 9
        self._elem_type = elem_type