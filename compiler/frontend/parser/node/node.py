import abc

class Node(abc.ABC):
    def __init__(self, name : str):
        self._name : str = name
    
    def __str__(self):
        return "name: " + self._name

    def get_name(self) -> str:
        return self._name
    
    def set_name(self, name : str):
        self._name = name
    
    @abc.abstractmethod
    def generate_code(self) -> str:
        pass