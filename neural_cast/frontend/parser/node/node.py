import abc
from neural_cast.frontend.common.common import CompilerConfig

class Node(abc.ABC):
    def __init__(self, name : str):
        self._name : str = name
    
    def __str__(self):
        return "name: " + self._name

    def get_name(self) -> str:
        return self._name
    
    def set_name(self, name : str):
        self._name = name
    
    def _expand_pattern(self, code : str, pattern : str, expanded : str):
        return code.replace(pattern, expanded)
    
    def _read_template_c(self, file_name : str) -> str:
        template_file_path : str = CompilerConfig()['codegen_c_path'] + file_name
        f = open(template_file_path)
        code : str = f.read()
        f.close()
        return code

    @abc.abstractmethod
    def generate_code(self) -> str:
        pass

    @abc.abstractmethod
    def generate_includes_code_c(self) -> str:
        pass

    @abc.abstractmethod
    def generate_declaration_code_c(self) -> str:
        pass

    @abc.abstractmethod
    def generate_code(self) -> str:
        pass

    @abc.abstractmethod
    def infer_output_shape(self) -> list[list[int]]:
        pass