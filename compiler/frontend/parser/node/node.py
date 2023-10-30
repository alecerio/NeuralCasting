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
    
    def _expand_pattern(self, code : str, pattern : str, expanded : str):
        return code.replace(pattern, expanded)
    
    def _read_template_c(self, file_name : str) -> str:
        import os
        curr_dir : str = os.path.dirname(__file__)
        template_file_dir : str = curr_dir + '/../../code_generation/c/'
        template_file_path : str = template_file_dir + file_name
        f = open(template_file_path)
        code : str = f.read()
        f.close()
        return code

    @abc.abstractmethod
    def generate_code(self) -> str:
        pass