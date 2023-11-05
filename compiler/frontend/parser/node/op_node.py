import abc
from compiler.frontend.parser.node.node import Node
from compiler.frontend.parser.node.output_node import OutputNode
from compiler.frontend.parser.node.input_node import InputNode
from compiler.frontend.exceptions.CompilerException import CompilerException

class OpNode(Node, abc.ABC):
    def __init__(self, name : str):
        super().__init__(name)
        self._inputs : list[Node] = []
        self._outputs : list[Node] = []
        self._input_varnames : list[str] = []
        self._output_varnames : list[str] = []

    def __str__(self):
        super_str : str = super().__str__()

        op_type : str = "op type: " + self.get_op_type()
        
        inputs : str = "inputs name: "
        for node in self._inputs:
            inputs = inputs + node.get_name() + ", "

        input_varnames : str = "input values name: "
        for name in self._input_varnames:
            input_varnames = input_varnames + name

        outputs : str = "outputs name: "
        for node in self._outputs:
            outputs = outputs + node.get_name() + ", "

        output_varnames : str = "output values name: "
        for name in self._output_varnames:
            output_varnames = output_varnames + name

        return super_str + "\n" + \
                op_type + "\n" + \
                inputs + "\n" + \
                input_varnames + "\n" + \
                outputs + "\n" + \
                output_varnames + "\n"


    def append_input(self, node : Node, name : str):
        if isinstance(node, OutputNode):
            raise CompilerException("Error: output node can't be the input for an op node")
        self._inputs.append(node)
        self._input_varnames.append(name)
    
    def remove_input_by_name(self, name : str):
        i : int = self._get_index_by_name(self._inputs, name)
        if i == -1:
            raise CompilerException("Error: input node to remove not found")
        else:
            self._inputs.pop(i)
            self._input_varnames.pop(i)
    
    def remove_input_by_index(self, index : int):
        if index < 0 or index >= len(self._inputs):
            raise CompilerException("Error: invalid input node index")
        self._inputs.pop(index)
        self._input_varnames.pop(index)
    
    def get_input_by_name(self, name : str) -> Node:
        i : int = self._get_index_by_name(self._inputs, name)
        if i == -1:
            raise CompilerException("Error: input node not found")
        else:
            return self._inputs[i]
    
    def get_input_varname_by_name(self, name : str) -> Node:
        i : int = self._get_index_by_name(self._inputs, name)
        if i == -1:
            raise CompilerException("Error: input node not found")
        else:
            return self._input_varnames[i]

    def get_input_name_by_index(self, index : int) -> Node:
        if index < 0 or index >= len(self._inputs):
            raise CompilerException("Error: invalid input node index")
        return self._inputs[index].get_name()

    def get_input_varname_by_index(self, index : int) -> Node:
        if index < 0 or index >= len(self._inputs):
            raise CompilerException("Error: invalid input node index")
        return self._input_varnames[index]

    def append_output(self, node : Node, name : str):
        if isinstance(node, InputNode):
            raise CompilerException("Error: input node can't be the output for an op node")
        self._outputs.append(node)
        self._output_varnames.append(name)

    def remove_output_by_name(self, name : str):
        i : int = self._get_index_by_name(self._outputs, name)
        if i == -1:
            raise CompilerException("Error: output node to remove not found")
        else:
            self._outputs.pop(i)
            self._output_varnames.pop(i)
    
    def remove_output_by_index(self, index : int):
        if index < 0 or index >= len(self._outputs):
            raise CompilerException("Error: invalid output node index")
        self._outputs.pop(index)
        self._output_varnames.pop(index)
    
    def get_output_by_name(self, name : str) -> Node:
        i : int = self._get_index_by_name(self._outputs, name)
        if i == -1:
            raise CompilerException("Error: output node not found")
        else:
            return self._outputs[i]
    
    def get_output_varname_by_name(self, name : str) -> Node:
        i : int = self._get_index_by_name(self._outputs, name)
        if i == -1:
            raise CompilerException("Error: output node not found")
        else:
            return self._output_varnames[i]
        
    def get_output_by_index(self, index : int) -> Node:
        if index < 0 or index >= len(self._outputs):
            raise CompilerException("Error: invalid output node index")
        return self._outputs[index]
    
    def get_output_varname_by_index(self, index : int) -> Node:
        if index < 0 or index >= len(self._outputs):
            raise CompilerException("Error: invalid output node index")
        return self._output_varnames[index]

    def get_input_names(self) -> list[str]:
        names : list[str] = []
        for input in self._inputs:
            name : str = input.get_name()
            names.append(name)
        return names

    def get_input_varnames(self) -> list[str]:
        varnames : list[str] = []
        for inputvar in self._input_varnames:
            varname : str = inputvar
            varnames.append(varname)
        return varnames

    def get_output_names(self) -> list[str]:
        names : list[str] = []
        for output in self._outputs:
            name : str = output.get_name()
            names.append(name)
        return names

    def get_output_varnames(self) -> list[str]:
        varnames : list[str] = []
        for outputvar in self._output_varnames:
            varname : str = outputvar
            varnames.append(varname)
        return varnames

    def num_outputs(self) -> int:
        return len(self._outputs)

    def num_inputs(self) -> int:
        return len(self._input)
    
    def get_output_nodes_list(self) -> list[Node]:
        return self._outputs
    
    def get_input_nodes_list(self) -> list[Node]:
        return self._inputs

    def _get_index_by_name(self, node_list : list[Node], name : str) -> int:
        i : int = 0
        for node in node_list:
            if node.get_name() == name:
                return i
            i = i+1
        return -1

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
    def get_op_type(self) -> str:
        pass

    @abc.abstractmethod
    def infer_output_shape(self) -> list[list[int]]:
        pass