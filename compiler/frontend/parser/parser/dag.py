from compiler.frontend.parser.node.node import Node
from compiler.frontend.parser.node.input_node import InputNode
from compiler.frontend.parser.node.output_node import OutputNode
from compiler.frontend.parser.node.op_node import OpNode
from compiler.frontend.parser.node_types.node_type import NodeType
from compiler.frontend.parser.node_types.tensor_type import TensorType
from compiler.frontend.common.common import onnx_tensor_elem_type_to_c_dictionary
from compiler.frontend.common.common import fix_identifier

class DAG:
    def __init__(self, nodes : list[Node]):
        self._nodes : list[Node] = []
        for node in nodes:
            self.append_node(node)

    def __str__(self):
        result : str = "DAG" + "\n"*2
        for node in self._nodes:
            result = result + str(node)+"\n"*2
        return result

    def append_node(self, node : Node):
        name = node.get_name()
        if self._is_name_in_list(name):
            raise Exception("Error: node name is unique in dag")
        self._nodes.append(node)

    def get_node(self, name : str):
        index : int = self._get_node_index_by_name(name)
        if index == -1:
            raise Exception("Error: node not found in dag")
        return self._nodes[index]

    def remove_node(self, name : str):
        index : int = self._get_node_index_by_name(name)
        if index == -1:
            raise Exception("Error: node not found in dag")
        self._nodes.pop(index)

    def get_list_names(self) -> list[str]:
        names : list[str] = []
        for node in self._nodes:
            name = node.get_name()
            names.append(name)
        return names

    def check_if_dag(self) -> bool:
        # TO DO ...
        pass

    def traversal_dag_and_generate_code(self) -> str:
        generated : list[Node] = []
        active : list[Node] = []
        code_generated : str = ""

        list_includes : list[str] = []
        for node in self._nodes:
            if isinstance(node, OpNode):
                inc_code : str = node.generate_includes_code_c()
                inc_lines : list[str] = inc_code.split('\n')
                filtered_inc_lines : list[str] = [line for line in inc_lines if line.startswith('#include')]
                for line in filtered_inc_lines:
                    list_includes.append(line) 
        list_includes = list(set(list_includes))
        code_generated += "// INCLUDE\n\n"
        for inc in list_includes:
            code_generated += inc + "\n"
        code_generated += "\n\n"

        # generate file header
        code_generated += self._gen_header_code()

        # generate declarations
        for node in self._nodes:
            if isinstance(node, OpNode):
                code_generated += node.generate_declaration_code_c()

        # generate function header code
        code_generated += self._gen_function_header_code()

        # set input nodes active
        for node in self._nodes:
            if isinstance(node, InputNode):
                active.append(node)
        
        gen_occured : bool = True
        while gen_occured:
            gen_occured = False
            for node in self._nodes:
                if node not in generated and node not in active:
                    # get the list of inputs for the current node
                    inputs : list[Node] = self._get_input_nodes_from_opnode_or_output_node(node)
                    
                    # check if node is ready to turn active
                    ready : bool = self._check_if_ready_to_turn_active(inputs, active, generated)
                    
                    # if the node is ready, turn it to active and generate the code
                    if ready:
                        code : str = self._turn_to_active_and_generated_code(inputs, active, generated, node)
                        code_generated = code_generated + code
                        gen_occured = True

        code_generated += "}" 
        return code_generated

    def _get_input_nodes_from_opnode_or_output_node(self, node : Node) -> list[Node]:
        if isinstance(node, OpNode):
            inputs : list[Node] = node.get_input_nodes_list()
        elif isinstance(node, OutputNode):
            inputs : list[Node] = node.get_input_nodes_list()
        return inputs

    def _check_if_ready_to_turn_active(self, inputs : list[Node], active : list[Node], generated : list[Node]) -> bool:
        ready : bool = True
        for input in inputs:
            if input not in active and input not in generated:
                ready = False
                break
        return ready
    
    def _turn_to_active_and_generated_code(self, inputs : list[Node], active : list[Node], generated : list[Node], node : Node) -> str:
        code_generated : str = ""

        for input in inputs:
            if input in active:
                # generate input node code
                code : str = input.generate_code()
                code_generated = code_generated + code + "\n"

                # remove input node from active
                for i in range(len(active)):
                    temp : Node = active[i]
                    if temp == input:
                        active.pop(i)
                        break
                # add input node in generated
                generated.append(input)
        active.append(node)
        return code_generated

    def _gen_header_code(self) -> str:
        import datetime
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        message : str = "// *****************************************************************************\n"
        message += "// \tTHIS CODE WAS AUTOMATICALLY GENERATED ON " + formatted_datetime + "\n"
        message  += "// *****************************************************************************\n\n"
        return message

    def _gen_function_header_code(self) -> str:
        header_code : str = "void run_inference("
        params_list : list[str] = []
        
        for node in self._nodes:
            if isinstance(node, InputNode):
                node_type : NodeType = node.get_node_type()
                if isinstance(node_type, TensorType):
                    name : str = "tensor_" + fix_identifier(node.get_name())
                    param :str = onnx_tensor_elem_type_to_c_dictionary(node_type.get_elem_type()) + " " + name
                    params_list.append(param)
                else:
                    raise Exception("Error: input tensor not supported")
        
        for node in self._nodes:
            if isinstance(node, OutputNode):
                node_type : NodeType = node.get_node_type()
                if isinstance(node_type, TensorType):
                    name : str = "tensor_" + fix_identifier(node.get_name())
                    param : str = onnx_tensor_elem_type_to_c_dictionary(node_type.get_elem_type()) + " " + name
                    params_list.append(param)
                else:
                    raise Exception("Error: input tensor not supported")

        n_params : int = len(params_list)

        for i in range(n_params):
            header_code += params_list[i]
            if i < n_params-1: header_code += ", "
        header_code += ") {\n"
        
        return header_code

    def _is_name_in_list(self, name : str) -> bool:
        return name in self.get_list_names()

    def _get_node_index_by_name(self, name : str) -> int:
        i : int = 0
        for node in self._nodes:
            if node.get_name() == name:
                return i
        return -1