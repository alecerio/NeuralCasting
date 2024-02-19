from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.parser.node.input_node import InputNode
from neural_cast.frontend.parser.node.init_node import InitializerNode
from neural_cast.frontend.parser.node.output_node import OutputNode
from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.common.common import CompilerLogger
from neural_cast.frontend.exceptions.CompilerException import CompilerException
from neural_cast.frontend.common.common import CompilerConfig
from neural_cast.frontend.parser.parser.codegen_c import pre_codegen_c, post_codegen_c

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
            raise CompilerException("Error: node name is unique in dag")
        self._nodes.append(node)

    def get_node(self, name : str):
        index : int = self._get_node_index_by_name(name)
        if index == -1:
            raise CompilerException("Error: node not found in dag")
        return self._nodes[index]

    def remove_node(self, name : str):
        index : int = self._get_node_index_by_name(name)
        if index == -1:
            raise CompilerException("Error: node not found in dag")
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

    def traversal_dag_and_generate_code(self) -> [list[str], list[str]]:
        CompilerLogger().info("Start code generation")

        output_code : str = CompilerConfig()['output_code']

        generated : list[Node] = []
        active : list[Node] = []

        if output_code == 'C':
            [header_file_code_c, source_file_code_c] = pre_codegen_c(self._nodes)
        if output_code == 'mlir_onnx':
            print("generate precode for mlir")

        # set input nodes active
        CompilerLogger().info("Set input nodes ready to generate code")
        for node in self._nodes:
            if isinstance(node, InputNode) or isinstance(node, InitializerNode):
                CompilerLogger().info("Set ready to generate code: " + node.get_name())
                active.append(node)
            elif isinstance(node, OpNode):
                if node.num_inputs() == 0:
                    active.append(node)
        
        gen_occured : bool = True
        while gen_occured:
            gen_occured = False
            for node in self._nodes:
                if node not in generated and node not in active:
                    # get the list of inputs for the current node
                    inputs : list[Node] = self._get_input_nodes_from_opnode_or_output_node(node)
                    
                    # check if node is ready to turn active
                    CompilerLogger().info("Check if node " + node.get_name() + " is ready to turn active")
                    ready : bool = self._check_if_ready_to_turn_active(inputs, active, generated)
                    
                    # if the node is ready, turn it to active and generate the code
                    if ready:
                        CompilerLogger().info("Generate code for " + node.get_name())
                        code : str = self._turn_to_active_and_generated_code(inputs, active, generated, node, output_code)
                        if  output_code == 'C':
                            source_file_code_c += code
                        elif output_code == 'mlir_onnx':
                            print("concatenate codegen for onnx unit")
                        gen_occured = True

        if output_code == 'C':
            source_file_code_c += post_codegen_c()
        elif output_code == 'mlir_onnx':
            print("codegen for onnx unit")

        if output_code == 'C':
            files_content : list[str] = [header_file_code_c, source_file_code_c]
            files_name : list[str] = [CompilerConfig()['name'] + ".h", CompilerConfig()['name'] + ".c"]
        elif output_code == 'mlir_onnx':
            files_content : list[str] = [""]
            files_name : list[str] = [CompilerConfig()['name'] + ".onnx"]
            print("create files info for onnx")

        return [files_content, files_name]

    def _get_input_nodes_from_opnode_or_output_node(self, node : Node) -> list[Node]:
        print(node.get_name())
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
    
    def _turn_to_active_and_generated_code(self, inputs : list[Node], active : list[Node], generated : list[Node], node : Node, output_code : str) -> str:
        code_generated : str = ""

        for input in inputs:
            if input in active:
                # generate input node code
                if output_code == 'C':
                    code : str = input.generate_code()
                elif output_code == 'mlir_onnx':
                    code : str = ""
                    print("generate code for onnx unit")
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

    def _is_name_in_list(self, name : str) -> bool:
        return name in self.get_list_names()

    def _get_node_index_by_name(self, name : str) -> int:
        i : int = 0
        for node in self._nodes:
            if node.get_name() == name:
                return i
        return -1