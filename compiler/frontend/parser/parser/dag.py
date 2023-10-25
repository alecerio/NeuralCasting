from compiler.frontend.parser.node.node import Node
from compiler.frontend.parser.node.input_node import InputNode
from compiler.frontend.parser.node.output_node import OutputNode
from compiler.frontend.parser.node.op_node import OpNode

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

    #def check_if_dag(self) -> bool:
    #    visited : dict = {}
    #    stack : list[Node]
#
    #    for node in self._nodes:
    #        if node not in visited:
    #            no_cycles_found = self._depth_first_search(node, visited, stack)
    #            if not no_cycles_found:
    #                return False
    #    return True
    #
    #def _depth_first_search(self, node : Node, visited : dict, stack : list[Node]):
    #    visited[node] = "visiting"
#
    #    for 

        

    def traversal_dag_and_generate_code(self) -> str:
        generated : list[Node] = []
        active : list[Node] = []

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
                        self._turn_to_active_and_generated_code(inputs, active, generated, node)
                        gen_occured = True
                        
        return ""

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
    
    def _turn_to_active_and_generated_code(self, inputs : list[Node], active : list[Node], generated : list[Node], node : Node):
        for input in inputs:
            if input in active:
                print(input)
                print(" ----------------- ")
                for i in range(len(active)):
                    temp : Node = active[i]
                    if temp == input:
                        active.pop(i)
                        break
                generated.append(input)
        active.append(node)

    def _is_name_in_list(self, name : str) -> bool:
        return name in self.get_list_names()

    def _get_node_index_by_name(self, name : str) -> int:
        i : int = 0
        for node in self._nodes:
            if node.get_name() == name:
                return i
        return -1