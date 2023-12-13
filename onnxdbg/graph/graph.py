import onnx
from onnx import helper
from onnxdbg.graph.graph_node import GraphNode
from onnxdbg.graph.op_node import OpNode
from onnxdbg.graph.input_node import InputNode
from onnxdbg.graph.init_node import InitNode
from onnxdbg.graph.output_node import OutputNode

class Graph():
    def __init__(self, nodes : list[GraphNode]):
        self._nodes = nodes
        self._init_references()
    
    def __str__(self):
        result : str = "GRAPH:\n"
        for node in self._nodes:
            result += str(node) + "\n"
        return result

    def add_node(self, node : GraphNode) -> bool:
        if node.get_name() in self.get_nodes_name():
            return False
        self._nodes.append(node)
        self._init_references()
        return True

    def remove_node(self, node_name : str) -> bool:
        if not node_name in self.get_nodes_name():
            return False
        index : int = self._get_index_by_name(node_name)
        self._nodes.pop(index)
        return True

    def remove_node(self, node_index : int) -> bool:
        if not self._is_node_index_in_range(node_index):
            return False
        self._nodes.pop(node_index)
        return True 

    def get_node(self, node_name : str) -> GraphNode:
        index : int = self._get_index_by_name(node_name)
        if not self._is_node_index_in_range(index):
            return None
        return self._nodes[index]

    def get_node(self, node_index : int) -> GraphNode:
        if not self._is_node_index_in_range(node_index):
            return None
        return self._nodes[node_index]

    def n_nodes(self) -> int:
        return len(self._nodes)

    def get_nodes_name(self) -> list[str]:
        names : list[str] = []
        for node in self._nodes:
            names.append(node.get_name())
        return names
    
    def export_onnx_file(self, model_name : str, path : str) -> None:
        init_nodes = []
        input_nodes = []
        op_nodes = []
        output_nodes = []
        for node in self._nodes:
            if isinstance(node, OpNode):
                n = node.get_node()
                op_nodes.append(n)
            elif isinstance(node, InputNode):
                n = node.get_input()
                input_nodes.append(n)
            elif isinstance(node, OutputNode):
                n = node.get_output()
                output_nodes.append(n)
            elif isinstance(node, InitNode):
                n = node.get_initializer()
                init_nodes.append(n)
        graph = helper.make_graph(op_nodes, model_name, input_nodes, output_nodes, init_nodes)
        model = helper.make_model(graph)
        onnx.save(model, path)
    
    def create_subgraph_data(self, output_node_name : str):
        index : int = self._get_index_by_name(output_node_name)
        if index < 0:
            return None
        output_node : GraphNode = self._nodes[index]
        initializers = []
        inputs = []
        outputs = []
        opnodes = []
        self._fill_subgraph_data(output_node, initializers, inputs, outputs, opnodes)
        return [initializers, inputs, opnodes, outputs]
        
    
    def _fill_subgraph_data(self, node : GraphNode, initializers : list, inputs : list, outputs : list, opnodes : list):
        if isinstance(node, OutputNode):
            outputs.append(node.get_output())
        elif isinstance(node, InputNode):
            inputs.append(node.get_input())
        elif isinstance(node, InitNode):
            initializers.append(node.get_initializer())
        elif isinstance(node, OpNode):
            opnodes.append(node.get_node())
        
        if isinstance(node, OpNode):
            n_inputs : int = node.n_inputs()
            for i in range(n_inputs):
                input_node : GraphNode = node.get_input(i)
                self._fill_subgraph_data(input_node, initializers,inputs, outputs, opnodes)

    def _init_references(self) -> None:
        for node in self._nodes:
            if isinstance(node, OpNode):
                self._update_node_references(node)
            elif isinstance(node, InputNode):
                self._update_input_references(node)
            elif isinstance(node, InitNode):
                self._update_init_references(node)
            elif isinstance(node, OutputNode):
                self._update_output_references(node)
    
    def _update_node_references(self, node : OpNode) -> None:
        node.clear_inputs()
        node.clear_outputs()
        inputs = node.get_node().input
        outputs = node.get_node().output
        for other_node in self._nodes:
            if isinstance(other_node, OpNode):
                other_inputs = other_node.get_node().input
                other_outputs = other_node.get_node().output
                for input in inputs:
                    if input in other_outputs:
                        node.add_input(other_node, str(input))
                for output in outputs:
                    if output in other_inputs:
                        node.add_output(other_node, str(output))
            elif isinstance(other_node, InputNode):
                input_name : str = other_node.get_name()
                if input_name in inputs:
                    node.add_input(other_node, input_name)
            elif isinstance(other_node, InitNode):
                init_name : str = other_node.get_name()
                if init_name in inputs:
                    node.add_input(other_node, init_name)
            elif isinstance(other_node, OutputNode):
                output_name : str = other_node.get_name()
                if output_name in outputs:
                    node.add_output(other_node, output_name)
    
    def _update_input_references(self, node : InputNode) -> None:
        node.clear_outputs()
        name : str = node.get_name()
        for other_node in self._nodes:
            if isinstance(other_node, OpNode):
                inputs = other_node.get_node().input
                if name in inputs:
                    node.add_output(other_node)
    
    def _update_init_references(self, node : InitNode) -> None:
        node.clear_outputs()
        name : str = node.get_name()
        for other_node in self._nodes:
            if isinstance(other_node, OpNode):
                inputs = other_node.get_node().input
                if name in inputs:
                    node.add_output(other_node)
    
    def _update_output_references(self, node : OutputNode) -> None:
        node.clear_inputs()
        name : str = node.get_name()
        for other_node in self._nodes:
            if isinstance(other_node, OpNode):
                outputs = other_node.get_node().output
                if name in outputs:
                    node.add_input(other_node)

    def _get_index_by_name(self, name : str) -> int:
        for i, node_name in enumerate(self.get_nodes_name()):
            if name == node_name:
                return i
        return -1
    
    def _is_node_index_in_range(self, index : int) -> bool:
        if index < 0 or index >= len(self._nodes):
            return False
        else:
            return True
    
    