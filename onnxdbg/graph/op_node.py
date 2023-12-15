from .graph_node import GraphNode
import onnx

class OpNode(GraphNode):
    def __init__(self, name : str, node : onnx.onnx_ml_pb2.NodeProto):
        super().__init__(name)
        self._inputs : list[GraphNode] = []
        self._input_names : list[str] = []
        self._outputs : list[GraphNode] = []
        self._output_names : list[str] = []
        self._attributes : dict = {}
        self._node : onnx.onnx_ml_pb2.NodeProto = node
    
    def __str__(self):
        result : str = ""
        result += super.__str__() + "\n"

        result += "INPUTS:\n"
        N_INPUTS : int = len(self._inputs)
        for n in range(N_INPUTS):
            input : GraphNode = self._inputs[n]
            input_name : str = self._input_names[n]
            result += "input: " + str(input) + "\n"
            result += "input name: " + input_name + "\n\n"
        
        result += "OUTPUTS:\n"
        N_OUTPUTS : int = len(self._outputs)
        for n in range(N_OUTPUTS):
            output : GraphNode = self._outputs[n]
            output_name : str = self._output_names[n]
            result += "output: " + str(output) + "\n"
            result += "output name: " + output_name + "\n\n"
        
        result += "node: " + str(self._node)

        return result
    
    def get_node(self) -> onnx.onnx_ml_pb2.NodeProto:
        return self._node
    
    def n_inputs(self) -> int:
        return len(self._inputs)
    
    def get_input(self, index : int) -> GraphNode:
        if not self._is_correct_input_index(index):
            return None
        else:
            return self._inputs[index]
    
    def get_input_name(self, index : int) -> str:
        if not self._is_correct_input_index(index):
            return None
        else:
            return self._input_names[index]
    
    def add_input(self, input : GraphNode, input_name : str) -> bool:
        if self._is_input_name_in_list(input_name) or self._is_input_node_in_list(input):
            return False
        self._inputs.append(input)
        self._input_names.append(input_name)
        return True
    
    def remove_input(self, index : int) -> bool:
        if not self._is_correct_input_index(index):
            return False
        self._inputs.pop(index)
        self._input_names.pop(index)
    
    def clear_inputs(self) -> None:
        self._inputs.clear()
        
    def n_outputs(self) -> int:
        return len(self._outputs)

    def get_output(self, index : int) -> GraphNode:
        if not self._is_correct_output_index(index):
            return None
        else:
            return self._outputs[index]
        
    def get_output_name(self, index : int) -> str:
        if not self._is_correct_output_index(index):
            return None
        else:
            return self._output_names[index]
    
    def add_output(self, output : GraphNode, output_name : str) -> bool:
        if self._is_output_name_in_list(output_name) or self._is_output_node_in_list(output):
            return False
        self._outputs.append(output)
        self._output_names.append(output_name)
        return True
    
    def remove_output(self, index : int) -> bool:
        if not self._is_correct_output_index(index):
            return False
        self._outputs.pop(index)
        self._output_names.pop(index)
    
    def clear_outputs(self) -> None:
        self._outputs.clear()

    def n_attributes(self) -> int:
        return len(self._attributes)

    def add_attribute(self, key : str, value) -> bool:
        if self._is_attribute_in_list(key):
            return False
        self._attributes[key] = value
        return True

    def remove_attribute(self, key : str) -> bool:
        if not self._is_attribute_in_list(key):
            return False
        self._attributes.pop(key)
        return True

    def get_attributes_keys(self) -> list[str]:
        return list(self._attributes.keys())

    def get_attribute_value(self, key : str):
        if not self._is_attribute_in_list(key):
            return None
        else:
            return self._attributes[key]

    def clear_attributes(self):
        self._attributes.clear()

    def _is_correct_input_index(self, index : int) -> bool:
        if index < 0 or index >= self.n_inputs():
            return False
        else:
            return True
    
    def _is_input_name_in_list(self, input_name : str) -> bool:
        for name in self._input_names:
            if name == input_name:
                return True
        return False
    
    def _is_input_node_in_list(self, node : GraphNode) -> bool:
        for input in self._inputs:
            if input.get_name() == node.get_name():
                return True
        return False
    
    def _is_correct_output_index(self, index : int) -> bool:
        if index < 0 or index >= self.n_outputs():
            return False
        else:
            return True
    
    def _is_output_name_in_list(self, output_name : str) -> bool:
        for name in self._output_names:
            if name == output_name:
                return True
        return False
    
    def _is_output_node_in_list(self, node : GraphNode) -> bool:
        for output in self._outputs:
            if output.get_name() == node.get_name():
                return True
        return False
    
    def _is_attribute_in_list(self, key : str):
        keys : list[str] = self.get_attributes_keys()
        for k in keys:
            if k == key:
                return True
        return False