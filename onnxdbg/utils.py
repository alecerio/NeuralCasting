import onnx
from onnxdbg.graph.op_node import OpNode
from onnxdbg.graph.graph_node import GraphNode
from onnxdbg.graph.init_node import InitNode
from onnxdbg.graph.input_node import InputNode
from onnxdbg.graph.output_node import OutputNode

def create_graph(onnx_path : str):
    model = onnx.load(onnx_path)
    nodes : list[GraphNode] = []

    # create op nodes
    for _, node in enumerate(model.graph.node):
        print("create op node: " + node.name)
        opnode : OpNode = OpNode(node.name, node)
        nodes.append(opnode)
    
    # create initializer nodes
    for init in model.graph.initializer:
        print("create initializer node: " + init.name)
        init_node : InitNode = InitNode(init.name, init)
        nodes.append(init_node)

    # create input nodes
    for input in model.graph.input:
        print("create input node: " + input.name)
        input_node : InputNode = InputNode(input.name, input)
        nodes.append(input_node)

    # create output nodes
    for output in model.graph.output:
        print("create output node: " + output.name)
        output_node : OutputNode = OutputNode(output.node, output)
        nodes.append(output_node)
    
    for node in nodes:
        if isinstance(node, OpNode):
            inputs = node.get_node().input
            outputs = node.get_node().output
            for other_node in nodes:
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
                
