import onnx
from onnxdbg.graph.op_node import OpNode
from onnxdbg.graph.graph_node import GraphNode
from onnxdbg.graph.init_node import InitNode
from onnxdbg.graph.input_node import InputNode
from onnxdbg.graph.output_node import OutputNode
from onnxdbg.graph.graph import Graph

def create_graph(onnx_path : str) -> Graph:
    model = onnx.load(onnx_path)
    nodes : list[GraphNode] = []

    # create op nodes
    for _, node in enumerate(model.graph.node):
        opnode : OpNode = OpNode(node.name, node)
        nodes.append(opnode)
    
    # create initializer nodes
    for init in model.graph.initializer:
        init_node : InitNode = InitNode(init.name, init)
        nodes.append(init_node)

    # create input nodes
    for input in model.graph.input:
        input_node : InputNode = InputNode(input.name, input)
        nodes.append(input_node)

    # create output nodes
    for output in model.graph.output:
        output_node : OutputNode = OutputNode(output.name, output)
        nodes.append(output_node)
    
    graph : Graph = Graph(nodes)

    return graph