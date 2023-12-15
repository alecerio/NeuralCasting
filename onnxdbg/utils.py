import onnx
import onnxruntime as ort
import numpy as np
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

def inference_onnx_runtime(path_onnx, input_data):
    session = ort.InferenceSession(path_onnx)
    
    name_dict = {}
    for i in range(len(input_data)):
        input_name = session.get_inputs()[i].name
        name_dict[input_name] = input_data[i]
    
    outputs = session.run(None, name_dict)

    n_outputs = len(outputs)
    outputs_onnx = []
    outputs_shape_onnx = []
    for i in range(n_outputs):
        output_onnx = outputs[i]
        output_shape_onnx = output_onnx.shape
        output_onnx = np.squeeze(output_onnx)
        output_onnx = output_onnx.flatten()
        outputs_onnx.append(output_onnx)
        outputs_shape_onnx.append(output_shape_onnx)
    return [outputs_onnx, outputs_shape_onnx]