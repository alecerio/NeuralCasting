import onnx
from compiler.frontend.parser.node_types.node_type import NodeType
from compiler.frontend.parser.node_types.tensor_type import TensorType
from compiler.frontend.parser.node.node import Node
from compiler.frontend.parser.node.input_node import InputNode
from compiler.frontend.parser.node.output_node import OutputNode

def parse(config):
    # load onnx file and create onnx graph
    graph : onnx.onnx_ml_pb2.GraphProto = __create_onnx_graph__(config)

    # create list of nodes
    nodes : list[Node] = []

    # create input nodes
    __create_input_nodes__(graph, nodes)

    # create output nodes
    __create_output_nodes__(graph, nodes)

    # print nodes
    for node in nodes:
        print(node)
        print(" ------------- ")

    ## Print basic information about the graph
    #print(f"Number of nodes in the graph: {len(graph.node)}")
    #print(f"Input names: {graph.input}")
    #print(f"Output names: {graph.output}")
    #
    #print("Input information:")
    #for input_info in graph.input:
    #    print(f"Name: {input_info.name}")
    #    print(f"Type: {input_info.type}")
    #    print(f"Shape: {input_info.type.tensor_type.shape.dim}")
    #    print()
#
    ## Access and print information about outputs
    #print("Output information:")
    #for output_info in graph.output:
    #    print(f"Name: {output_info.name}")
    #    print(f"Type: {output_info.type}")
    #    print(f"Shape: {output_info.type.tensor_type.shape.dim}")
    #    print()
    #
    ## Iterate through the nodes in the graph
    #for node in graph.node:
    #    print(f"Node name: {node.name}")
    #    print(f"Op type: {node.op_type}")
    #    print(f"Input names: {node.input}")
    #    print(f"Output names: {node.output}")


def __create_onnx_graph__(config):
    temp_path : str = str(config.temp_path)
    name : str = str(config.name)
    path = temp_path + "/" + name + ".onnx"
    model = onnx.load(path)
    graph = model.graph
    return graph

def __create_input_nodes__(graph : onnx.onnx_ml_pb2.GraphProto, nodes : list[Node]):
    for input in graph.input:
        name = input.name
        type_info : str = str(input.type)
        type_name : str = type_info.split("{")[0].strip()
        if type_name == 'tensor_type':
            shape = input.type.tensor_type.shape.dim
            shape_values = [0] * len(shape)
            for i in range(len(shape)):
                shape_values[i] = shape[i].dim_value
            elem_type = input.type.tensor_type.elem_type
            
            node_type : NodeType = TensorType(shape=shape_values, elem_type=elem_type)
            input_node : InputNode = InputNode(name=name, type=node_type)
            nodes.append(input_node)
        else:
            raise Exception("Error: unexpected type name of the input node")

def __create_output_nodes__(graph : onnx.onnx_ml_pb2.GraphProto, nodes : list[Node]):
    for output in graph.output:
        name = output.name
        type_info : str = str(output.type)
        type_name : str = type_info.split("{")[0].strip()
        if type_name == 'tensor_type':
            shape = output.type.tensor_type.shape.dim
            shape_values = [0] * len(shape)
            for i in range(len(shape)):
                shape_values[i] = shape[i].dim_value
            elem_type = output.type.tensor_type.elem_type
            
            node_type : NodeType = TensorType(shape=shape_values, elem_type=elem_type)
            output_node : OutputNode = OutputNode(name=name, type=node_type)
            nodes.append(output_node)
        else:
            raise Exception("Error: unexpected type name of the output node")