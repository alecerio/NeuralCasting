import onnx
from compiler.frontend.parser.node_types.node_type import NodeType
from compiler.frontend.parser.node_types.tensor_type import TensorType
from compiler.frontend.parser.node.node import Node
from compiler.frontend.parser.node.input_node import InputNode
from compiler.frontend.parser.node.output_node import OutputNode
from compiler.frontend.parser.node.op_node import OpNode
from compiler.frontend.parser.ops.gemm import Gemm
from compiler.frontend.parser.ops.relu import ReLu

def parse(config):
    # load onnx file and create onnx graph
    graph : onnx.onnx_ml_pb2.GraphProto = __create_onnx_graph__(config)

    # create list of nodes
    nodes : list[Node] = []

    # create input nodes
    __create_input_nodes__(graph, nodes)

    # create output nodes
    __create_output_nodes__(graph, nodes)

    # create op nodes
    __create_op_nodes__(graph, nodes)

    # print nodes
    for node in nodes:
        print(node)
        print(" ------------- ")

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

def __create_op_nodes__(graph : onnx.onnx_ml_pb2.GraphProto, nodes : list[Node]):
    in_dict = {}
    out_dict = {}
    for op in graph.node:
        name : str = op.name
        optype : str = op.op_type
        in_dict[name] = op.input
        out_dict[name] = op.output
        if optype == 'Gemm':
            opnode : Gemm = Gemm(name)
        elif optype == 'Relu':
            opnode : ReLu = ReLu(name)
        else:
            raise Exception("Error: unexpected operation node")
        nodes.append(opnode)
    
    for node in nodes:
        if isinstance(node, OpNode):
            in_names = in_dict[node.get_name()]
            out_names = out_dict[node.get_name()]
            if isinstance(node, Gemm):
                pass
            elif isinstance(node, ReLu):
                pass
            else:
                raise Exception("Error: unexpected operation node")
            
            print(node.get_name())
            print(in_names)
            print(out_names)

def __get_node_reference__(nodes : list[Node], name : str) -> Node:
    for node in nodes:
        if name == node.get_name():
            return node
    return None