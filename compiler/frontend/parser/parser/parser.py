import onnx
import numpy as np
from compiler.frontend.parser.node_types.node_type import NodeType
from compiler.frontend.parser.node_types.tensor_type import TensorType
from compiler.frontend.parser.node.node import Node
from compiler.frontend.parser.node.input_node import InputNode
from compiler.frontend.parser.node.output_node import OutputNode
from compiler.frontend.parser.node.op_node import OpNode
from compiler.frontend.parser.ops.gemm import Gemm
from compiler.frontend.parser.ops.relu import ReLu

def parse(config) -> list[Node]:
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

    return nodes

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
    
    # update op nodes references
    for node in nodes:
        if isinstance(node, OpNode):
            in_names = in_dict[node.get_name()]
            out_names = out_dict[node.get_name()]
            if isinstance(node, Gemm):
                __fill_gemm_node__(node, nodes, in_names, out_names, in_dict, out_dict, graph)
            elif isinstance(node, ReLu):
                __fill_relu_node__(node, nodes, in_names, out_names, in_dict, out_dict)
            else:
                raise Exception("Error: unexpected operation node")
    
    # update input nodes references
    for input_node in nodes:
        if isinstance(input_node, InputNode):
            for op_node in nodes:
                if isinstance(op_node, OpNode):
                    if input_node.get_name() in op_node.get_input_names():
                        input_node.append_output_node(op_node)
    
    # update output nodes references
    for output_node in nodes:
        if isinstance(output_node, OutputNode):
            for op_node in nodes:
                if isinstance(op_node, OpNode):
                    if output_node.get_name() in op_node.get_output_names():
                        output_node.append_input_node(op_node)

def __fill_gemm_node__(node : Gemm, nodes : list[Node], in_names : list[str], out_names : list[str], in_dict : dict, out_dict : dict, graph: onnx.onnx_ml_pb2.GraphProto):
    in_node = __get_input_node_reference__(nodes, in_names[0], out_dict)
    out_node = __get_output_node_reference__(nodes, out_names[0], in_dict)
    node.append_input(in_node, in_names[0])
    node.append_output(out_node, out_names[0])
    for init in graph.initializer:
        if init.name == in_names[1]:
            dims = init.dims
            break
    node.set_weights_and_bias(
        np.zeros((dims[1], dims[0]), dtype=float), 
        np.zeros((dims[0],), dtype=float)
    )

def __fill_relu_node__(node : ReLu, nodes : list[Node], in_names : list[str], out_names : list[str], in_dict : dict, out_dict : dict):
    in_node = __get_input_node_reference__(nodes, in_names[0], out_dict)
    out_node = __get_output_node_reference__(nodes, out_names[0], in_dict)
    node.append_input(in_node, in_names[0])
    node.append_output(out_node, out_names[0])

def __get_input_node_reference__(nodes : list[Node], in_name : str, out_dict : dict) -> Node:
    for node in nodes:
        name : str = node.get_name()
        if isinstance(node, OpNode):
            out_list : list[str] = out_dict[name]
            if in_name in out_list:
                return node
        elif isinstance(node, InputNode):
            if in_name == name:
                return node
    return None

def __get_output_node_reference__(nodes : list[Node], out_name : str, in_dict : dict) -> Node:
    for node in nodes:
        name : str = node.get_name()
        if isinstance(node, OpNode):
            in_list : list[str] = in_dict[name]
            if out_name in in_list:
                return node
        elif isinstance(node, OutputNode):
            if out_name == name:
                return node
    return None