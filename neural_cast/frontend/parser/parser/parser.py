import onnx
import numpy as np
from neural_cast.frontend.parser.node_types.node_type import NodeType
from neural_cast.frontend.parser.node_types.tensor_type import TensorType
from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.parser.node.input_node import InputNode
from neural_cast.frontend.parser.node.init_node import InitializerNode
from neural_cast.frontend.parser.node.output_node import OutputNode
from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.ops.gemm import Gemm
from neural_cast.frontend.parser.ops.relu import ReLu
from neural_cast.frontend.parser.ops.sigmoid import Sigmoid
from neural_cast.frontend.parser.ops.tanh import Tanh
from neural_cast.frontend.parser.ops.add import Add
from neural_cast.frontend.parser.ops.mul import Mul
from neural_cast.frontend.parser.ops.sub import Sub
from neural_cast.frontend.parser.ops.matmul import MatMul
from neural_cast.frontend.parser.ops.constant import Constant
from neural_cast.frontend.parser.ops.gather import Gather
from neural_cast.frontend.parser.ops.transpose import Transpose
from neural_cast.frontend.parser.ops.squeeze import Squeeze
from neural_cast.frontend.common.common import CompilerLogger
from neural_cast.frontend.exceptions.CompilerException import CompilerException
from neural_cast.frontend.common.common import CompilerConfig
from neural_cast.frontend.parser.ops.gru import GRU

def parse() -> list[Node]:
    CompilerLogger().info("run parser")

    # load onnx file and create onnx graph
    graph : onnx.onnx_ml_pb2.GraphProto = _create_onnx_graph()

    # create list of nodes
    nodes : list[Node] = []

    # create input nodes
    _create_input_nodes(graph, nodes)

    # create init nodes
    _create_init_nodes(graph, nodes)

    # create output nodes
    _create_output_nodes(graph, nodes)

    # create op nodes
    [in_dict, out_dict] = _create_op_nodes(graph, nodes)

    # update op nodes references
    _update_opnodes_references(nodes, in_dict, out_dict)

    # update init nodes references
    _update_init_nodes_references(nodes)

    # update input nodes references
    _update_input_nodes_references(nodes)

    # update output nodes references
    _update_output_nodes_references(nodes)

    return nodes

def _create_onnx_graph():
    CompilerLogger().info("Create onnx graph")

    temp_path : str = str(CompilerConfig()['temp_path'])
    name : str = str(CompilerConfig()['name'])
    path = temp_path + "/" + name + ".onnx"
    model = onnx.load(path)
    graph = model.graph

    return graph

def _create_input_nodes(graph : onnx.onnx_ml_pb2.GraphProto, nodes : list[Node]):
    CompilerLogger().info("Create input nodes")
    for input in graph.input:
        CompilerLogger().info("Create input node: " + input.name)

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
            raise CompilerException("Error: unexpected type name of the input node")

def _create_init_nodes(graph : onnx.onnx_ml_pb2.GraphProto, nodes : list[Node]):
    CompilerLogger().info("Create init nodes")
    for init in graph.initializer:
        CompilerLogger().info("Create initializer node: " + init.name)

        name : str = init.name
        data_type = init.data_type
        tensor = onnx.numpy_helper.to_array(init)

        init_node : InitializerNode = InitializerNode(name, tensor, data_type)
        nodes.append(init_node)

def _create_output_nodes(graph : onnx.onnx_ml_pb2.GraphProto, nodes : list[Node]):
    CompilerLogger().info("Create output nodes")

    for output in graph.output:
        CompilerLogger().info("Create output node: " + output.name)

        name = output.name
        type_info : str = str(output.type)
        type_name : str = type_info.split("{")[0].strip()
        if type_name == 'tensor_type' or type_name == '':
            shape = output.type.tensor_type.shape.dim
            shape_values = [0] * len(shape)
            for i in range(len(shape)):
                shape_values[i] = shape[i].dim_value
            elem_type = output.type.tensor_type.elem_type
            
            node_type : NodeType = TensorType(shape=shape_values, elem_type=elem_type)
            output_node : OutputNode = OutputNode(name=name, type=node_type)
            nodes.append(output_node)
        else:
            raise CompilerException("Error: unexpected type name of the output node")

def _create_op_nodes(graph : onnx.onnx_ml_pb2.GraphProto, nodes : list[Node]) -> [dict, dict]:
    CompilerLogger().info("Create op nodes")
    in_dict = {}
    out_dict = {}
    for op in graph.node:
        CompilerLogger().info("Create op node: " + op.name)

        name : str = op.name
        optype : str = op.op_type
        in_dict[name] = op.input
        out_dict[name] = op.output
        if optype == 'Gemm':
            opnode : Gemm = Gemm(name)
        elif optype == 'Relu':
            opnode : ReLu = ReLu(name)
        elif optype == 'Sigmoid':
            opnode : Sigmoid = Sigmoid(name)
        elif optype == 'Tanh':
            opnode : Tanh = Tanh(name)
        elif optype == 'Add':
            opnode : Add = Add(name)
        elif optype == 'Mul':
            opnode : Mul = Mul(name)
        elif optype == 'Sub':
            opnode : Sub = Sub(name)
        elif optype == 'MatMul':
            opnode : MatMul = MatMul(name)
        elif optype == 'Constant':
            data_type : int = int(op.attribute[0].t.data_type)
            tensor = onnx.numpy_helper.to_array(op.attribute[0].t)
            opnode : Constant = Constant(name, tensor, data_type)
        elif optype == 'Gather':
            axis : int = int(op.attribute[0].i)
            opnode : Gather = Gather(name, axis)
        elif optype == 'Transpose':
            perm = onnx.helper.get_attribute_value(op.attribute[0])
            opnode : Transpose = Transpose(name, perm)
        elif optype == 'Squeeze':
            opnode : Squeeze = Squeeze(name)
        elif optype == 'GRU':
            opnode : GRU = GRU(name)
        else:
            raise CompilerException("Error: unexpected operation node: " + optype)
        nodes.append(opnode)
    
    return [in_dict, out_dict]

def _update_opnodes_references(nodes : list[Node], in_dict : dict, out_dict : dict) -> None:
    CompilerLogger().info("Create input / output nodes references for op nodes")
    for node in nodes:
        if isinstance(node, OpNode):
            CompilerLogger().info("Create input / output nodes references for op node: " + node.get_name())
            in_names = in_dict[node.get_name()]
            out_names = out_dict[node.get_name()]
            if isinstance(node, Gemm):
                _fill_gemm_node(node, nodes, in_names, out_names, in_dict, out_dict)
            elif isinstance(node, ReLu):
                _fill_relu_node(node, nodes, in_names, out_names, in_dict, out_dict)
            elif isinstance(node, Sigmoid):
                _fill_sigmoid_node(node, nodes, in_names, out_names, in_dict, out_dict)
            elif isinstance(node, Tanh):
                _fill_tanh_node(node, nodes, in_names, out_names, in_dict, out_dict)
            elif isinstance(node, Add):
                _fill_add_node(node, nodes, in_names, out_names, in_dict, out_dict)
            elif isinstance(node, Mul):
                _fill_mul_node(node, nodes, in_names, out_names, in_dict, out_dict)
            elif isinstance(node, Sub):
                _fill_sub_node(node, nodes, in_names, out_names, in_dict, out_dict)
            elif isinstance(node, MatMul):
                _fill_matmul_node(node, nodes, in_names, out_names, in_dict, out_dict)
            elif isinstance(node, Constant):
                _fill_constant_node(node, nodes, out_names, in_dict)
            elif isinstance(node, Gather):
                _fill_gather_node(node, nodes, in_names, out_names, in_dict, out_dict)
            elif isinstance(node, Transpose):
                _fill_transpose_node(node, nodes, in_names, out_names, in_dict, out_dict)
            elif isinstance(node, Squeeze):
                _fill_squeeze_node(node, nodes, in_names, out_names, in_dict, out_dict)
            elif isinstance(node, GRU):
                _fill_gru_node(node, nodes, in_names, out_names, in_dict, out_dict)
            else:
                raise CompilerException("Error: unexpected op node")

def _update_init_nodes_references(nodes : list[Node]) -> None:
    CompilerLogger().info("Create references for init nodes")
    for init_node in nodes:
        if isinstance(init_node, InitializerNode):
            CompilerLogger().info("Create references for init node: " + init_node.get_name())
            for op_node in nodes:
                if isinstance(op_node, OpNode):
                    if init_node.get_name() in op_node.get_input_names():
                        init_node.append_output_node(op_node)

def _update_input_nodes_references(nodes : list[Node]) -> None:
    CompilerLogger().info("Create references for input nodes")
    for input_node in nodes:
        if isinstance(input_node, InputNode):
            CompilerLogger().info("Create references for input node: " + input_node.get_name())
            for op_node in nodes:
                if isinstance(op_node, OpNode):
                    if input_node.get_name() in op_node.get_input_names():
                        input_node.append_output_node(op_node)


def _update_output_nodes_references(nodes : list[Node]):
    CompilerLogger().info("Create references for output nodes")
    for output_node in nodes:
        if isinstance(output_node, OutputNode):
            CompilerLogger().info("Create references for output node: " + output_node.get_name())
            for op_node in nodes:
                if isinstance(op_node, OpNode):
                    if output_node.get_name() in op_node.get_output_names():
                        output_node.append_input_node(op_node)

def _fill_gemm_node(node : Gemm, nodes : list[Node], in_names : list[str], out_names : list[str], in_dict : dict, out_dict : dict):
    CompilerLogger().info("Update Gemm node")
    in_node = _get_input_node_reference(nodes, in_names[0], out_dict)
    in_node_w = _get_input_node_reference(nodes, in_names[1], out_dict)
    in_node_b = _get_input_node_reference(nodes, in_names[2], out_dict)
    out_node = _get_output_node_reference(nodes, out_names[0], in_dict)
    node.append_input(in_node, in_names[0])
    node.append_input(in_node_w, in_names[1])
    node.append_input(in_node_b, in_names[2])
    node.append_output(out_node, out_names[0])

def _fill_relu_node(node : ReLu, nodes : list[Node], in_names : list[str], out_names : list[str], in_dict : dict, out_dict : dict):
    CompilerLogger().info("Update ReLu node")
    in_node = _get_input_node_reference(nodes, in_names[0], out_dict)
    out_node = _get_output_node_reference(nodes, out_names[0], in_dict)
    node.append_input(in_node, in_names[0])
    node.append_output(out_node, out_names[0])

def _fill_sigmoid_node(node : Sigmoid, nodes : list[Node], in_names : list[str], out_names : list[str], in_dict : dict, out_dict : dict):
    CompilerLogger().info("Update Sigmoid node")
    in_node = _get_input_node_reference(nodes, in_names[0], out_dict)
    out_node = _get_output_node_reference(nodes, out_names[0], in_dict)
    node.append_input(in_node, in_names[0])
    node.append_output(out_node, out_names[0])

def _fill_tanh_node(node : Tanh, nodes : list[Node], in_names : list[str], out_names : list[str], in_dict : dict, out_dict : dict):
    CompilerLogger().info("Update Tanh node")
    in_node = _get_input_node_reference(nodes, in_names[0], out_dict)
    out_node = _get_output_node_reference(nodes, out_names[0], in_dict)
    node.append_input(in_node, in_names[0])
    node.append_output(out_node, out_names[0])

def _fill_add_node(node : Add, nodes : list[Node], in_names : list[str], out_names : list[str], in_dict : dict, out_dict : dict):
    CompilerLogger().info("Update Add node")
    in_node_1 = _get_input_node_reference(nodes, in_names[0], out_dict)
    in_node_2 = _get_input_node_reference(nodes, in_names[1], out_dict)
    out_node = _get_output_node_reference(nodes, out_names[0], in_dict)
    node.append_input(in_node_1, in_names[0])
    node.append_input(in_node_2, in_names[1])
    node.append_output(out_node, out_names[0])

def _fill_mul_node(node : Mul, nodes : list[Node], in_names : list[str], out_names : list[str], in_dict : dict, out_dict : dict):
    CompilerLogger().info("Update Mul node")
    in_node_1 = _get_input_node_reference(nodes, in_names[0], out_dict)
    in_node_2 = _get_input_node_reference(nodes, in_names[1], out_dict)
    out_node = _get_output_node_reference(nodes, out_names[0], in_dict)
    node.append_input(in_node_1, in_names[0])
    node.append_input(in_node_2, in_names[1])
    node.append_output(out_node, out_names[0])

def _fill_sub_node(node : Sub, nodes : list[Node], in_names : list[str], out_names : list[str], in_dict : dict, out_dict : dict):
    CompilerLogger().info("Update Sub node")
    in_node_1 = _get_input_node_reference(nodes, in_names[0], out_dict)
    in_node_2 = _get_input_node_reference(nodes, in_names[1], out_dict)
    out_node = _get_output_node_reference(nodes, out_names[0], in_dict)
    node.append_input(in_node_1, in_names[0])
    node.append_input(in_node_2, in_names[1])
    node.append_output(out_node, out_names[0])

def _fill_matmul_node(node : MatMul, nodes : list[Node], in_names : list[str], out_names : list[str], in_dict : dict, out_dict : dict):
    CompilerLogger().info("Update MatMul node")
    in_node_1 = _get_input_node_reference(nodes, in_names[0], out_dict)
    in_node_2 = _get_input_node_reference(nodes, in_names[1], out_dict)
    out_node = _get_output_node_reference(nodes, out_names[0], in_dict)
    node.append_input(in_node_1, in_names[0])
    node.append_input(in_node_2, in_names[1])
    node.append_output(out_node, out_names[0])

def _fill_constant_node(node : Constant, nodes : list[Node], out_names : list[str], in_dict : dict):
    CompilerLogger().info("Update Constant node")
    out_node = _get_output_node_reference(nodes, out_names[0], in_dict)
    node.append_output(out_node, out_names[0])

def _fill_gather_node(node : Gather, nodes : list[Node], in_names : list[str], out_names : list[str], in_dict : dict, out_dict : dict):
    CompilerLogger().info("Update Gather node")
    in_node_1 = _get_input_node_reference(nodes, in_names[0], out_dict)
    in_node_2 = _get_input_node_reference(nodes, in_names[1], out_dict)
    out_node = _get_output_node_reference(nodes, out_names[0], in_dict)
    node.append_input(in_node_1, in_names[0])
    node.append_input(in_node_2, in_names[1])
    node.append_output(out_node, out_names[0])

def _fill_transpose_node(node : Transpose, nodes : list[Node], in_names : list[str], out_names : list[str], in_dict : dict, out_dict : dict):
    CompilerLogger().info("Update Transpose node")
    in_node = _get_input_node_reference(nodes, in_names[0], out_dict)
    out_node = _get_output_node_reference(nodes, out_names[0], in_dict)
    node.append_input(in_node, in_names[0])
    node.append_output(out_node, out_names[0])

def _fill_squeeze_node(node : Squeeze, nodes : list[Node], in_names : list[str], out_names : list[str], in_dict : dict, out_dict : dict):
    CompilerLogger().info("Update Squeeze node")
    n_inputs : int =  len(in_names)
    in_node = _get_input_node_reference(nodes, in_names[0], out_dict)
    if n_inputs == 2:
        axes_node = _get_input_node_reference(nodes, in_names[1], out_dict)
    out_node = _get_output_node_reference(nodes, out_names[0], in_dict)
    node.append_input(in_node, in_names[0])
    if n_inputs == 2:
        node.append_input(axes_node, in_names[1])
    node.append_output(out_node, out_names[0])

def _fill_gru_node(node : GRU, nodes : list[Node], in_names : list[str], out_names : list[str], in_dict : dict, out_dict : dict):
    CompilerLogger().info("Update GRU node")
    in_node = _get_input_node_reference(nodes, in_names[0], out_dict)
    in_node_W = _get_input_node_reference(nodes, in_names[1], out_dict)
    in_node_R = _get_input_node_reference(nodes, in_names[2], out_dict)
    in_node_B = _get_input_node_reference(nodes, in_names[3], out_dict)
    in_node_initH = _get_input_node_reference(nodes, in_names[5], out_dict)
    out_node = _get_output_node_reference(nodes, out_names[0], in_dict)
    out_hidden = _get_output_node_reference(nodes, out_names[1], in_dict)

    node.append_input(in_node, in_names[0])
    node.append_input(in_node_W, in_names[1])
    node.append_input(in_node_R, in_names[2])
    node.append_input(in_node_B, in_names[3])
    node.append_input(in_node_initH, in_names[5])
    node.append_output(out_node, out_names[0])
    node.append_output(out_hidden, out_names[1])

def _get_input_node_reference(nodes : list[Node], in_name : str, out_dict : dict) -> Node:
    for node in nodes:
        name : str = node.get_name()
        if isinstance(node, OpNode):
            out_list : list[str] = out_dict[name]
            if in_name in out_list:
                return node
        elif isinstance(node, InputNode):
            if in_name == name:
                return node
        elif isinstance(node, InitializerNode):
            if in_name == name:
                return node
    return None

def _get_output_node_reference(nodes : list[Node], out_name : str, in_dict : dict) -> Node:
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