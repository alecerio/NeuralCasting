import pickle
import onnx
from onnx import helper
from onnxdbg.graph.graph import Graph
from onnxdbg.graph.graph_node import GraphNode
from onnxdbg.graph.output_node import OutputNode
from onnxdbg.graph.op_node import OpNode
from onnxdbg.utils import create_graph, inference_onnx_runtime

def onnxdbg_copy(**kwargs):
    src : str = str(kwargs['src'])
    dst : str = str(kwargs['dst'])
    mdl : str = str(kwargs['mdl'])
    graph : Graph = create_graph(src)
    graph.export_onnx_file(mdl, dst)

def onnxdbg_subgr(**kwargs):
    src : str = str(kwargs['src'])
    dst : str = str(kwargs['dst'])
    mdl : str = str(kwargs['mdl'])
    out : str = str(kwargs['out'])
    graph : Graph = create_graph(src)
    [initializers, inputs, opnodes, outputs] = graph.create_subgraph_data(out)
    graph = helper.make_graph(opnodes, mdl, inputs, outputs, initializers)
    model = helper.make_model(graph)
    onnx.save(model, dst)

def onnxdbg_inferdbg(**kwargs):
    srcp : str = str(kwargs['srcp'])
    dstp : str = str(kwargs['dstp'])
    mdl : str = str(kwargs['mdl'])
    input_data = kwargs['input']
    
    src : str = srcp + mdl + '.onnx'
    dst : str = dstp + mdl + '.onnx'
    input_names : list[str] = input_data.keys()

    graph : Graph = create_graph(src)
    n_nodes : int = graph.n_nodes()

    dict_output = {}
    dict_output_shape = {}

    for i in range(n_nodes):
        node : GraphNode = graph.get_node(i)
        if isinstance(node, OpNode) or isinstance(node, OutputNode):
            node_name : str = node.get_name()
            [initializers, inputs, opnodes, outputs] = graph.create_subgraph_data(node_name)
            graph_onnx = helper.make_graph(opnodes, mdl, inputs, outputs, initializers)
            model_onnx = helper.make_model(graph_onnx)
            node_name_file_format = 'file_' + node_name.replace(":", "").replace("/", "")
            temp_onnx_name : str = dstp + node_name_file_format + '.onnx'
            onnx.save(model_onnx, temp_onnx_name)

            sub_inputs = []
            model_temp = onnx.load(temp_onnx_name)
            input_names_temp = [input.name for input in model_temp.graph.input]
            for input_name_temp in input_names:
                if input_name_temp in input_names_temp:
                    sub_inputs.append(input_data[input_name_temp])
            
            [outputs_ort, outputs_shape_ort] = inference_onnx_runtime(temp_onnx_name, sub_inputs)

            dict_output[node_name] = outputs_ort
            dict_output_shape[node_name] = outputs_shape_ort
    
    pickle_output_file = dstp + 'output.pkl'
    with open(pickle_output_file, "wb") as pkl_file:
        pickle.dump(dict_output, pkl_file)
    
    pickle_output_shape_file = dstp + 'output_shape.pkl'
    with open(pickle_output_shape_file, "wb") as pkl_file:
        pickle.dump(dict_output_shape, pkl_file)