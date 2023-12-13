import onnx
from onnx import helper
from onnxdbg.graph.graph import Graph
from onnxdbg.utils import create_graph

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