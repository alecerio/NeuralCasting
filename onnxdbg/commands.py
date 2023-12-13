from onnxdbg.graph.graph import Graph
from onnxdbg.utils import create_graph

def onnxdbg_copy(**kwargs):
    src : str = str(kwargs['src'])
    dst : str = str(kwargs['dst'])
    mdl : str = str(kwargs['mdl'])
    graph : Graph = create_graph(src)
    graph.export_onnx_file(mdl, dst)