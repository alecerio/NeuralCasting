from graph.ncast_graph import NCastGraph
from typing import List

def rewrite(ncast_graph: NCastGraph) -> None:
    start_idx = 0
    found_pattern = True
    while found_pattern:
        idx = _find_pattern(ncast_graph, start_idx)
        if idx == -1:
            found_pattern = False
            continue
        else:
            ncast_graph
        start_idx = idx + 1

def _find_pattern(ncast_graph: NCastGraph, start_idx: int) -> int:
    n_nodes = len(ncast_graph.graph.node)
    for idx in range(n_nodes-3):
        node_0 = ncast_graph.graph.node[idx]
        node_1 = ncast_graph.graph.node[idx+1]
        node_2 = ncast_graph.graph.node[idx+2]
        node_3 = ncast_graph.graph.node[idx+3]

        if (node_0.op_type == "DequantizeLinear" and
            node_1.op_type == "PRelu" and
            node_2.op_type == "BatchNormalization" and
            node_3.op_type == "QuantizeLinear"):
            return idx
    return -1
            

    
