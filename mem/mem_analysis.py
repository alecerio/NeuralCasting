from graph.ncast_graph import NCastGraph
from config.config import ONNX_DIR

def memory_analysis(onnx_path):
    ncgraph = NCastGraph(onnx_path)
    ncgraph.graph
    for op in ncgraph.ops:
        keys = list(op.out_dict.keys())
        for key in keys:
            shape = op.out_dict[key].shape
            size = 1
            for s in shape:
                size = size * s
            print(f"{op.onnx_unit.op_type};{op.onnx_unit.name};{size}")
    
    for initializer in ncgraph.graph.initializer:
        print(initializer.name, initializer.dims)


if __name__ == '__main__':
    memory_analysis(f"{ONNX_DIR}/nsnet2_reimplemented_int8.onnx")
