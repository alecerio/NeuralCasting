from graph.ncast_graph import NCastGraph
import node_fusion.dq_prelu_bnorm_q_rewrite as dpbq

if __name__ == "__main__":
    model_path = "/media/alessandro/SecondDisk1/ncast/NeuralCasting/onnx/convsenet_int8_optimized.onnx"
    #model_path = "/media/alessandro/SecondDisk1/ncast/NeuralCasting/onnx/nsnet2.onnx"
    ncast_graph = NCastGraph(model_path)
    output = ncast_graph.compile(None)
    
    with open(f"/home/alessandro/Desktop/aaa/{ncast_graph.graph.name}.c", "w", encoding="utf-8") as f:
        f.write(output[0])

    with open(f"/home/alessandro/Desktop/aaa/{ncast_graph.graph.name}.h", "w", encoding="utf-8") as f:
        f.write(output[1])

    