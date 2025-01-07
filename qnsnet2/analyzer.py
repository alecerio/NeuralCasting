import onnx

def analyze_onnx_graph(model_path):
    # Carica il modello ONNX
    model = onnx.load(model_path)
    graph = model.graph

    print(f"Analisi del modello ONNX: {model_path}")
    print(f"Numero di nodi: {len(graph.node)}")
    print("=== Informazioni sui nodi ===")

    for idx, node in enumerate(graph.node):
        print(f"\nNodo {idx + 1}:")
        print(f"  Nome: {node.name if node.name else 'Nessun nome'}")
        print(f"  Tipo di operazione (op_type): {node.op_type}")

        # Input del nodo
        print(f"  Input:")
        for input_name in node.input:
            print(f"    - {input_name}")

        # Output del nodo
        print(f"  Output:")
        for output_name in node.output:
            print(f"    - {output_name}")

        # Attributi del nodo
        print(f"  Attributi:")
        for attr in node.attribute:
            print(f"    - {attr.name}: {onnx.helper.get_attribute_value(attr)}")

    
    #print("\n=== Informazioni sugli inizializzatori ===")
    #for initializer in graph.initializer:
    #    print(f"Inizializzatore: {initializer.name}")
    #    print(f"  Shape: {[dim.dim_value for dim in initializer.dims]}")
    #    print(f"  Tipo: {onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[initializer.data_type]}")
#
    #print("\n=== Informazioni sugli input del modello ===")
    #for input_tensor in graph.input:
    #    print(f"Input: {input_tensor.name}")
    #    shape = [
    #        dim.dim_value if dim.dim_value > 0 else "dinamico"
    #        for dim in input_tensor.type.tensor_type.shape.dim
    #    ]
    #    print(f"  Shape: {shape}")
    #    print(f"  Tipo: {onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_tensor.type.tensor_type.elem_type]}")
    #
    #print("\n=== Informazioni sugli output del modello ===")
    #for output_tensor in graph.output:
    #    print(f"Output: {output_tensor.name}")
    #    shape = [
    #        dim.dim_value if dim.dim_value > 0 else "dinamico"
    #        for dim in output_tensor.type.tensor_type.shape.dim
    #    ]
    #    print(f"  Shape: {shape}")
    #    print(f"  Tipo: {onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[output_tensor.type.tensor_type.elem_type]}")

if __name__ == "__main__":
    # Sostituisci con il percorso del tuo modello ONNX
    model_path = "nsnet2_reimplemented_uint8_static.onnx"
    analyze_onnx_graph(model_path)
