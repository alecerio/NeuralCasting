import onnx
from onnx import numpy_helper

# Carica il modello ONNX
model_path = "nsnet2_reimplemented_int8_static.onnx"
model = onnx.load(model_path)
graph = model.graph

print("Analisi completa del modello ONNX (senza valori binari)")
print("=" * 60)

# 1. Informazioni sugli Input
print("\n[1] Input del modello:")
for inp in graph.input:
    name = inp.name
    elem_type = inp.type.tensor_type.elem_type
    shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
    print(f"- Nome: {name}, Tipo: {onnx.TensorProto.DataType.Name(elem_type)}, Forma: {shape}")

# 2. Informazioni sugli Output
print("\n[2] Output del modello:")
for out in graph.output:
    name = out.name
    elem_type = out.type.tensor_type.elem_type
    shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
    print(f"- Nome: {name}, Tipo: {onnx.TensorProto.DataType.Name(elem_type)}, Forma: {shape}")

# 3. Informazioni sui Nodi
print("\n[3] Nodi nel grafo:")
for node in graph.node:
    print(f"- Nome: {node.name}, Tipo: {node.op_type}")
    print(f"  Input: {node.input}")
    print(f"  Output: {node.output}")
    for attr in node.attribute:
        print(f"  Attributo: {attr.name}")

# 4. Tensori (Value Info)
print("\n[4] Informazioni sui tensori intermedi:")
for value_info in graph.value_info:
    name = value_info.name
    elem_type = value_info.type.tensor_type.elem_type
    shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
    print(f"- Nome: {name}, Tipo: {onnx.TensorProto.DataType.Name(elem_type)}, Forma: {shape}")

# 5. Costanti e Initializer
print("\n[5] Informazioni sulle costanti e sugli initializer (senza valori):")
for initializer in graph.initializer:
    name = initializer.name
    elem_type = initializer.data_type
    shape = list(initializer.dims)
    print(f"- Nome: {name}, Tipo: {onnx.TensorProto.DataType.Name(elem_type)}, Forma: {shape}")

# 6. Controllo per Nodi Quantizzati
print("\n[6] Controllo per nodi di quantizzazione:")
quantized_nodes = []
for node in graph.node:
    if node.op_type in ["QuantizeLinear", "DequantizeLinear", "ConvInteger", "MatMulInteger"]:
        quantized_nodes.append(node)
        print(f"- Nodo quantizzato trovato: {node.name} ({node.op_type})")
        for attr in node.attribute:
            print(f"  Attributo: {attr.name}")

if not quantized_nodes:
    print("Nessun nodo quantizzato trovato.")

# 7. Altri Attributi del Modello
print("\n[7] Altri attributi del modello:")
if model.metadata_props:
    for prop in model.metadata_props:
        print(f"- {prop.key}: {prop.value}")
else:
    print("Nessun attributo aggiuntivo trovato.")

print("\nAnalisi completata.")
