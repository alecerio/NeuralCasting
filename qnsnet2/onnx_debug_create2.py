import onnx
from onnx import helper, TensorProto

# Carica il modello ONNX
model_path = "nsnet2_reimplemented_int8_static.onnx"
model = onnx.load(model_path)

# Ottieni il grafo del modello
graph = model.graph

# Nome dell'output che vuoi aggiungere al grafo
target_output_name = "/MatMul_4_output_0_quantized"

# Tipo di dato e forma del tensore
elem_type = TensorProto.INT8  # Tipo INT8
shape = [400]  # Forma del tensore (modifica se necessario)

# Cerca il tensore nel value_info del grafo
found = False
for value_info in graph.value_info:
    if value_info.name == target_output_name:
        # Aggiungi l'output al grafo
        graph.output.append(value_info)
        found = True
        print(f"Aggiunto '{target_output_name}' come output del grafo.")
        break

# Se il tensore non è trovato, crealo manualmente
if not found:
    print(f"'{target_output_name}' non trovato nel grafo. Creazione manuale del value_info.")
    new_value_info = helper.make_tensor_value_info(
        target_output_name,  # Nome del tensore
        elem_type,           # Tipo di dato (INT8)
        shape                # Forma del tensore
    )
    graph.output.append(new_value_info)
    print(f"Creato manualmente e aggiunto '{target_output_name}' come output del grafo.")

# Salva il modello modificato
debug_model_path = "nsnet2_reimplemented_int8_static_debug.onnx"
onnx.save(model, debug_model_path)
print(f"Modello modificato salvato in: {debug_model_path}")
