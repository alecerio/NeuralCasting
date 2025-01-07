import onnxruntime as ort
import numpy as np

# Carica il modello ONNX
model_path = "nsnet2_reimplemented_int8_static_debug.onnx"
session = ort.InferenceSession(model_path)

# Nome degli input
input_names = [inp.name for inp in session.get_inputs()]
print("Input del modello:", input_names)

# Nome degli output
output_names = [out.name for out in session.get_outputs()]
print("Output del modello:", output_names)

# Prepara i tensori di input
# in_noisy è un tensore di dimensioni 1x1x257 riempito di 1
in_noisy = np.ones((1, 1, 257), dtype=np.float32) 

# h1 e h2 sono tensori di dimensioni 1x1x400 riempiti di 0
h1 = np.zeros((1, 1, 400), dtype=np.float32)
h2 = np.zeros((1, 1, 400), dtype=np.float32)

# Crea il dizionario degli input
inputs = {
    "in_noisy": in_noisy,
    "h1": h1,
    "h2": h2,
}

# Esegui l'inferenza
outputs = session.run(None, inputs)

# Stampa i risultati degli output
for name, output in zip(output_names, outputs):
    print(f"Output '{name}':")
    print(output)
    print("-" * 40)
