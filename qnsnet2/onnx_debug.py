import onnx
import onnxruntime as ort
import numpy as np

# Carica il modello ONNX
model_path = "nsnet2_reimplemented_int8_static_debug.onnx"
session = ort.InferenceSession(model_path)

# Ottieni tutti i nomi degli input e degli output
input_names = [inp.name for inp in session.get_inputs()]
output_names = [out.name for out in session.get_outputs()]

# Prepara dati fittizi per gli input
# Modifica questi valori in base al modello
input_data = {
    name: np.random.randn(*inp.shape).astype(np.float32)
    for name, inp in zip(input_names, session.get_inputs())
}

# Abilita il debug degli output intermedi
options = ort.SessionOptions()
options.log_severity_level = 0  # Log dettagliati

# Ricarica il modello con debug abilitato
session_debug = ort.InferenceSession(model_path, sess_options=options)

# Esegui l'inferenza con l'output di tutti i nodi intermedi
intermediate_outputs = session_debug.run(None, input_data)

# Stampa i risultati
for i, (name, output) in enumerate(zip(output_names, intermediate_outputs)):
    print(f"Output {i} ({name}):")
    print(output)
    print("-" * 40)
