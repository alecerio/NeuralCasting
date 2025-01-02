import onnxruntime as ort
import numpy as np
#from sklearn.metrics import mean_squared_error

def mean_squared_error(y_true, y_pred):
    print(len(y_true))
    print(len(y_pred))
    """
    Calcola il Mean Squared Error (MSE) tra due array.
    
    Parameters:
    - y_true: Array dei valori veri.
    - y_pred: Array dei valori predetti.
    
    Returns:
    - mse: Mean Squared Error.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Le due liste devono avere la stessa lunghezza.")
    
    # Somma dei quadrati delle differenze
    squared_diff = [(a - b) ** 2 for a, b in zip(y_true, y_pred)]
    
    # MSE = media delle differenze quadratiche
    mse = sum(squared_diff) / len(y_true)
    
    return mse


def run_inference(model_path, inputs):
    """
    Esegue l'inferenza su un modello ONNX con gli input forniti.
    """
    session = ort.InferenceSession(model_path)
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    # Prepara i dati di input
    feed_dict = {name: inputs[name] for name in input_names}

    # Esegui inferenza
    outputs = session.run(output_names, feed_dict)

    return outputs

# Input simulati
inputs = {
    "in_noisy": np.random.rand(1, 1, 257).astype(np.float32),
    "h1": np.random.rand(1, 1, 400).astype(np.float32),
    "h2": np.random.rand(1, 1, 400).astype(np.float32),
}

# Modelli ONNX
original_model = "nsnet2_reimplemented.onnx"
quantized_model = "nsnet2_reimplemented_uint8_static.onnx"

# Esegui inferenza per entrambi i modelli
outputs_original = run_inference(original_model, inputs)[0]
outputs_quantized = run_inference(quantized_model, inputs)[0]

mse = mean_squared_error(outputs_original, outputs_quantized)
print(mse)

# Calcola MSE tra i due set di output
#mse_values = []
#for orig, quant in zip(outputs_original, outputs_quantized):
#    mse = mean_squared_error(orig.flatten(), quant.flatten())
#    mse_values.append(mse)
#    print(f"MSE tra gli output: {mse}")
#
## Stampa i risultati
#print("\nConfronto tra i due modelli:")
#for i, mse in enumerate(mse_values):
#    print(f"Output {i + 1}: MSE = {mse}")
