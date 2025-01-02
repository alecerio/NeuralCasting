from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import numpy as np
from onnxruntime.quantization import QuantFormat, QuantType

# Implementa un DataReader per fornire il dataset di calibrazione
class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, input_data):
        self.data = input_data
        self.index = 0

    def get_next(self):
        if self.index >= len(self.data):
            return None
        # Restituisce un dizionario con i nomi degli input e i dati corrispondenti
        batch = self.data[self.index]
        self.index += 1
        return {
            "in_noisy": batch[0],  # Input 1: (1, 1, 257)
            "h1": batch[1],  # Input 2: (1, 1, 400)
            "h2": batch[2],  # Input 3: (1, 1, 400)
        }

# Genera dati casuali per la calibrazione
calibration_data = [
    [
        np.random.rand(1, 1, 257).astype(np.float32),  # Input 1
        np.random.rand(1, 1, 400).astype(np.float32),  # Input 2
        np.random.rand(1, 1, 400).astype(np.float32),  # Input 3
    ]
    for _ in range(10)  # Usa 10 campioni di calibrazione
]

# Nome del modello originale e del modello quantizzato
model_input = "nsnet2_reimplemented.onnx"
model_output = "nsnet2_reimplemented_int8_static.onnx"

# Esegui la quantizzazione statica
quant_format = QuantFormat.QOperator
quantize_static(
    model_input,                       # Modello originale
    model_output,                      # Modello quantizzato
    MyCalibrationDataReader(calibration_data),  # DataReader per la calibrazione
    quant_format=quant_format,     # Quantizzazione unsigned int8
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8
)

print(f"Modello quantizzato salvato in {model_output}")
