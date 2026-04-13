import numpy as np
import onnxruntime as ort

MODEL_PATH = "/media/alessandro/SecondDisk1/ncast/NeuralCasting/onnx/convsenet.onnx"
INPUT_SHAPE = (1, 257, 50)

# Crea sessione
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# Prende info input (nome + tipo)
inp = sess.get_inputs()[0]
input_name = inp.name
input_type = inp.type  # es: "tensor(float)"

# Mappa tipo ORT -> numpy dtype
ort2np = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}
dtype = ort2np.get(input_type, np.float32)

# Input di soli 1
x = np.ones(INPUT_SHAPE, dtype=dtype)*0.5

# Inferenzia (None = tutti gli output del modello)
outputs = sess.run(None, {input_name: x})

# Stampa info base sugli output
out_infos = sess.get_outputs()
for info, val in zip(out_infos, outputs):
    print(f"{info.name}: shape={getattr(val, 'shape', None)}, dtype={getattr(val, 'dtype', type(val))}")
    print(val)
    np.save(f"output_{info.name}.npy", val)
