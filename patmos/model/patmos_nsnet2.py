from patmos.model.patmos_model import patmos_model, analyze_output
from config.config import ONNX_DIR

if __name__ == '__main__':
    onnx_path = f"{ONNX_DIR}/nsnet2_reimplemented_int8_optimized.onnx"
    patmos_model(onnx_path)
    analyze_output()