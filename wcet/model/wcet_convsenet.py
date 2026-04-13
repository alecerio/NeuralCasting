from config.config import ONNX_DIR
from wcet.model.wcet_model import wcet_model, analyze_output

if __name__ == '__main__':
    onnx_path = f"{ONNX_DIR}/convsenet_int8_optimized.onnx"
    wcet_model(onnx_path)
    analyze_output()
