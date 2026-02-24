import argparse
import onnx
import onnxoptimizer
import numpy as np
from config.config import ONNX_DIR

from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader,
    QuantType, QuantFormat, CalibrationMethod
)

class MyDataReader(CalibrationDataReader):
    def __init__(self, input_name, samples):
        self.input_name = input_name
        self.samples = iter(samples)

    def get_next(self):
        try:
            x = next(self.samples)
            return {self.input_name: x}
        except StopIteration:
            return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ONNX Quantizer")

    parser.add_argument("--model", type=str, required=True,
                    help="ONNX file name (without path)")
    parser.add_argument("--path", type=str, default=ONNX_DIR,
                    help="Input/Output ONNX file path")

    args = parser.parse_args()



    model_fp32 = f"{args.path}/{args.model}.onnx"
    model_int8 = f"{args.path}/{args.model}_int8.onnx"

    input_name = "stft_noisy"
    samples = [np.random.randn(1, 257, 50).astype(np.float32) for _ in range(50)]
    dr = MyDataReader(input_name, samples)

    quantize_static(
        model_input=model_fp32,
        model_output=model_int8,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax
    )

    print("Saved: ", model_int8)


    m = onnx.load(f"{args.path}/{args.model}_int8.onnx")
    passes = ["eliminate_identity"]
    m2 = onnxoptimizer.optimize(m, passes)
    onnx.save(m2, f"{args.path}/{args.model}_int8_optimized.onnx")
