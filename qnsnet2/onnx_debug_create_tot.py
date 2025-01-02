import onnx
from onnx import helper, TensorProto

def adjust_name(name):
    if name[0] == '/':
        name = name[1:]
    if name[0] == '1' or name[0] == '2':
        name = '_' + name
    return name

class OutputDebug():
    def __init__(self, name, shape, type):
        self.name = name
        self.type = type
        self.shape = shape

outputs = [
    OutputDebug('/Squeeze_1_output_0', [1,400], TensorProto.FLOAT),
    OutputDebug('/Squeeze_2_output_0', [1,400], TensorProto.FLOAT),
    OutputDebug('/Squeeze_output_0', [1,257], TensorProto.FLOAT),
    OutputDebug('/Constant_output_0_quantized', [400,257], TensorProto.INT8),
    OutputDebug('/Constant_1_output_0_quantized', [400], TensorProto.INT8),
    OutputDebug('/Constant_2_output_0_quantized', [1,400,400], TensorProto.INT8),
    OutputDebug('/Constant_3_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Constant_4_output_0_quantized', [1,400,400], TensorProto.INT8),
    OutputDebug('/Constant_5_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Constant_6_output_0_quantized', [1,400,400], TensorProto.INT8),
    OutputDebug('/Constant_7_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Constant_8_output_0_quantized', [1,400,400], TensorProto.INT8),
    OutputDebug('/Constant_9_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Constant_10_output_0_quantized', [1,400,400], TensorProto.INT8),
    OutputDebug('/Constant_11_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Constant_12_output_0_quantized', [1,400,400], TensorProto.INT8),
    OutputDebug('/Constant_13_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Constant_15_output_0_quantized', [1,400,400], TensorProto.INT8),
    OutputDebug('/Constant_16_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Constant_17_output_0_quantized', [1,400,400], TensorProto.INT8),
    OutputDebug('/Constant_18_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Constant_19_output_0_quantized', [1,400,400], TensorProto.INT8),
    OutputDebug('/Constant_20_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Constant_21_output_0_quantized', [1,400,400], TensorProto.INT8),
    OutputDebug('/Constant_22_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Constant_23_output_0_quantized', [1,400,400], TensorProto.INT8),
    OutputDebug('/Constant_24_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Constant_25_output_0_quantized', [1,400,400], TensorProto.INT8),
    OutputDebug('/Constant_26_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Constant_28_output_0_quantized', [600,400], TensorProto.INT8),
    OutputDebug('/Constant_29_output_0_quantized', [600], TensorProto.INT8),
    OutputDebug('/Constant_30_output_0_quantized', [600,600], TensorProto.INT8),
    OutputDebug('/Constant_31_output_0_quantized', [600], TensorProto.INT8),
    OutputDebug('/Constant_32_output_0_quantized', [257,600], TensorProto.INT8),
    OutputDebug('/Constant_33_output_0_quantized', [257], TensorProto.INT8),
    OutputDebug('/Squeeze_1_output_0_quantized', [400], TensorProto.INT8),
    OutputDebug('/Squeeze_2_output_0_quantized', [400], TensorProto.INT8),
    OutputDebug('/Squeeze_output_0_quantized', [257], TensorProto.INT8),
    OutputDebug('/MatMul_2_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/MatMul_4_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/MatMul_6_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/MatMul_8_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/MatMul_10_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/MatMul_12_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/MatMul_output_0_quantized', [400], TensorProto.INT8),
    OutputDebug('/Add_2_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Add_4_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Add_6_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Add_12_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Add_14_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Add_16_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Add_output_0_quantized', [400], TensorProto.INT8),
    OutputDebug('/MatMul_1_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/MatMul_3_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/MatMul_5_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Add_1_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Add_3_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Add_5_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Add_7_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Add_8_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Sigmoid_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Sigmoid_1_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Mul_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Sigmoid_1_output_0', [1,400], TensorProto.INT8),
    OutputDebug('/Mul_2_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Add_9_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Sub_output_0', [1,400], TensorProto.FLOAT),
    OutputDebug('/Add_9_output_0', [1,400], TensorProto.INT8),
    OutputDebug('/Sub_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Tanh_output_0', [1,400], TensorProto.FLOAT),
    OutputDebug('/Tanh_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Mul_1_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Add_10_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('h1n_quantized', [400], TensorProto.INT8),
    OutputDebug('/MatMul_7_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/MatMul_9_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/MatMul_11_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('h1n', [1,400], TensorProto.FLOAT),
    OutputDebug('/Add_11_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Add_13_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Add_15_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Add_17_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Add_18_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Sigmoid_2_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Sigmoid_3_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Mul_3_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Sigmoid_3_output_0', [1,400], TensorProto.INT8),
    OutputDebug('/Mul_5_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Add_19_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Sub_1_output_0', [1,400], TensorProto.FLOAT),
    OutputDebug('/Add_19_output_0', [1,400], TensorProto.INT8),
    OutputDebug('/Sub_1_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Tanh_1_output_0', [1,400], TensorProto.FLOAT),
    OutputDebug('/Tanh_1_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Mul_4_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('/Add_20_output_0_quantized', [1,400], TensorProto.INT8),
    OutputDebug('h2n_quantized', [400], TensorProto.INT8),
    OutputDebug('/MatMul_13_output_0_quantized', [600], TensorProto.INT8),
    OutputDebug('h2n', [1,400], TensorProto.FLOAT),
    OutputDebug('/Add_21_output_0_quantized', [600], TensorProto.INT8),
    OutputDebug('/MatMul_14_output_0_quantized', [600], TensorProto.INT8),
    OutputDebug('/Add_22_output_0_quantized', [600], TensorProto.INT8),
    OutputDebug('/MatMul_15_output_0_quantized', [257], TensorProto.INT8),
    OutputDebug('/Add_23_output_0_quantized', [257], TensorProto.INT8),
    OutputDebug('mask_pred_quantized', [257], TensorProto.INT8),
    OutputDebug('mask_pred', [1,257], TensorProto.FLOAT)
]

for output in outputs:
    model_path = "nsnet2_reimplemented_int8_static.onnx"
    model = onnx.load(model_path)
    graph = model.graph
    
    found = False
    for value_info in graph.value_info:
        if value_info.name == output.name:
            graph.output.append(value_info)
            found = True
            print(f"Added '{output.name}' as output.")
            break

    if not found:
        print(f"'{output.name}' not found. Manual creation of value_info.")
        new_value_info = helper.make_tensor_value_info(
            output.name, 
            output.type,           
            output.shape            
        )
        graph.output.append(new_value_info)
        print(f"Added manual '{output.name}' as output.")

    debug_model_path = "debug_models/" + adjust_name(output.name) + ".onnx"
    onnx.save(model, debug_model_path)
    print(f"Modello modificato salvato in: {debug_model_path}")
