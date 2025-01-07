import onnx
from onnx import helper, TensorProto

model_path = "nsnet2_reimplemented_int8_static.onnx"
model = onnx.load(model_path)

graph = model.graph
for node in graph.node:
    for output in node.output:
        intermediate_output = helper.make_tensor_value_info(output, TensorProto.FLOAT, None)
        graph.output.append(intermediate_output)

debug_model_path = "nsnet2_reimplemented_int8_static_debug.onnx"
onnx.save(model, debug_model_path)
print(f"Modello modificato salvato in: {debug_model_path}")
