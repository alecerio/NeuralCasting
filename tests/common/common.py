import onnxruntime as ort
import numpy as np
import json
import torch

def inference_onnx_runtime(path_onnx, input_data):
    session = ort.InferenceSession(path_onnx)
    
    name_dict = {}
    for i in range(len(input_data)):
        input_name = session.get_inputs()[i].name
        name_dict[input_name] = input_data[i]
    
    outputs = session.run(None, name_dict)

    n_outputs = len(outputs)
    outputs_onnx = []
    outputs_shape_onnx = []
    for i in range(n_outputs):
        output_onnx = outputs[i]
        output_shape_onnx = output_onnx.shape
        output_onnx = np.squeeze(output_onnx)
        output_onnx = output_onnx.flatten()
        outputs_onnx.append(output_onnx)
        outputs_shape_onnx.append(output_shape_onnx)
    return [outputs_onnx, outputs_shape_onnx]

def create_main_c(test_path, output_path, name, main_name='main.c'):
    # read main.c code and add include to nn
    f = open(test_path + main_name, 'r')
    main_code : str = "#include \"" + name + ".h\"\n"
    main_code += f.read()
    f.close()

    # generate main.c in output directory
    f = open(output_path + 'main.c', 'w')
    f.write(main_code)
    f.close()

def read_inferred_output_shape(temp_path):
    output_shape_path : str = temp_path + "out_shape.json"
    with open(output_shape_path, 'r') as json_file:
        data = json.load(json_file)
        output_keys = list(data.keys())
    output_shape_c = data[output_keys[0]]
    return output_shape_c

def read_output_c(output_path):
    f = open(output_path + "test_output.txt")
    output_text : str = f.read()
    f.close()
    output_values_str : list[str] = output_text.split(" ")
    output_c : list[float] = []
    for i in range(len(output_values_str)):
        output_c.append(float(output_values_str[i]))
    return output_c

def print_test_header(test_name : str, n_spaces : int = 0, n_newlines : int = 2):
    print("\n" * n_newlines)
    print("####################################################################################")
    print("\t"*n_spaces + test_name)
    print("####################################################################################")
    print("\n" * n_newlines)

def inference_pytorch(model, input_data):
    output_pytorch = model(input_data)
    output_shape_pytorch = output_pytorch.shape
    output_pytorch = torch.squeeze(output_pytorch)
    return [output_pytorch, output_shape_pytorch]

def compare_shape(test, shape1, shape2, label1, label2, verbose=True):
    if verbose:
        print("Output shape ", label1, ": ", shape1)
        print("Output shape ", label2, ": ", shape2)
    test.assertEqual(len(shape1), len(shape2))
    for i in range(len(shape1)):
        test.assertEquals(shape1[i], shape2[i])

def compare_results(test, output1, output2, label1, label2, delta, verbose=True):
    if verbose:
        print("Output ", label1, ": ", output1)
        print("Output ", label2, ": ", output2)
    test.assertEqual(len(output1), len(output2))
    N : int = len(output1)
    for i in range(N):
        test.assertAlmostEqual(output1[i], output2[i], delta=delta)