import onnxruntime as ort
import numpy as np
import os

debug_models_folder = 'debug_models'
file_names = os.listdir(debug_models_folder)

for model_name in file_names:
    model_path = "debug_models/" + model_name
    session = ort.InferenceSession(model_path)

    input_names = [inp.name for inp in session.get_inputs()]
    print("Inputs:", input_names)

    output_names = [out.name for out in session.get_outputs()]
    print("Outputs:", output_names)

    in_noisy = np.ones((1, 1, 257), dtype=np.float32) 
    h1 = np.zeros((1, 1, 400), dtype=np.float32)
    h2 = np.zeros((1, 1, 400), dtype=np.float32)

    inputs = {
        "in_noisy": in_noisy,
        "h1": h1,
        "h2": h2,
    }

    outputs = session.run(None, inputs)

    outs = []
    names = []
    for name, output in zip(output_names, outputs):
        outs.append(output)
        names.append(name)
    
    with open(f"debug_outputs_expected/{names[-1]}.txt", "w") as file:
        tensor = ''
        curr_out = outs[-1].flatten()
        for index, val in enumerate(curr_out):
            tensor += str(val)
            if index < len(curr_out)-1:
                tensor += ','
        file.write(tensor)

