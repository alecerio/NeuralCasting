import onnx
from onnx import numpy_helper
from collections import defaultdict, deque

shapes = {
    'h1': 400,
    'h2': 400,
    'in_noisy': 257
}

def generate_print_debug(name, shape, type):
    return f"NCAST_PRINT_MAT({adjust_name(name)}, 1, {shape}, {type}, \"{adjust_name(name)}\")\n"

def generate_file(filename, code):
    try:
        with open(filename, 'w') as file:
            file.write(code)
    except Exception as e:
        print(f"Error file generation: {e}")

def get_optype(graph, node):
    for value_info in graph.value_info:
        if value_info.name == node.output[0]:
            output_type = value_info.type.tensor_type.elem_type
            if output_type == 1:
                type_code = "float32_t"
            else:
                raise Exception(f"data type not supported: {type(output_type)}")
            return [output_type, type_code]
    return [1, "int8_t"]

def ncast_debug(output_name, shape_x, shape_y, output_type):
    if output_type == 1:
        type_debug = "%f"
    else:
        raise Exception(f"data type not supported: {output_type}")
    print("#ifdef NCAST_DEBUG")
    print(f"NCAST_PRINT_MAT({adjust_name(output_name)}, {shape_x}, {shape_y}, {type_debug}, \"{adjust_name(output_name)}\")")
    print("#endif")

def get_initializer_value(graph, tensor_name):
    for initializer in graph.initializer:
        if initializer.name == tensor_name:
            return numpy_helper.to_array(initializer)
    return None

def get_args(node):
    ins = ''
    for input_name in node.input:
        input_name = adjust_name(input_name)
        ins += input_name + ', '
    outs = ''
    for output_name in node.output:
        output_name = adjust_name(output_name)
        outs += output_name + ', '
    args = ins + outs
    if args != '':
        args = args[:-2]
    return args

def adjust_name(name):
    if name[0] == '/':
        name = name[1:]
    if name[0] == '1' or name[0] == '2':
        name = '_' + name
    return name

def generate_constants(model_path, file_h):
    model = onnx.load(model_path)
    graph = model.graph

    for idx, node in enumerate(graph.node):
        if node.op_type == 'Constant':
            name = str(node.output[0])[1:]
            dims = ''
            for dim in node.attribute[0].t.dims:
                dims += (str(dim) + '*')
            dims = dims[:-1]
            if dims == '':
                dims = '1'
            file_h += "static float32_t " + name + "[" + dims + "] = {\n"

            shapes[name] = dims

            for attr in node.attribute:
                if attr.name == "value":
                    constant_value = onnx.numpy_helper.to_array(attr.t)
                    constant_value = constant_value.flatten()
                    for val  in constant_value:
                        file_h += str(val) + ", "
                    break
            
            file_h += "};\n"
    return file_h

def generate_inference(model_path, file_c, file_h, is_debug=True):
    model = onnx.load(model_path)
    graph = model.graph

    for idx, node in enumerate(graph.node):
        if node.op_type == 'QuantizeLinear':
            zero_point_tensor = node.input[2] if len(node.input) > 2 else None
            scale = get_initializer_value(graph, node.input[1])
            scale = int(scale * 2**16)
            zero_point = get_initializer_value(graph, zero_point_tensor) if zero_point_tensor else 0
            args = get_args(node)
            if adjust_name(node.input[0]) in shapes.keys():
                shape = shapes[adjust_name(node.input[0])]
            file_h += f"static int32_t {adjust_name(node.input[1])} = {scale};\n"
            file_h += f"static int8_t {adjust_name(node.input[2])} = {zero_point};\n"
            file_h += f"static int8_t {adjust_name(node.output[0])}[{shape}];\n"
            file_c += f"QLINEAR({args}, {shape})\n"
            if adjust_name(node.input[0]) in shapes.keys():
                shapes[adjust_name(node.output[0])] = shapes[adjust_name(node.input[0])]
            else:
                print(f"####: {adjust_name(node.input[0])} -> {adjust_name(node.output[0])}")
                print(shapes.keys())
            if is_debug:
                file_c += generate_print_debug(node.output[0], shape, "%d")
        
        elif node.op_type == 'QLinearMatMul':
            shape_a = shapes[adjust_name(node.input[0])]
            shape_a_split = str(shape_a).split('*')
            if(len(shape_a_split) >= 2):
                dim0 = shape_a_split[-2]
                dimi = shape_a_split[-1]
            else:
                dim0 = '1'
                dimi = shape_a_split[0]

            shape_b = shapes[adjust_name(node.input[3])]
            shape_b_split = str(shape_b).split('*')
            if(len(shape_b_split) >= 2):
                dim1 = shape_b_split[-1]
            else:
                dim1 = '1'
            
            if dim0 == '' and dim1 != '':
                out_shape = dim1
            elif dim0 != '' and dim1 == '':
                out_shape = dim0
            elif dim0 == '' and dim1 == '':
                out_shape = '1'
            else:
                out_shape = dim0 + '*' + dim1

            zero_point_tensor = node.input[7]
            zero_point = get_initializer_value(graph, zero_point_tensor) if zero_point_tensor else 0
            scale = get_initializer_value(graph, node.input[6])
            scale = int(scale * 2**16)
            file_h += f"static int32_t {adjust_name(node.input[6])} = {scale};\n"
            file_h += f"static int8_t {adjust_name(node.input[7])} = {zero_point};\n"
            file_h += f"static int8_t {adjust_name(node.output[0])}[{out_shape}];\n"
            args = get_args(node)
            file_c += f"NCAST_QMATMUL_FIX({args}, {dim0}, {dimi}, {dim1})\n"
            if adjust_name(node.input[0]) in shapes.keys():
                shapes[adjust_name(node.output[0])] = out_shape
            else:
                print(f"####: {adjust_name(node.input[0])} -> {adjust_name(node.output[0])}")
                print(shapes.keys())
            if is_debug:
                file_c += generate_print_debug(node.output[0], out_shape, "%d")
        
        elif node.op_type == 'QLinearAdd':
            zero_point_tensor = node.input[7]
            zero_point = get_initializer_value(graph, zero_point_tensor) if zero_point_tensor else 0
            scale = get_initializer_value(graph, node.input[6])
            scale = int(scale * 2**16)
            add_shape = shapes[adjust_name(node.input[0])]
            file_h += f"static int32_t {adjust_name(node.input[6])} = {scale};\n"
            file_h += f"static int8_t {adjust_name(node.input[7])} = {zero_point};\n"
            file_h += f"static int8_t {adjust_name(node.output[0])}[{add_shape}];\n"
            args = get_args(node)
            file_c += f"NCAST_QADD_FIXED({args}, {add_shape})\n"
            if adjust_name(node.input[0]) in shapes.keys():
                shapes[adjust_name(node.output[0])] = add_shape
            if is_debug:
                file_c += generate_print_debug(node.output[0], add_shape, "%d")
        
        elif node.op_type == 'QLinearSigmoid':
            if adjust_name(node.input[0]) in shapes.keys():
                shape = shapes[adjust_name(node.input[0])]
            zero_point_tensor = node.input[4]
            zero_point = get_initializer_value(graph, zero_point_tensor) if zero_point_tensor else 0
            scale = get_initializer_value(graph, node.input[3])
            scale = int(scale * 2**16)
            file_h += f"static int32_t {adjust_name(node.input[3])} = {scale};\n"
            file_h += f"static int8_t {adjust_name(node.input[4])} = {zero_point};\n"
            file_h += f"static int8_t {adjust_name(node.output[0])}[{shape}];\n"
            args = get_args(node)
            file_c += f"NCAST_QSIGMOID({args}, {shape})\n"
            if adjust_name(node.input[0]) in shapes.keys():
                shapes[adjust_name(node.output[0])] = shapes[adjust_name(node.input[0])]
            if is_debug:
                file_c += generate_print_debug(node.output[0], shape, "%d")
        
        elif node.op_type == 'QLinearMul':
            zero_point_tensor = node.input[7]
            zero_point = get_initializer_value(graph, zero_point_tensor) if zero_point_tensor else 0
            scale = get_initializer_value(graph, node.input[6])
            scale = int(scale * 2**16)
            file_h += f"static int32_t {adjust_name(node.input[6])} = {scale};\n"
            file_h += f"static int8_t {adjust_name(node.input[7])} = {zero_point};\n"
            file_h += f"static int8_t {adjust_name(node.output[0])}[{shape}];\n"
            args = get_args(node)
            mul_shape = shapes[adjust_name(node.input[0])]
            file_c += f"NCAST_QMUL_FIX({args}, {mul_shape})\n"
            if adjust_name(node.input[0]) in shapes.keys():
                shapes[adjust_name(node.output[0])] = shapes[adjust_name(node.input[0])]
            if is_debug:
                file_c += generate_print_debug(node.output[0], mul_shape, "%d")
        
        elif node.op_type == 'DequantizeLinear':
            shape = shapes[adjust_name(node.input[0])]
            file_h += f"static float32_t {adjust_name(node.output[0])}[{shape}];\n"
            args = get_args(node)
            file_c += f"NCAST_DQLINEAR({args}, {shape})\n"
            if adjust_name(node.input[0]) in shapes.keys():
                shapes[adjust_name(node.output[0])] = shapes[adjust_name(node.input[0])]
            if is_debug:
                file_c += generate_print_debug(node.output[0], shape, "%f")
        
        elif node.op_type == 'Sub':
            shape = shapes[adjust_name(node.input[1])]
            file_h += f"static float32_t {adjust_name(node.output[0])}[{shape}];\n"
            args = get_args(node)
            file_c += f"NCAST_SUB({args}, {shape})\n"
            if adjust_name(node.input[1]) in shapes.keys():
                shapes[adjust_name(node.output[0])] = shapes[adjust_name(node.input[1])]
            if is_debug:
                file_c += generate_print_debug(node.output[0], shape, "%f")
        
        elif node.op_type == 'Tanh':
            shape = shapes[adjust_name(node.input[0])]
            file_h += f"static float32_t {adjust_name(node.output[0])}[{shape}];\n"
            args = get_args(node)
            file_c += f"NCAST_TANH({args}, {shape})\n"
            if adjust_name(node.input[0]) in shapes.keys():
                shapes[adjust_name(node.output[0])] = shapes[adjust_name(node.input[0])]
            if is_debug:
                file_c += generate_print_debug(node.output[0], shape, "%f")
        
        elif node.op_type == 'Squeeze':
            [output_type, type_code] = get_optype(graph, node)
            for value_info in graph.value_info:
                if value_info.name == node.output[0]:
                    output_type = value_info.type.tensor_type.elem_type
                    if output_type == 1:
                        type_code = "float32_t"
                    else:
                        raise Exception(f"data type not supported: {type(output_type)}")
                    break
            if adjust_name(node.input[0]) in shapes.keys():
                shape = shapes[adjust_name(node.input[0])]
            file_h += f"static {type_code} {adjust_name(node.output[0])}[{shape}];\n"
            args = get_args(node)
            file_c += f"SQUEEZE({args}, {shape})\n"
            if adjust_name(node.input[0]) in shapes.keys():
                shapes[adjust_name(node.output[0])] = shapes[adjust_name(node.input[0])]
            else:
                print(f"!!!!: {adjust_name(node.input[0])} -> {adjust_name(node.output[0])}")
                print(shapes.keys())
            if is_debug:
                file_c += generate_print_debug(node.output[0], shape, "%f")
            
    return [file_c, file_h]

def analyze_onnx_graph(model_path):
    model = onnx.load(model_path)
    graph = model.graph

    for idx, node in enumerate(graph.node):
        
        if node.op_type == 'QuantizeLinear':
            zero_point_tensor = node.input[2] if len(node.input) > 2 else None
            scale = get_initializer_value(graph, node.input[1])
            scale = int(scale * 2**16)
            zero_point = get_initializer_value(graph, zero_point_tensor) if zero_point_tensor else 0
            args = get_args(node)
            if adjust_name(node.input[0]) in shapes.keys():
                shape = shapes[adjust_name(node.input[0])]
            print(f"int32_t {adjust_name(node.input[1])} = {scale};")
            print(f"int8_t {adjust_name(node.input[2])} = {zero_point};")
            print(f"int8_t {adjust_name(node.output[0])}[{shape}];")
            print(f"QLINEAR({args}, {shape})")
            if adjust_name(node.input[0]) in shapes.keys():
                shapes[adjust_name(node.output[0])] = shapes[adjust_name(node.input[0])]
            else:
                print(f"####: {adjust_name(node.input[0])} -> {adjust_name(node.output[0])}")
                print(shapes.keys())
        
        elif node.op_type == 'QLinearMatMul':
            shape_a = shapes[adjust_name(node.input[0])]
            shape_a_split = str(shape_a).split('*')
            if(len(shape_a_split) >= 2):
                dim0 = shape_a_split[-2]
                dimi = shape_a_split[-1]
            else:
                dim0 = '1'
                dimi = shape_a_split[0]

            shape_b = shapes[adjust_name(node.input[3])]
            shape_b_split = str(shape_b).split('*')
            if(len(shape_b_split) >= 2):
                dim1 = shape_b_split[-1]
            else:
                dim1 = '1'
            
            if dim0 == '' and dim1 != '':
                out_shape = dim1
            elif dim0 != '' and dim1 == '':
                out_shape = dim0
            elif dim0 == '' and dim1 == '':
                out_shape = '1'
            else:
                out_shape = dim0 + '*' + dim1

            zero_point_tensor = node.input[7]
            zero_point = get_initializer_value(graph, zero_point_tensor) if zero_point_tensor else 0
            scale = get_initializer_value(graph, node.input[6])
            scale = int(scale * 2**16)
            print(f"int32_t {adjust_name(node.input[7])} = {scale};")
            print(f"int8_t {adjust_name(node.input[6])} = {zero_point};")
            print(f"int8_t {adjust_name(node.output[0])}[{out_shape}];")
            args = get_args(node)
            print(f"NCAST_QMATMUL_FIX({args}, {dim0}, {dimi}, {dim1})")
            if adjust_name(node.input[0]) in shapes.keys():
                shapes[adjust_name(node.output[0])] = out_shape
            else:
                print(f"####: {adjust_name(node.input[0])} -> {adjust_name(node.output[0])}")
                print(shapes.keys())
        
        elif node.op_type == 'QLinearAdd':
            zero_point_tensor = node.input[7]
            zero_point = get_initializer_value(graph, zero_point_tensor) if zero_point_tensor else 0
            scale = get_initializer_value(graph, node.input[6])
            scale = int(scale * 2**16)
            add_shape = shapes[adjust_name(node.input[0])]
            print(f"int32_t {adjust_name(node.input[7])} = {scale};")
            print(f"int8_t {adjust_name(node.input[6])} = {zero_point};")
            print(f"int8_t {adjust_name(node.output[0])}[{add_shape}];")
            args = get_args(node)
            print(f"NCAST_QADD_FIXED({args}, {add_shape})")
            if adjust_name(node.input[0]) in shapes.keys():
                shapes[adjust_name(node.output[0])] = add_shape
        
        elif node.op_type == 'QLinearSigmoid':
            if adjust_name(node.input[0]) in shapes.keys():
                shape = shapes[adjust_name(node.input[0])]
            zero_point_tensor = node.input[4]
            zero_point = get_initializer_value(graph, zero_point_tensor) if zero_point_tensor else 0
            scale = get_initializer_value(graph, node.input[3])
            scale = int(scale * 2**16)
            print(f"int32_t {adjust_name(node.input[3])} = {scale};")
            print(f"int8_t {adjust_name(node.input[4])} = {zero_point};")
            print(f"int8_t {adjust_name(node.output[0])}[{shape}];")
            args = get_args(node)
            print(f"NCAST_QSIGMOID({args}, {shape})")
            if adjust_name(node.input[0]) in shapes.keys():
                shapes[adjust_name(node.output[0])] = shapes[adjust_name(node.input[0])]
        
        elif node.op_type == 'QLinearMul':
            zero_point_tensor = node.input[7]
            zero_point = get_initializer_value(graph, zero_point_tensor) if zero_point_tensor else 0
            scale = get_initializer_value(graph, node.input[6])
            scale = int(scale * 2**16)
            print(f"int32_t {adjust_name(node.input[7])} = {scale};")
            print(f"int8_t {adjust_name(node.input[6])} = {zero_point};")
            print(f"int8_t {adjust_name(node.output[0])}[{shape}];")
            args = get_args(node)
            mul_shape = shapes[adjust_name(node.input[0])]
            print(f"NCAST_QMUL_FIX({args}, {mul_shape})")
            if adjust_name(node.input[0]) in shapes.keys():
                shapes[adjust_name(node.output[0])] = shapes[adjust_name(node.input[0])]
        
        elif node.op_type == 'DequantizeLinear':
            shape = shapes[adjust_name(node.input[0])]
            print(f"float32_t {adjust_name(node.output[0])}[{shape}];")
            args = get_args(node)
            print(f"NCAST_DQLINEAR({args}, {shape})")
            if adjust_name(node.input[0]) in shapes.keys():
                shapes[adjust_name(node.output[0])] = shapes[adjust_name(node.input[0])]
        
        elif node.op_type == 'Sub':
            shape = shapes[adjust_name(node.input[1])]
            print(f"int8_t {adjust_name(node.output[0])}[{shape}];")
            args = get_args(node)
            print(f"NCAST_SUB({args}, {shape})")
            if adjust_name(node.input[1]) in shapes.keys():
                shapes[adjust_name(node.output[0])] = shapes[adjust_name(node.input[1])]
        
        elif node.op_type == 'Tanh':
            shape = shapes[adjust_name(node.input[0])]
            print(f"int8_t {adjust_name(node.output[0])}[{shape}];")
            args = get_args(node)
            print(f"NCAST_TANH({args}, {shape})")
            if adjust_name(node.input[0]) in shapes.keys():
                shapes[adjust_name(node.output[0])] = shapes[adjust_name(node.input[0])]
        
        elif node.op_type == 'Squeeze':
            [output_type, type_code] = get_optype(graph, node)
            for value_info in graph.value_info:
                if value_info.name == node.output[0]:
                    output_type = value_info.type.tensor_type.elem_type
                    if output_type == 1:
                        type_code = "float32_t"
                    else:
                        raise Exception(f"data type not supported: {type(output_type)}")
                    break
            if adjust_name(node.input[0]) in shapes.keys():
                shape = shapes[adjust_name(node.input[0])]
            print(f"{type_code} {adjust_name(node.output[0])}[{shape}];")
            args = get_args(node)
            print(f"SQUEEZE({args}, {shape})")
            if adjust_name(node.input[0]) in shapes.keys():
                shapes[adjust_name(node.output[0])] = shapes[adjust_name(node.input[0])]
            else:
                print(f"!!!!: {adjust_name(node.input[0])} -> {adjust_name(node.output[0])}")
                print(shapes.keys())
            

def ncgencode(name, onnx_path, output_path, debug):

    model_path = onnx_path
    file_c = ""
    file_h = ""
    nn_name = name

    file_h += "#ifndef __" + nn_name.upper() + "__\n"
    file_h += "#define __" + nn_name.upper() + "__\n"
    file_h += "#include <stdio.h>\n"
    file_h += "#include \"quant.h\"\n"
    file_h += "#include \"utils.h\"\n"
    file_h += "#include \"squeeze.h\"\n"
    file_h += "#include \"qlinear.h\"\n"
    file_h += "#include \"qmatmul.h\"\n"
    file_h += "#include \"qadd.h\"\n"
    file_h += "#include \"qsigmoid.h\"\n"
    file_h += "#include \"qmul.h\"\n"
    file_h += "#include \"dqlinear.h\"\n"
    file_h += "#include \"sub.h\"\n"
    file_h += "#include \"tanh.h\"\n"
    file_h += "void run_inference(float32_t* in_noisy, float32_t* h1, float32_t* h2);\n"
    file_h = generate_constants(model_path, file_h)

    file_c += "#include \"" + nn_name + ".h\"\n"
    file_c += "void run_inference(float32_t* in_noisy, float32_t* h1, float32_t* h2) {\n"
    [file_c, file_h] = generate_inference(model_path, file_c, file_h, debug)
    file_c += "}\n"
    generate_file(output_path + "/" + nn_name + ".c", file_c)

    file_h += "#endif // __"  + nn_name.upper() + "__\n"
    generate_file(output_path + "/" + nn_name + ".h", file_h)

