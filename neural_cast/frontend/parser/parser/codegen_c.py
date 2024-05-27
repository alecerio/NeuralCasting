from neural_cast.frontend.common.common import CompilerLogger, CompilerConfig
from neural_cast.frontend.exceptions.CompilerException import CompilerException
from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.init_node import InitializerNode
from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.parser.node.input_node import InputNode
from neural_cast.frontend.parser.node.output_node import OutputNode
from neural_cast.frontend.parser.node_types.node_type import NodeType
from neural_cast.frontend.parser.node_types.tensor_type import TensorType
from neural_cast.frontend.common.common import fix_identifier, onnx_tensor_elem_type_to_c_dictionary, onnx_type_to_python_struct_type
import struct

def pre_codegen_c(nodes : list[Node]) -> [str, str]:
    alloc_type : str = CompilerConfig()['alloc']
    parallel : str = CompilerConfig()['parallel']

    _gen_binary_matrices(nodes)

    header_file_code : str = ""
    code_generated : str = ""

    # generate include code
    header_file_code += _gen_include_code_c(nodes, parallel)

    # generate define code
    header_file_code += _gen_define_code_c()
    
    #add readmat macro
    if alloc_type == 'heap':
        header_file_code += _gen_readmat_macro()
    
    # add benchmark macro
    #header_file_code += _gen_benchmark_macro()

    # generate file header
    code_generated += _gen_header_code_c()

    # generate include in source file
    code_generated += _gen_includes_in_source_c()

    # declaration matrices (heap)
    if alloc_type == 'heap':
        code_generated += _gen_declarations_matrices(nodes)

    # allocnn definition
    if alloc_type == 'heap':
        code_generated += _gen_allocnn_definition(nodes)

    # freenn definition
    if alloc_type == 'heap':
        code_generated += _gen_freenn_definition(nodes)

    # generate declarations
    if alloc_type == 'data':
        code_generated += _gen_declarations_c(nodes)
    elif alloc_type == 'heap':
        header_file_code += _gen_declaration_allocnn()
        header_file_code += _gen_declaration_freenn()
    else:
        CompilerException("Error: unknown memory type allocation")
    
    # generate function header code
    function_header : str = _gen_function_header_code_c(nodes)
    code_generated += function_header
    header_file_code += function_header[:-3] + ";\n"

    if parallel == 'omp':
        code_generated += _gen_omp_setup()

    code_generated += _gen_benchmark_setup()

    return [header_file_code, code_generated]

def post_codegen_c() -> str:
    return "}"

def _gen_include_code_c(nodes :list[Node], parallel: str) -> str:
    CompilerLogger().info("Generate include code")

    code_generated : str = ""
    list_includes : list[str] = []
    
    if parallel == 'omp':
        list_includes.append("#include <omp.h>")

    for node in nodes:
        if isinstance(node, OpNode):
            CompilerLogger().info("Generate include code for: " + node.get_name())
            inc_code : str = node.generate_includes_code_c()
            inc_lines : list[str] = inc_code.split('\n')
            filtered_inc_lines : list[str] = [line for line in inc_lines if line.startswith('#include')]
            for line in filtered_inc_lines:
                list_includes.append(line) 
    list_includes = list(set(list_includes))

    stdint_inc : str = "#include <stdint.h>"
    list_includes.append(stdint_inc)

    stdlib_inc : str = "#include <stdlib.h>"
    list_includes.append(stdlib_inc)

    stdio_inc : str = "#include <stdio.h>"
    list_includes.append(stdio_inc)

    code_generated += "// INCLUDE\n\n"
    for inc in list_includes:
        code_generated += inc + "\n"
    code_generated += "\n\n"

    code_generated += "typedef float float32_t;\n\n"

    return code_generated

def _gen_header_code_c() -> str:
    CompilerLogger().info("Generate file header code")
    import datetime
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    message : str = "// *****************************************************************************\n"
    message += "// \tTHIS CODE WAS AUTOMATICALLY GENERATED ON " + formatted_datetime + "\n"
    message  += "// *****************************************************************************\n\n"
    return message

def _gen_includes_in_source_c() -> str:
    code_gen : str = "#include \"" + CompilerConfig()['name'] + ".h\"\n\n"
    return code_gen

def _gen_declarations_c(nodes : list[Node]) -> str:
    CompilerLogger().info("Generate declaration C code")
    code_generated : str = ""
    for node in nodes:
        if isinstance(node, OpNode) or isinstance(node, InitializerNode):
            CompilerLogger().info("Generate declaration code C for: " + node.get_name())
            code_generated += node.generate_declaration_code_c()
    return code_generated

def _gen_function_header_code_c(nodes : list[Node]) -> str:
    CompilerLogger().info("Generate function header")

    header_code : str = "void run_inference("
    params_list : list[str] = []
        
    for node in nodes:
        if isinstance(node, InputNode):
            node_type : NodeType = node.get_node_type()
            if isinstance(node_type, TensorType):
                name : str = "tensor_" + fix_identifier(node.get_name())
                param :str = onnx_tensor_elem_type_to_c_dictionary(node_type.get_elem_type()) + " " + name
                params_list.append(param)
            else:
                raise CompilerException("Error: input tensor not supported")
        
    for node in nodes:
        if isinstance(node, OutputNode):
            node_type : NodeType = node.get_node_type()
            if isinstance(node_type, TensorType):
                name : str = "tensor_" + fix_identifier(node.get_name())
                param : str = onnx_tensor_elem_type_to_c_dictionary(node_type.get_elem_type()) + " " + name
                params_list.append(param)
            else:
                raise CompilerException("Error: input tensor not supported")

    n_params : int = len(params_list)

    for i in range(n_params):
        header_code += params_list[i]
        if i < n_params-1: header_code += ", "
    header_code += ") {\n"
        
    return header_code

def _gen_declarations_matrices(nodes : list[Node]) -> str:
    code : str = "// matrices delaration\n"
    for node in nodes:
        mat_type : str = onnx_tensor_elem_type_to_c_dictionary(node.infer_output_type())
        if isinstance(node, InitializerNode):
            code += 'static ' + mat_type + " tensor_" + fix_identifier(node.get_name()) + ";\n"
    code += "\n"
    return code

def _gen_declaration_allocnn() -> str:
    code : str = "// allocnn\n"
    code += "void allocnn();\n"
    code += "\n"
    return code

def _gen_declaration_freenn() -> str:
    code : str = "// freenn\n"
    code += "void freenn();\n"
    code += "\n"
    return code

def _gen_readmat_macro() -> str:
    codegen_c_path : str = CompilerConfig()['codegen_c_path']
    f = open(codegen_c_path + '/READMAT.c')
    macro : str = f.read()
    macro = macro + '\n\n'
    f.close()
    return macro

def _gen_benchmark_macro() -> str:
    codegen_c_path : str = CompilerConfig()['codegen_c_path']
    f = open(codegen_c_path + '/Benchmark.h')
    macro : str = f.read()
    macro = macro + '\n\n'
    f.close()
    return macro

def _gen_allocnn_definition(nodes : list[Node]) -> str:
    code : str = ''
    code += 'void allocnn() {\n'
    code += 'FILE *fp;\n'
    for node in nodes:
        if isinstance(node, InitializerNode):
            tensor_type : str = onnx_tensor_elem_type_to_c_dictionary(node.get_data_type())[:-1]
            tensor_shape : int = node.get_tensor().shape
            tensor_size : int = 1
            for dim in tensor_shape:
                tensor_size = tensor_size * dim
            name : str = fix_identifier(node.get_name())
            tensor_name : str = 'tensor_' + name
            code += 'READMAT(' + tensor_name + ', ' + str(tensor_size) + ', \"' + name + '.bin\", ' + tensor_type + ')\n'
    code += '}\n\n'
    return code

def _gen_freenn_definition(nodes: list[Node]) -> str:
    code : str = ''
    code += 'void freenn() {\n'
    for node in nodes:
        if isinstance(node, InitializerNode):
            name : str = fix_identifier(node.get_name())
            tensor_name : str = 'tensor_' + name
            code += 'free(' + tensor_name + ');\n'
    code += '}\n\n'
    return code

def _gen_binary_matrices(nodes : list[Node]):
    output_path : str = CompilerConfig()['output_path']
    for node in nodes:
        if isinstance(node, InitializerNode):
            filename : str = fix_identifier(node.get_name())
            with open(output_path + '/' + filename + '.bin', 'wb') as f:
                tensor = node.get_tensor()
                tensor = tensor.flatten()
                tensor_type = onnx_type_to_python_struct_type(node.get_data_type())
                for val in tensor:
                    f.write(struct.pack(tensor_type, val))

def _gen_define_code_c():
    code : str = "// MACROS\n\n"
    
    parallel : str = CompilerConfig()['parallel']

    if parallel == 'omp':
        num_threads = CompilerConfig()['num_threads']
        code += "#define NUM_THREADS (" + str(num_threads) + ")\n"

    code += "\n\n"

    return code

def _gen_omp_setup() -> str:
    code : str = ""
    num_threads = CompilerConfig()['num_threads']
    code += "omp_set_num_threads(" + str(num_threads) + ");\n\n"
    return code

def _gen_benchmark_setup() -> str:
    return "#ifdef COMPILER_BENCHMARK\n" + \
        "double neuralcasting_time_benchmark = 0.0f;\n" + \
        "double neuralcasting_end_benchmark = 0.0f;\n" + \
        "double neuralcasting_start_benchmark = 0.0f;\n" + \
        "#endif\n\n"