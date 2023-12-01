from neural_cast.frontend.common.common import CompilerLogger, CompilerConfig
from neural_cast.frontend.exceptions.CompilerException import CompilerException
from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.init_node import InitializerNode
from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.parser.node.input_node import InputNode
from neural_cast.frontend.parser.node.output_node import OutputNode
from neural_cast.frontend.parser.node_types.node_type import NodeType
from neural_cast.frontend.parser.node_types.tensor_type import TensorType
from neural_cast.frontend.common.common import fix_identifier, onnx_tensor_elem_type_to_c_dictionary

def pre_codegen_c(nodes : list[Node]) -> [str, str]:
    header_file_code : str = ""
    code_generated : str = ""

    # generate include code
    header_file_code += _gen_include_code_c(nodes)
        
    # generate file header
    code_generated += _gen_header_code_c()

    # generate include in source file
    code_generated += _gen_includes_in_source_c()

    # generate declarations
    code_generated += _gen_declarations_c(nodes)

    # generate function header code
    function_header : str = _gen_function_header_code_c(nodes)
    code_generated += function_header
    header_file_code += function_header[:-3] + ";\n"

    return [header_file_code, code_generated]

def post_codegen_c() -> str:
    return "}"

def _gen_include_code_c(nodes :list[Node]) -> str:
    CompilerLogger().info("Generate include code")

    code_generated : str = ""
    list_includes : list[str] = []
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