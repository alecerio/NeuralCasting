import logging

def is_valid_onnx_data_type(data_type_index : int):
    if data_type_index < 0 or data_type_index > 15:
        return False
    return True

def onnx_type_to_c_dictionary(data_type_index : int) -> str:
    if data_type_index == 0: 
        return "void*"
    elif data_type_index == 1: 
        return "float32_t"
    elif data_type_index == 2: 
        return "uint8_t"
    elif data_type_index == 3: 
        return "int8_t"
    elif data_type_index == 4: 
        return "uint16_t"
    elif data_type_index == 5: 
        return "int16_t"
    elif data_type_index == 6: 
        return "int32_t"
    elif data_type_index == 7: 
        return "int64_t"
    elif data_type_index == 8: 
        return "char*"
    elif data_type_index == 9: 
        return "bool"
    elif data_type_index == 10: 
        return "float16_t"
    elif data_type_index == 11: 
        return "double"
    elif data_type_index == 12: 
        return "uint32_t"
    elif data_type_index == 13: 
        return "uint64_t"
    elif data_type_index == 14: 
        return "float complex"
    elif data_type_index == 15: 
        return "double complex"
    else:
        raise Exception("Error: unknown onnx data type")
    
def onnx_tensor_elem_type_to_c_dictionary(tensor_elem_type : int) -> str:
    if tensor_elem_type == 0:
        return "void*"
    elif tensor_elem_type == 1:
        return "float32_t*"
    elif tensor_elem_type == 2:
        return "uint8_t*"
    elif tensor_elem_type == 3:
        return "int8_t*"
    elif tensor_elem_type == 4:
        return "uint16_t*"
    elif tensor_elem_type == 5:
        return "int16_t*"
    elif tensor_elem_type == 6:
        return "int32_t*"
    elif tensor_elem_type == 7:
        return "int64_t*"
    elif tensor_elem_type == 8:
        return "char*"
    elif tensor_elem_type == 9:
        return "bool*"
    else:
        raise Exception("Error: unknown input tensor elem type")

def fix_identifier(name : str) -> str:
    return name.replace("/", "").replace(":", "").replace(".", "")

def generate_files(code : list[str], names : list[str]):
    if(len(code) != len(names)):
        raise Exception("Error: code content and code names should be the same number")
    
    output_path : str = CompilerConfig().output_path
    N : int = len(names)
    for i in range(N):
        path : str = output_path + names[i]
        f = open(path, 'w')
        f.write(code[i])
        f.close()

class CompilerLogger:
    _logger = None

    def __new__(cls, config=None):
        if cls._logger is None:
            logger_name : str = 'log_compiler'
            cls._logger = logging.getLogger('log_compiler')
            cls._logger.setLevel(logging.DEBUG)
            log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            path : str = config.framework.temp_path
            filename : str = path + logger_name + ".log"
            log_handler = logging.FileHandler(filename)
            log_handler.setLevel(logging.DEBUG)
            log_handler.setFormatter(log_format)
            cls._logger.addHandler(log_handler)
        return cls._logger
    
class CompilerConfig:
    _config = None

    def __new__(cls, config=None):
        if cls._config is None:
            cls._config = config
        return cls._config