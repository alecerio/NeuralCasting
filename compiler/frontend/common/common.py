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
