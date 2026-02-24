import onnx
from typing import List
from ops.ncast_op import NCastOp

def set_valid_tensor_identifier(name: str) -> str:
    valid_name = name.replace(".", "_").replace("/", "_").replace("-", "_").replace(":", "_")
    valid_name = "t_" + valid_name
    return valid_name

def is_tensor_in_output(tensor_name: str, graph: onnx.onnx_ml_pb2.GraphProto) -> bool:
    for output in graph.output:
        if output.name == tensor_name:
            return True
    return False

def is_input_tensor(tensor_name: str, graph: onnx.onnx_ml_pb2.GraphProto) -> bool:
    for input in graph.input:
        if input.name == tensor_name:
            return True
    return False

def get_input_tensor(tensor_name: str, graph: onnx.onnx_ml_pb2.GraphProto) -> onnx.onnx_ml_pb2.ValueInfoProto | None:
    for input in graph.input:
        if input.name == tensor_name:
            return input
    return None

def set_onnx_data_type_to_string(dtype):
    dtype_str: str = onnx.TensorProto.DataType.Name(dtype)
    dtype_str = dtype_str.replace("FLOAT", "float").replace("INT64", "int64_t").replace("INT32", "int32_t").replace("DOUBLE", "double").replace("UINT8", "uint8_t").replace("INT8", "int8_t").replace("UINT16", "uint16_t").replace("INT16", "int16_t").replace("FLOAT16", "float16_t").replace("BOOL", "bool")
    return dtype_str

##################################################################################################################################
#                                       Functions to retrieve input tensor information                                 #       #    
##################################################################################################################################

def retrieve_input(graph, ops: List[NCastOp], input):
    result = _retrieve_if_input_tensor(graph, input)
    if result:
        return result
    result = _retrieve_if_intermediate(ops, input)
    if result:
        return result
    result = _retrieve_if_initializer(graph, input)
    if result:
        return result
    return None

def _retrieve_if_input_tensor(graph, input):
    input_tensor = get_input_tensor(input, graph)
    if input_tensor:
        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            shape.append(dim.dim_value)
        dtype = input_tensor.type.tensor_type.elem_type
        return [shape, dtype]
    return None
    
def _retrieve_if_intermediate(ops: List[NCastOp], input):
    for op in ops:
        if input in op.out_dict.keys():
            shape: List[int] = op.out_dict[input].shape
            dtype: int = op.out_dict[input].data_type
            return [shape, dtype]
    return None
    
def _retrieve_if_initializer(graph, input):
    for tensor in graph.initializer:
        if tensor.name == input:
            shape = list(tensor.dims)
            dtype = tensor.data_type
            return [shape, dtype]
    return None