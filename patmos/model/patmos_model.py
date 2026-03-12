from config.config import ONNX_DIR, PATMOS_OUT_PATH, WCET_OUT_PATH
from graph.ncast_graph import NCastGraph
import warnings
from patmos.qlinearadd.patmos_qlinearadd import qlinearadd_patmos_analysis
from patmos.qlinearconv.patmos_qlinearconv import qlinearconv_patmos_analysis
from patmos.qlinearmatmul.patmos_qlinearmatmul import qlinearmatmul_patmos_analysis
from patmos.qlinearmul.patmos_qlinearmul import qlinearmul_patmos_analysis
from patmos.qlinearprelu.patmos_qlinearprelu import qlinearprelu_patmos_analysis
from patmos.qlinearrelu.patmos_qlinearrelu import qlinearrelu_patmos_analysis
from patmos.qlinearsigmoid.patmos_qlinearsigmoid import qlinearsigmoid_patmos_analysis
from patmos.qlinearsub.patmos_qlinearsub import qlinearsub_patmos_analysis
from patmos.qlineartanh.patmos_qlineartanh import qlineartanh_patmos_analysis
from patmos.transpose.patmos_transpose import transpose_patmos_analysis
from patmos.unsqueeze.patmos_unsqueeze import unsqueeze_patmos_analysis
from common.common import set_valid_tensor_identifier
from wcet.model.wcet_model import _extract_output_size
from pathlib import Path
import re
from onnx import helper
import subprocess

def patmos_model(onnx_path):
    ncgraph = NCastGraph(onnx_path)
    _clear_patmos_out()

    for op in ncgraph.ops:
        optype = op.onnx_unit.op_type
        if optype == 'QLinearAdd':
            _qlinearadd_analysis(op)
        elif optype == 'QLinearConv':
            _qlinearconv_analysis(op, ncgraph)
        elif optype == 'QLinearMatMul':
            _qlinearmatmul_analysis(op, ncgraph)
        elif optype == 'QLinearMul':
            _qlinearmul_analysis(op)
        elif optype == 'PRelu':
            _qlinearprelu_analysis(op)
        elif optype == 'Relu':
            _qlinearrelu_analysis(op)
        elif optype == 'Sigmoid':
            _qlinearsigmoid_analysis(op)
        elif optype == 'Sub':
            _qlinearsub_analysis(op)
        elif optype == 'Tanh':
            _qlineartanh_analysis(op)
        elif optype == 'Transpose':
            _transpose_analysis(op)
        elif optype == 'Unsqueeze':
            _unsqueeze_analysis(op)
        else:
            warnings.warn(f"Operator {optype} not supported on Patmos benchmark.")

def _qlinearadd_analysis(op):
    name = _gen_name(op)
    size = _extract_output_size(op, 0)
    acctype = "int32_t"
    qlinearadd_patmos_analysis(name, size, acctype)

def _qlinearconv_analysis(op, ncgraph):
    name = _gen_name(op)
    Q = 15
    input_shape = ncgraph.get_tensor_shape(op.onnx_unit.input[0])
    w_shape = ncgraph.get_tensor_shape(op.onnx_unit.input[3])
    output_names = list(op.out_dict.keys())
    output_shape = op.out_dict[output_names[0]].shape
    attrs = {a.name: helper.get_attribute_value(a) for a in op.onnx_unit.attribute}
    group = attrs.get("group", 1)
    CIN = w_shape[1] * group
    KS = w_shape[2]
    LIN = input_shape[2]
    COUT = w_shape[0]
    PAD = attrs.get("pads", [0, 0])[0]
    DIL = attrs.get("dilations", 1)[0]
    STR = attrs.get("strides", 1)[0]
    acctype = "int32_t"
    qlinearconv_patmos_analysis(name, KS, CIN, LIN, COUT, PAD, DIL, STR, Q, acctype)

def _qlinearmatmul_analysis(op, ncgraph):
    name = _gen_name(op)
    a_shape = ncgraph.get_tensor_shape(op.onnx_unit.input[0])
    b_shape = ncgraph.get_tensor_shape(op.onnx_unit.input[3])
    
    if len(a_shape) > 1:
        a_shape = a_shape[-2:]
    else:
        a_shape = [1, a_shape[0]]
    
    if len(b_shape) > 1:
        b_shape = b_shape[-2:]
    else:
        b_shape = [a_shape[0], 1]

    M = a_shape[-2]
    K = a_shape[-1]
    N = b_shape[-1]
    Q = 15
    acctype = "int32_t"
    qlinearmatmul_patmos_analysis(name, M, K, N, Q, acctype)

def _qlinearmul_analysis(op):
    name = _gen_name(op)
    size = _extract_output_size(op, 0)
    Q = 15
    acctype = "int32_t"
    qlinearmul_patmos_analysis(name, size, Q, acctype)

def _qlinearprelu_analysis(op):
    name = _gen_name(op)
    size = _extract_output_size(op, 0)
    Q = 15
    acctype = "int32_t"
    qlinearprelu_patmos_analysis(name, size, Q, acctype)

def _qlinearrelu_analysis(op):
    name = _gen_name(op)
    size = _extract_output_size(op, 0)
    acctype = "int32_t"
    qlinearrelu_patmos_analysis(name, size, acctype)

def _qlinearsigmoid_analysis(op):
    name = _gen_name(op)
    size = _extract_output_size(op, 0)
    acctype = "int32_t"
    qlinearsigmoid_patmos_analysis(name, size, acctype)

def _qlinearsub_analysis(op):
    name = _gen_name(op)
    size = _extract_output_size(op, 0)
    acctype = "int32_t"
    qlinearsub_patmos_analysis(name, size, acctype)

def _qlineartanh_analysis(op):
    name = _gen_name(op)
    size = _extract_output_size(op, 0)
    acctype = "int32_t"
    qlineartanh_patmos_analysis(name, size, acctype)

def _transpose_analysis(op, ncgraph):
    name = _gen_name(op)
    input_shape = ncgraph.get_tensor_shape(op.onnx_unit.input[0])
    rows = input_shape[-2]
    cols = input_shape[-1]
    transpose_patmos_analysis(name, cols, rows)

def _unsqueeze_analysis(op):
    name = _gen_name(op)
    size = _extract_output_size(op, 0)
    unsqueeze_patmos_analysis(name, size)

def analyze_output():
    analysis_output = ""
    files = [f.name for f in Path(f"{PATMOS_OUT_PATH}").glob("*.txt")]
    for file in files:
        filename = file.split('.')[0]
        if filename[-11:] == '_patmos_out':
            with open(f"{PATMOS_OUT_PATH}/{file}", "r") as f:
                results = f.read()
                cycles = _extract_num_cycles(results)
                analysis_output += f"{filename}: {cycles}\n"
    with open(f"{WCET_OUT_PATH}/wcet-analysis-output.txt", "w") as f:
        f.write(analysis_output)

def _extract_num_cycles(results: str):
    match = re.search(r"cpu-cycles:\s*(-?\d+)", results)
    if match:
        cycles = int(match.group(1))
        return int(cycles)
    return None

def _gen_name(op):
    return f"{op.onnx_unit.op_type}-{set_valid_tensor_identifier(op.onnx_unit.name)}"

def _clear_patmos_out():
    subprocess.run("rm -f *", cwd=f"{PATMOS_OUT_PATH}", shell=True)


