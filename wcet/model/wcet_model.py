import onnx
from onnx import helper
from config.config import ONNX_DIR, WCET_OUT_PATH
from graph.ncast_graph import NCastGraph
from ops.ncast_op import NCastOp
from wcet.qlinearadd.wcet_qlinearadd import qlinearadd_wcet_analysis
from wcet.qlinearconv.wcet_qlinearconv import qlinearconv_wcet_analysis
from wcet.qlinearmatmul.wcet_qlinearmatmul import qlinearmatmul_wcet_analysis
from wcet.qlinearmul.wcet_qlinearmul import qlinearmul_wcet_analysis
from wcet.qlinearprelu.wcet_qlinearprelu import qlinearprelu_wcet_analysis
from wcet.qlinearrelu.wcet_qlinearrelu import qlinearrelu_wcet_analysis
from wcet.qlinearsigmoid.wcet_qlinearsigmoid import qlinearsigmoid_wcet_analysis
from wcet.qlinearsub.wcet_qlinearsub import qlinearsub_wcet_analysis
from wcet.qlineartanh.wcet_qlineartanh import qlineartanh_wcet_analysis
from wcet.transpose.wcet_transpose import transpose_wcet_analysis
from wcet.unsqueeze.wcet_unsqueeze import unsqueeze_wcet_analysis
from common.common import set_valid_tensor_identifier
import warnings
from pathlib import Path
import re
import subprocess

def wcet_model(onnx_path):
    ncgraph = NCastGraph(onnx_path)

    _clear_wcet_out()

    for op in ncgraph.ops:
        optype = op.onnx_unit.op_type
        if optype == 'QLinearAdd':
            _qlinearadd_analysis(op)
        elif optype == 'QLinearConv':
            _qlinearconv_analysis(op, ncgraph)
        elif optype == 'QLinearMatMul':
            _qlinearmatmul_analysis(op)
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
            warnings.warn(f"Operator {optype} not supported for WCET analysis.")


def _qlinearadd_analysis(op):
    return
    name = set_valid_tensor_identifier(op.onnx_unit.name)
    size = _extract_output_size(op, 0)
    qlinearadd_wcet_analysis(name=name, size=size)

def _qlinearconv_analysis(op: NCastOp, ncgraph:NCastGraph):
    name = set_valid_tensor_identifier(op.onnx_unit.name)
    Q = 31
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

    qlinearconv_wcet_analysis(name, COUT, CIN, KS, LIN, PAD, DIL, STR, Q)

def _qlinearmatmul_analysis(op, ncgraph):
    return
    name = set_valid_tensor_identifier(op.onnx_unit.name)
    a_shape = ncgraph.get_tensor_shape(op.onnx_unit.input[0])
    b_shape = ncgraph.get_tensor_shape(op.onnx_unit.input[3])
    M = a_shape[-2]
    K = a_shape[-1]
    N = b_shape[-1]
    qlinearmatmul_wcet_analysis(name, M, N, K)

def _qlinearmul_analysis(op):
    return
    name = set_valid_tensor_identifier(op.onnx_unit.name)
    size = _extract_output_size(op, 0)
    qlinearmul_wcet_analysis(name, size)
    
def _qlinearprelu_analysis(op):
    return
    name = set_valid_tensor_identifier(op.onnx_unit.name)
    size = _extract_output_size(op, 0)
    qlinearprelu_wcet_analysis(name, size)

def _qlinearrelu_analysis(op):
    return
    name = set_valid_tensor_identifier(op.onnx_unit.name)
    size = _extract_output_size(op, 0)
    qlinearrelu_wcet_analysis(name, size)

def _qlinearsigmoid_analysis(op):
    return
    name = set_valid_tensor_identifier(op.onnx_unit.name)
    size = _extract_output_size(op, 0)
    qlinearsigmoid_wcet_analysis(name, size)

def _qlinearsub_analysis(op):
    return
    name = set_valid_tensor_identifier(op.onnx_unit.name)
    size = _extract_output_size(op, 0)
    qlinearsub_wcet_analysis(name, size)

def _qlineartanh_analysis(op):
    return
    name = set_valid_tensor_identifier(op.onnx_unit.name)
    size = _extract_output_size(op, 0)
    qlineartanh_wcet_analysis(name, size)

def _transpose_analysis(op, ncgraph):
    return
    name = set_valid_tensor_identifier(op.onnx_unit.name)
    input_shape = ncgraph.get_tensor_shape(op.onnx_unit.input[0])
    rows = input_shape[-2]
    cols = input_shape[-1]
    transpose_wcet_analysis(name, cols, rows)

def _unsqueeze_analysis(op):
    return
    name = set_valid_tensor_identifier(op.onnx_unit.name)
    size = _extract_output_size(op, 0)
    unsqueeze_wcet_analysis(name, size)

def _extract_output_size(op, index):
    out_keys = list(op.out_dict.keys())
    shape = op.out_dict[out_keys[0]].shape
    size = 1
    for dim in shape:
        size = size * dim
    return size

def analyze_output():
    files = [f.name for f in Path(f"{WCET_OUT_PATH}").glob("*.txt")]
    for file in files:
        filename = file.split('.')[0]
        if filename[-5:] == '_wcet':
            with open(f"{WCET_OUT_PATH}/{file}", "r") as f:
                results = f.read()
                cycles = _extract_num_cycles(results)
                print(f"{filename}: {cycles}")

def _extract_num_cycles(results: str):
    match = re.search(r"cycles:\s*(-?\d+)", results)
    if match:
        cycles = int(match.group(1))
        return int(cycles)
    return None

def _clear_wcet_out():
    subprocess.run("rm -f *", cwd=f"{WCET_OUT_PATH}", shell=True)

if __name__ == '__main__':
    onnx_path = f"{ONNX_DIR}/convsenet_int8_optimized.onnx"
    wcet_model(onnx_path)
    analyze_output()

