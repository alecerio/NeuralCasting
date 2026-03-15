import onnx
from onnx import helper
from config.config import ONNX_DIR, WCET_OUT_PATH
from graph.ncast_graph import NCastGraph
from ops.ncast_op import NCastOp
from wcet.qlinearadd.wcet_qlinearadd import qlinearadd_wcet_analysis
from wcet.qlinearconv.wcet_qlinearconv import qlinearconv_wcet_analysis
from wcet.qlinearconv2.wcet_qlinearconv2 import qlinearconv2d_wcet_analysis
from wcet.qlinearmatmul.wcet_qlinearmatmul import qlinearmatmul_wcet_analysis
from wcet.qlinearmul.wcet_qlinearmul import qlinearmul_wcet_analysis
from wcet.qlinearprelu.wcet_qlinearprelu import qlinearprelu_wcet_analysis
from wcet.qlinearrelu.wcet_qlinearrelu import qlinearrelu_wcet_analysis
from wcet.qlinearsigmoid.wcet_qlinearsigmoid import qlinearsigmoid_wcet_analysis
from wcet.qlinearsub.wcet_qlinearsub import qlinearsub_wcet_analysis
from wcet.qlineartanh.wcet_qlineartanh import qlineartanh_wcet_analysis
from wcet.transpose.wcet_transpose import transpose_wcet_analysis
from wcet.unsqueeze.wcet_unsqueeze import unsqueeze_wcet_analysis
from wcet.qgemm.wcet_qgemm import qgemm_wcet_analysis
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
        elif optype == 'QGemm':
            _qgemm_analysis(op)
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
            warnings.warn(f"Operator {optype} not supported for WCET analysis.")

def _qlinearadd_analysis(op):
    name = _gen_name(op)
    size = _extract_output_size(op, 0)
    qlinearadd_wcet_analysis(name=name, size=size, acctype="int32_t")

def _qlinearconv_analysis(op: NCastOp, ncgraph:NCastGraph):
    rank = op._get_conv_rank(ncgraph.graph, ncgraph.ops)
    print(f"rank: {rank}")
    if rank == 1:
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
        qlinearconv_wcet_analysis(name, COUT, CIN, KS, LIN, PAD, DIL, STR, Q, acctype)
    elif rank == 2:
        name = _gen_name(op)
        Q = 15

        input_shape = ncgraph.get_tensor_shape(op.onnx_unit.input[0])
        w_shape = ncgraph.get_tensor_shape(op.onnx_unit.input[1])

        output_names = list(op.out_dict.keys())
        output_shape = op.out_dict[output_names[0]].shape

        attrs = {a.name: helper.get_attribute_value(a) for a in op.onnx_unit.attribute}

        group = attrs.get("group", 1)

        CIN = w_shape[1] * group
        KH = w_shape[2]
        KW = w_shape[3]

        HIN = input_shape[2]
        WIN = input_shape[3]

        COUT = w_shape[0]

        pads = attrs.get("pads", [0, 0, 0, 0])
        PADH = pads[0]
        PADW = pads[1]

        dilations = attrs.get("dilations", [1, 1])
        DILH = dilations[0]
        DILW = dilations[1]

        strides = attrs.get("strides", [1, 1])
        STRH = strides[0]
        STRW = strides[1]

        acctype = "int32_t"

        qlinearconv2d_wcet_analysis(
            name=name,
            COUT=COUT,
            CIN=CIN,
            KH=KH,
            KW=KW,
            HIN=HIN,
            WIN=WIN,
            PADH=PADH,
            PADW=PADW,
            DILH=DILH,
            DILW=DILW,
            STRH=STRH,
            STRW=STRW,
            Q=Q,
            acctype=acctype,
        )
    else:
        raise Exception("COnvolution rank not supported.")

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
    qlinearmatmul_wcet_analysis(name, M, N, K, Q, acctype)

def _qgemm_analysis(op, ncgraph):
    name = _gen_name(op)
    a_shape = ncgraph.get_tensor_shape(op.onnx_unit.input[0])
    b_shape = ncgraph.get_tensor_shape(op.onnx_unit.input[3])

    attrs = {a.name: helper.get_attribute_value(a) for a in op.onnx_unit.attribute}
    transA = attrs.get("transA", 0)
    transB = attrs.get("transB", 0)
    
    dimsA = len(a_shape)
    dimsB = len(b_shape)
    if dimsA > 1 and dimsB > 1:
        idxA = -2 if transA == 0 else -1
        idxK = -1 if transA == 0 else -2
        idxB = -1 if transB == 0 else -2
        M = a_shape[idxA]
        K = a_shape[idxK]
        N = b_shape[idxB]
    elif dimsA > 1 and dimsB == 1:
        idxA = -2 if transA == 0 else -1
        idxK = -1 if transA == 0 else -2
        M = a_shape[idxA]
        K = a_shape[idxK]
        N = 1
    elif dimsA == 1 and dimsB > 1:
        idxB = -1 if transB == 0 else -2
        idxK = -2 if transB == 0 else -1
        M = 1
        K = b_shape[idxK]
        N = b_shape[idxB]
    else:
        M = 1
        N = 1
        K = a_shape[0]
    Q = 15
    acctype = "int32_t"
    qgemm_wcet_analysis(name, M, N, K, Q, acctype)

def _qlinearmul_analysis(op):
    name = _gen_name(op)
    size = _extract_output_size(op, 0)
    acctype = "int32_t"
    Q = 15
    qlinearmul_wcet_analysis(name, size, Q, acctype)
    
def _qlinearprelu_analysis(op):
    name = _gen_name(op)
    size = _extract_output_size(op, 0)
    acctype = "int32_t"
    Q = 15
    qlinearprelu_wcet_analysis(name, size, Q, acctype)

def _qlinearrelu_analysis(op):
    name = _gen_name(op)
    size = _extract_output_size(op, 0)
    acctype = "int32_t"
    qlinearrelu_wcet_analysis(name, size, acctype)

def _qlinearsigmoid_analysis(op):
    name = _gen_name(op)
    size = _extract_output_size(op, 0)
    acctype = "int32_t"
    qlinearsigmoid_wcet_analysis(name, size, acctype)

def _qlinearsub_analysis(op):
    name = _gen_name(op)
    size = _extract_output_size(op, 0)
    acctype = "int32_t"
    qlinearsub_wcet_analysis(name, size, acctype)

def _qlineartanh_analysis(op):
    name = _gen_name(op)
    size = _extract_output_size(op, 0)
    acctype = "int32_t"
    qlineartanh_wcet_analysis(name, size, acctype)

def _transpose_analysis(op, ncgraph):
    name = _gen_name(op)
    input_shape = ncgraph.get_tensor_shape(op.onnx_unit.input[0])
    rows = input_shape[-2]
    cols = input_shape[-1]
    transpose_wcet_analysis(name, cols, rows)

def _unsqueeze_analysis(op):
    name = _gen_name(op)
    size = _extract_output_size(op, 0)
    unsqueeze_wcet_analysis(name, size)

def _extract_output_size(op, index):
    out_keys = list(op.out_dict.keys())
    shape = op.out_dict[out_keys[index]].shape
    size = 1
    for dim in shape:
        size = size * dim
    return size

def analyze_output():
    analysis_output = ""
    files = [f.name for f in Path(f"{WCET_OUT_PATH}").glob("*.txt")]
    for file in files:
        filename = file.split('.')[0]
        if filename[-5:] == '_wcet':
            with open(f"{WCET_OUT_PATH}/{file}", "r") as f:
                results = f.read()
                cycles = _extract_num_cycles(results)
                analysis_output += f"{filename}: {cycles}\n"
    with open(f"{WCET_OUT_PATH}/wcet-analysis-output.txt", "w") as f:
        f.write(analysis_output)

def _extract_num_cycles(results: str):
    match = re.search(r"cycles:\s*(-?\d+)", results)
    if match:
        cycles = int(match.group(1))
        return int(cycles)
    return None

def _gen_name(op):
    return f"{op.onnx_unit.op_type}-{set_valid_tensor_identifier(op.onnx_unit.name)}"

def _clear_wcet_out():
    subprocess.run("rm -f *", cwd=f"{WCET_OUT_PATH}", shell=True)


