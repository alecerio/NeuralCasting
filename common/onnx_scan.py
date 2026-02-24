import argparse
import onnx
from config.config import ONNX_DIR

def add_to_ops_dict(ops_dict, op_type):
    if op_type not in ops_dict:
        ops_dict[op_type] = 1
    else:
        ops_dict[op_type] += 1

def scan_onnx_ops(model_path):
    model = onnx.load(model_path)
    graph = model.graph

    print(f"Model: {model_path}")
    print(f"Number of nodes: {len(graph.node)}\n")

    ops_dict = {}

    for idx, node in enumerate(graph.node):
        op_type = node.op_type
        name = node.name if node.name else "<unnamed>"

        print(f"{idx:4d} | {op_type:20s} | {name}")

        add_to_ops_dict(ops_dict, op_type)
    
    print("\nOperators summary:")
    print(ops_dict)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ONNX Scanner")

    parser.add_argument("--model", type=str, required=True,
                    help="ONNX file name (without path)")
    parser.add_argument("--path", type=str, default=ONNX_DIR,
                    help="Input/Output ONNX file path")

    args = parser.parse_args()

    scan_onnx_ops(f"{args.path}/{args.model}.onnx")
