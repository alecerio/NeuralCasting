import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort


import onnx
from onnx import helper, TensorProto, shape_inference

def make_all_node_outputs_as_graph_outputs(in_path: str, out_path: str):
    model = onnx.load(in_path)

    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[WARN] shape_inference fallita: {e}. Proseguo comunque.")

    g = model.graph

    existing_outputs = {o.name for o in g.output}
    input_names = {i.name for i in g.input}
    init_names = {t.name for t in g.initializer}

    vi_map = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        vi_map[vi.name] = vi

    node_output_names = []
    for node in g.node:
        for name in node.output:
            if name and name not in node_output_names:
                node_output_names.append(name)

    added, skipped_unknown = 0, 0

    for name in node_output_names:
        if name in existing_outputs or name in input_names or name in init_names:
            continue
        elem_type = None
        if name in vi_map:
            tt = vi_map[name].type.tensor_type
            if tt is not None and tt.elem_type != 0:
                elem_type = tt.elem_type

        if elem_type is None:
            skipped_unknown += 1
            continue

        g.output.append(helper.make_tensor_value_info(name, elem_type, None))
        added += 1

    onnx.save(model, out_path)
    print(f"[OK] Salvato: {out_path} (aggiunti {added} output, saltati {skipped_unknown} senza tipo)")



def run_ones_and_dump_txt(model_path: str, out_txt: str, default_dim: int = 1):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    feed = {}
    for inp in sess.get_inputs():
        # inp.shape: lista con int, None o stringhe simboliche
        shape = []
        for d in inp.shape:
            if isinstance(d, int) and d > 0:
                shape.append(d)
            else:
                shape.append(default_dim)

        ort2np = {
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
            "tensor(float16)": np.float16,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(int16)": np.int16,
            "tensor(int8)": np.int8,
            "tensor(uint8)": np.uint8,
            "tensor(bool)": np.bool_,
        }
        dtype = ort2np.get(inp.type, np.float32)

        x = np.ones(shape, dtype=dtype)*0.5
        feed[inp.name] = x
        print(f"[INFO] input {inp.name}: shape={x.shape}, dtype={x.dtype}, ort_type={inp.type}")

    outs = sess.run(None, feed)
    out_names = [o.name for o in sess.get_outputs()]

    act = dict(zip(out_names, outs))

    np.set_printoptions(threshold=200, linewidth=160, edgeitems=3)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Num tensors dumped: {len(act)}\n\n")
        for name, val in act.items():
            f.write(f"=== {name} ===\n")
            if isinstance(val, np.ndarray):
                f.write(f"shape: {val.shape}, dtype: {val.dtype}\n")
                f.write(np.array2string(val))
            else:
                f.write(f"type: {type(val)}\n")
                f.write(repr(val))
            f.write("\n\n")
            #np.save(f"activation_{name}.npy", val)
            if name == "/frontend/frontend.0/Conv_output_0_quantized":
                np.save(f"activation_special_{name.replace('/', '_')}.npy", val)
                np.savetxt(f"activation_special_{name.replace('/', '_')}.txt", val.flatten(), fmt="%f")

    print(f"[OK] Dump salvato in: {out_txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_model", required=True)
    ap.add_argument("--out_model", default="model_all_outputs.onnx")
    ap.add_argument("--out_txt", default="activations_dump.txt")
    ap.add_argument("--default_dim", type=int, default=1, help="Valore per dimensioni dinamiche (None/simboliche)")
    args = ap.parse_args()

    make_all_node_outputs_as_graph_outputs(args.in_model, args.out_model)
    run_ones_and_dump_txt(args.out_model, args.out_txt, default_dim=args.default_dim)


if __name__ == "__main__":
    main()
