import subprocess
from config.config import TEST_TMP_PATH
import numpy as np

def run_bash_command(command: str, verbose: bool = False) -> None:
    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        check=False
    )
    if verbose:
        print(result.stdout)
        print(result.stderr)
        print(result.returncode)

def clear_compilation_folder():
    run_bash_command(f"rm -rf {TEST_TMP_PATH}/*")

def generate_main_c(code: str, file_name: str):
    with open(f"{TEST_TMP_PATH}/{file_name}.c", "w") as f:
        f.write(code)

def read_output(outfilename):
    with open(outfilename, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    return [int(x) for x in text.split(",")]


# ---- HELPER FUNZTIONS FOR QUANTIZATION ----

def quantize_linear(x: np.array, s: float, z: int) -> np.array:
    q_x = np.clip(x / s + z, -128, 127).astype(np.int8)
    return q_x

def dequantize_linear(q_x: np.array, s: float, z: int) -> np.array:
    r_x = s * (q_x.astype(np.float32) - z)
    return r_x

def quantize_linear_fixed_point(x: np.array, s_fx: int, z_fx: int, Q: int) -> np.array:
    z_fx = np.array(z_fx).astype(np.int64)
    s_fx = np.array(s_fx).astype(np.int64)
    q_x = (x*(2**Q) + z_fx * s_fx) / s_fx
    q_x = np.clip(q_x, -128, 127).astype(np.int8)
    return q_x

def dequentize_linear_fixed_point(q_x: np.array, s_fx: int, z_fx: int, Q: int) -> np.array:
    a0 = np.int64(q_x) - z_fx
    r_x = (s_fx * a0) / (2**Q)
    return r_x

def compute_s_z(x_np: np.array):
    s = (np.max(x_np) - np.min(x_np)) / 255.0
    z = np.round(-128.0-np.min(x_np) / s).astype(np.int32)
    return s, z

def compute_sfx_zfx(s: float, z: int, Q: int):
    s_fx = np.round(s * (2.**Q)).astype(np.int64)
    z_fx = z
    return s_fx, z_fx