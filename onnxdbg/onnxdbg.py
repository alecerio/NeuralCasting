from .commands import onnxdbg_copy, onnxdbg_subgr

def onnxdbg(command : str, **kwargs) -> None:
    if command == "copy":
        onnxdbg_copy(**kwargs)
    elif command == "subgr":
        onnxdbg_subgr(**kwargs)