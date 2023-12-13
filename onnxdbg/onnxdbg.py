from .commands import onnxdbg_copy

def onnxdbg(command : str, **kwargs) -> None:
    if command == "copy":
        onnxdbg_copy(**kwargs)