from .commands import onnxdbg_copy, onnxdbg_subgr, onnxdbg_inferdbg

def onnxdbg(command : str, **kwargs) -> None:
    if command == "copy":
        onnxdbg_copy(**kwargs)
    elif command == "subgr":
        onnxdbg_subgr(**kwargs)
    elif command == "inferdbg":
        onnxdbg_inferdbg(**kwargs)
    else:
        print("Command '" + command + "' not existing")