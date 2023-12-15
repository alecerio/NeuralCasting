from .commands import onnxdbg_copy, onnxdbg_subgr, onnxdbg_inferdbg, onnxdbg_help

def onnxdbg(command : str, **kwargs) -> None:
    if command == "copy":
        onnxdbg_copy(**kwargs)
    elif command == "subgr":
        onnxdbg_subgr(**kwargs)
    elif command == "inferdbg":
        onnxdbg_inferdbg(**kwargs)
    elif command == "help":
        onnxdbg_help(**kwargs)
    else:
        print("Command '" + command + "' not existing")