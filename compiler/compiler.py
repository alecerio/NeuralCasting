import hydra
import numpy as np
import torch.nn as nn
from compiler.frontend.torch2onnx.torch2onnx import torch2onnx
from compiler.frontend.parser.parser.parser import parse
from compiler.frontend.parser.node.node import Node
from compiler.frontend.parser.parser.dag import DAG

def run(config, model, dummy_input, params = None):
    print("run compiler ...")

    # lower from python framework to onnx
    fr = config.framework
    if(fr.framework_name == 'pytorch'):
        # load parameters to the model
        model.load_state_dict(params)

        # create onnx file
        temp_path : str = str(fr.temp_path)
        name : str = str(fr.name)
        path = temp_path + name + '.onnx'
        verbose : bool = bool(fr.verbose)
        torch2onnx(model, dummy_input, path, verbose)
    else:
        raise Exception("Error: unexpected framework")
    
    # parse onnx
    nodes : list[Node] = parse(config)

    # create dag
    dag : DAG = DAG(nodes)
    
    # generated code
    code : str = dag.traversal_dag_and_generate_code()

    print(code)