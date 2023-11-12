import hydra
import numpy as np
import torch.nn as nn
from compiler.frontend.torch2onnx.torch2onnx import torch2onnx
from compiler.frontend.parser.parser.parser import parse
from compiler.frontend.parser.node.node import Node
from compiler.frontend.parser.parser.dag import DAG
from compiler.frontend.exceptions.CompilerException import CompilerException
from compiler.frontend.common.common import CompilerLogger
from compiler.frontend.common.common import CompilerConfig
from compiler.frontend.common.common import generate_files
import onnx

def run(config, **kwargs):

    # init config singleton
    CompilerConfig(config)

    # create logger
    CompilerLogger(config).info("Run compiler")

    # lower from python framework to onnx
    framework = kwargs['framework']
    if framework == 'pytorch':
        model = kwargs['model']
        dummy_input = kwargs['dummy_input']
        params = kwargs['params']

        CompilerLogger().info("Converting pytorch model to onnx")

        # load parameters to the model
        model.load_state_dict(params)

        # create onnx file
        temp_path : str = str(CompilerConfig().temp_path)
        name : str = str(CompilerConfig().name)
        path = temp_path + name + '.onnx'
        verbose : bool = bool(CompilerConfig().framework.verbose)
        torch2onnx(model, dummy_input, path, verbose)
    elif framework == 'onnx':
        path = kwargs['path']

        CompilerLogger().info("Copy onnx file to temp path")

        model = onnx.load(path)
        temp_path : str = str(CompilerConfig().temp_path)
        name : str = str(CompilerConfig().name)
        onnx.save(model, temp_path + name + '.onnx')
    else:
        raise CompilerException("Error: unexpected framework")
    
    # parse onnx
    nodes : list[Node] = parse()

    # create dag
    dag : DAG = DAG(nodes)
    
    # generated code
    [code, names] = dag.traversal_dag_and_generate_code()

    # generate files
    if CompilerConfig().create_output_files:
        generate_files(code, names)

    return [code, names]