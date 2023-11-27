import numpy as np
import torch.nn as nn
from neural_cast.frontend.torch2onnx.torch2onnx import torch2onnx
from neural_cast.frontend.parser.parser.parser import parse
from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.parser.node.output_node import OutputNode
from neural_cast.frontend.parser.parser.dag import DAG
from neural_cast.frontend.exceptions.CompilerException import CompilerException
from neural_cast.frontend.common.common import CompilerLogger
from neural_cast.frontend.common.common import CompilerConfig
from neural_cast.frontend.common.common import generate_files
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
        temp_path : str = str(CompilerConfig()['temp_path'])
        name : str = str(CompilerConfig()['name'])
        path = temp_path + name + '.onnx'
        torch2onnx(model, dummy_input, path, True)
    elif framework == 'onnx':
        path = kwargs['path']

        CompilerLogger().info("Copy onnx file to temp path")

        model = onnx.load(path)
        temp_path : str = str(CompilerConfig()['temp_path'])
        name : str = str(CompilerConfig()['name'])
        onnx.save(model, temp_path + name + '.onnx')
    else:
        raise CompilerException("Error: unexpected framework")
    
    # parse onnx
    nodes : list[Node] = parse()

    # create dag
    dag : DAG = DAG(nodes)

    # export output shape
    _export_output_shape(nodes)
    
    # generated code
    [code, names] = dag.traversal_dag_and_generate_code()

    # generate files
    if CompilerConfig()['create_output_files']:
        generate_files(code, names)

    return [code, names]

def _export_output_shape(nodes : list[Node]) -> None:
    output_nodes : list[OutputNode] = []
    for node in nodes:
        if isinstance(node, OutputNode):
            output_nodes.append(node)
    json_output_shape : str = "{\n"
    for output_node in output_nodes:
        json_output_shape += "\"" + output_node.get_name() + "\"" + ": " + str(output_node.infer_output_shape()) + "\n"
    json_output_shape += "}"
    temp_path : str = CompilerConfig()['temp_path'] + "out_shape.json"
    f = open(temp_path, 'w')
    f.write(json_output_shape)
    f.close()