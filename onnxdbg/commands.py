import pickle
import onnx
from onnx import helper
from onnxdbg.graph.graph import Graph
from onnxdbg.graph.graph_node import GraphNode
from onnxdbg.graph.output_node import OutputNode
from onnxdbg.graph.op_node import OpNode
from onnxdbg.utils import create_graph, inference_onnx_runtime

onnxdbg_copy_args : list[str] = ['src', 'dst', 'mdl']
onnxdbg_subgraph_args : list[str] = ['src', 'dst', 'mdl', 'out']
onnxdbg_inferdbg_args : list[str] = ['srcp', 'dstp', 'mdl', 'input']

def onnxdbg_copy(**kwargs):
    args : list[str] = onnxdbg_copy_args

    if not _check_valid_arguments(args, **kwargs):
        return

    src : str = str(kwargs[args[0]])
    dst : str = str(kwargs[args[1]])
    mdl : str = str(kwargs[args[2]])
    graph : Graph = create_graph(src)
    graph.export_onnx_file(mdl, dst)

def onnxdbg_subgr(**kwargs):
    args : list[str] = onnxdbg_subgraph_args

    if not _check_valid_arguments(args, **kwargs):
        return

    src : str = str(kwargs[args[0]])
    dst : str = str(kwargs[args[1]])
    mdl : str = str(kwargs[args[2]])
    out : str = str(kwargs[args[3]])
    graph : Graph = create_graph(src)
    [initializers, inputs, opnodes, outputs] = graph.create_subgraph_data(out)
    graph = helper.make_graph(opnodes, mdl, inputs, outputs, initializers)
    model = helper.make_model(graph)
    onnx.save(model, dst)

def onnxdbg_inferdbg(**kwargs):
    args : list[str] = onnxdbg_inferdbg_args

    if not _check_valid_arguments(args, **kwargs):
        return

    srcp : str = str(kwargs[args[0]])
    dstp : str = str(kwargs[args[1]])
    mdl : str = str(kwargs[args[2]])
    input_data = kwargs[args[3]]
    
    src : str = srcp + mdl + '.onnx'
    input_names : list[str] = input_data.keys()

    graph : Graph = create_graph(src)
    n_nodes : int = graph.n_nodes()

    dict_output = {}
    dict_output_shape = {}

    for i in range(n_nodes):
        node : GraphNode = graph.get_node(i)
        if isinstance(node, OpNode) or isinstance(node, OutputNode):
            node_name : str = node.get_name()
            [initializers, inputs, opnodes, outputs] = graph.create_subgraph_data(node_name)
            graph_onnx = helper.make_graph(opnodes, mdl, inputs, outputs, initializers)
            model_onnx = helper.make_model(graph_onnx)
            node_name_file_format = 'file_' + node_name.replace(":", "").replace("/", "")
            temp_onnx_name : str = dstp + node_name_file_format + '.onnx'
            onnx.save(model_onnx, temp_onnx_name)

            sub_inputs = []
            model_temp = onnx.load(temp_onnx_name)
            input_names_temp = [input.name for input in model_temp.graph.input]
            for input_name_temp in input_names:
                if input_name_temp in input_names_temp:
                    sub_inputs.append(input_data[input_name_temp])
            
            [outputs_ort, outputs_shape_ort] = inference_onnx_runtime(temp_onnx_name, sub_inputs)

            dict_output[node_name] = outputs_ort
            dict_output_shape[node_name] = outputs_shape_ort
    
    pickle_output_file = dstp + 'output.pkl'
    with open(pickle_output_file, "wb") as pkl_file:
        pickle.dump(dict_output, pkl_file)
    
    pickle_output_shape_file = dstp + 'output_shape.pkl'
    with open(pickle_output_shape_file, "wb") as pkl_file:
        pickle.dump(dict_output_shape, pkl_file)

def onnxdbg_help(**kwargs) -> None:
    nargs : int = len(kwargs.keys())
    if nargs != 0:
        print("No arguments expected for help")
        return
    
    _print_help_header()

    _print_command(
        'copy',
        'This command allows to copy an onnx file from a source directory to a destionation directory.\n'+
        'The file is not simply copied, but the whole graph is re-created.',
        onnxdbg_copy_args,
        ['source onnx file', 'destination onnx file', 'onnx model name'],
        "src='/home/workdir_source/my_model.onnx'\n" +
        "dst='/home/workdir_destination/my_model.onnx'\n" + 
        "mdl='model_onnx'"
    )

    _print_command(
        'help',
        'This command prints the available commands of onnxdbg.\n',
        [],
        [],
        ""
    )

    _print_command(
        'inferdbg',
        'This command creates a sub-graph for possible operation node and output node and it runs the inference of the sub-graph.\n' +
        'It provides, given an input, the output and the output shape for each node in the onnx file, this makes it perfect for debugging\n' +
        'The results are returned in form of pkl files.',
        onnxdbg_inferdbg_args,
        ['source path', 'destination path', 'onnx model name', 'input dictionary for the inference'],
        "src='/home/workdir_source/'\n" +
        "dst='/home/workdir_destination/'\n" + 
        "mdl='model_onnx'\n" +
        "input={input1: [0, 0, 0], input2: [1, 1, 1, 1]}"
    )

    _print_command(
        'subgr',
        'This command allows to create a new onnx file that is a sub-graph of the original file.\n',
        onnxdbg_subgraph_args,
        ['source onnx file', 'destination sub-graph onnx file', 'onnx model name', 'name of onnx node to use as new output'],
        "src='/home/workdir_source/my_model.onnx'\n" +
        "dst='/home/workdir_destination/my_sub_model.onnx'\n" + 
        "mdl='sub_model_onnx'\n" +
        "out=/Tanh"
    )

    _print_help_tail()

def _check_valid_arguments(args : list[str], **kwargs) -> bool:
    keys = kwargs.keys()
    for key in keys:
        if not (key in args):
            print("Argument '" + key + "' unexpected") 
            return False
    return True

def _print_command(command_name : str, description : str, args_name : list[str], args_desc : list[str], example : str) -> None:
    print('Command: ' + command_name)
    print(description)
    n_args : int = len(args_name)
    for i in range(n_args):
        arg_name : str = args_name[i]
        arg_desc : str = args_desc[i]
        print('\t' + arg_name + ' - ' + arg_desc) 
    print('Example:\n')
    print(example)
    print('\n' * 2)

def _print_help_header() -> None:
    print("\n" * 2)
    print("################################################################")
    print("\t\t\t ONNXDBG INFO")
    print("################################################################")
    print("\n" * 2)

def _print_help_tail() -> None:
    print("\n" * 2)
    print("################################################################")
    print("\n" * 2)