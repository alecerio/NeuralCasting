from neural_cast.compiler import run
from neural_cast.frontend.common.common import CompilerConfig
import os
import yaml
import shutil

def move_to_output_folder(source : str, dest : str):
    try:
        shutil.copy(source, dest)
    except FileNotFoundError:
        print("File not found.")
    except PermissionError:
        print("Permission denied.")
    except Exception as e:
        print("An error occurred:", e)

curr_file = os.path.abspath(__file__)
curr_path = os.path.dirname(curr_file)
with open(curr_path + '/config.yaml', 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
CompilerConfig(config)

# init config file
name : str = CompilerConfig()['name']
output_path : str = CompilerConfig()['output_path']
test_path : str = CompilerConfig()['repo'] + 'debug_framework/'
temp_path : str = CompilerConfig()['temp_path']
path_onnx = test_path + 'nsnet2.onnx'

# run compiler 
run(CompilerConfig(), framework='onnx', path=path_onnx)

# copy main.c to output folder
source : str = CompilerConfig()['repo'] + '/examples/nsnet/main.c'
dest : str = CompilerConfig()['output_path'] + '/main.c'
move_to_output_folder(source, dest)

# copy build.sh to output folder
source : str = CompilerConfig()['repo'] + '/examples/nsnet/build.sh'
dest : str = CompilerConfig()['output_path'] + '/build.sh'
move_to_output_folder(source, dest)