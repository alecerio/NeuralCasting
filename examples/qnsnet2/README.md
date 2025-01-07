# Description

This is an experiment to generate the quantized code of an NSNet2

# Usage

## Install Environment

First, go to the top-level directory of the repository:

```bash
cd /your/top/level/directory/
```

Install the virtual environment to run NeuralCasting:

```bash
python3 -m venv ncenv
```

Activate the virtual environment:

```bash
source nome_ambiente/bin/activate
```

Install packages in virtual environment:

```bash
pip install -r requirements.txt
```

## Generate quantized NSNet2 Code

Go to the quantized NSNet2 quantized folder:

```bash
cd /examples/qnsnet2
```

Inside the Python script *gen_qt_nsnet2.py* to generate the files *nsnet.c* and *nsnet2.h*. Change the fields related to the code generation:

```python
ncgencode(
    name='nsnet2', 
    onnx_path='/your/top/level/directory/examples/qnsnet2/nsnet2_reimplemented_int8_static.onnx',
    output_path='/your/output/directory',
    debug=True
)
```

Set *debug* to *False* if you don't need to print on console the intermediate activations of the neural network.

Finally, use *main.c* and the header files provided in the folder *qt_include* to compile.

You should get a folder with the following files:
- dqlinear.h
- main.c
- nsnet2.c
- nsnet2.h
- qadd.h
- qlinear.h
- qmatmul.h
- qmul.h
- qsigmoid.h
- quant.h
- squeeze.h
- sub.h
- tanh.h
- utils.h

Finally, you can compile the code using a compiler. For example:

```bash
gcc -o nsnet2 nsnet2.c main.c
```
