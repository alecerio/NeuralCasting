# NeuralCasting

## Description

*Neural Casting* is a front-end compiler to convert ONNX format (*Open Neural Network Exchange*) to a specific programming language. Currently, conversion to C code is supported, but the software aims to provide an infrastructure suited to convert ONNX structure to different programming languages and intermediate representations. This makes *Neural Casting* a suitable entry point for backend compilers.
The core of *Neural Casting* is the use of the *Direct Acyclic Graph* (DAG) provided by ONNX structure to replace the *Abstract Syntax Tree* (AST) of the traditional compilers. As a result, the code generation is possible with an exploration algorithm of such data structure, considering the dependencies between the different operators of the encoded neural network.  

## Setup

For the setup of *Neural Casting*, use the following procedure:

1. Install the conda environment:

```shell
conda env create -f env.yaml
```

You can find `env.yaml` in the main page of the repository.

2. Activate the conda environment:

```shell
conda activate neural_casting
```

3. To install the compiler package in `neural_casting`, open the terminal and go the repository directory:

```shell
cd /path/to/repository/
```

Consequently, run the setup file:

```shell
python setup.py install
```

4. Install *gcc* in your system. For example, here is reported the installation for Ubuntu 22.04.

```shell
sudo apt update
sudo apt install gcc
gcc --version
```

5. In *config/config.yaml*, update the settings:

- *name*: the name of the project you want to build (for example, if name is *dummy* and you compile to C code, the generated files will be *dummy.c* and *dummy.h*).
- *repo*: the path where you cloned the repository.
- *workdir*: the work directory where you want to generate temporary and output files when you build the neural network.

## Usage

You can build you neural network calling the python script `neuralcasting.py`.

WORK IN PROGRESS ...

### Examples

WORK IN PROGRESS ...

## Tests

There are different tests implemented to ensure the generated code is correct. You can find the list of implemented tests in `tests/neural_networks/`:

- *constant*: test for ONNX constant operator.
- *fc_add*: test for a neural network composed of a fully connected layer and an element-wise addition.
- *fc_mul*: test for a neural network composed of a fully connected layer and an element-wise multiplication.
- *fc_relu*: test for a neural network composed of a fully connected layer and a ReLu activation function.
- *fc_relu*: test for a neural network composed of a first fully connected layer, a first ReLu activation function, a second fully connected layer and a second ReLu activation function.
- *fc_sigmoid*: test for a neural network composed of a fully connected layer and a sigmoid activation function.
- *fc_sub*: test for a neural network composed of a fully connected layer and an element-wise subtraction.
- *fc_tanh*: test for a neural network composed of a fully connected layer and a tanh activation function.
- *gather*: test for ONNX gather operator.
- *matmul*: test for ONNX matmul operator.
- *reimplemented_gru*: test for a reimplementation of a GRU layer.

### Run the tests

Go the the repository directory:

```shell
cd /path/to/repository/
```

Ensure the compiler package is up to date:

```shell
python setup.py install
```

Run the tests wrapper:

```shell
python tests/neural_networks/run_tests_wrapper.py
``` 

## Open Works

The current main goal of the project is to run audio related neural networks on an STM32 board, specifically for de-noising.

Nevertheless, the project aims to provide a general infrastructure and to cover a wide range of use cases. Consequently, there are many oppurtunities of features integration in the future:

- Supporting a wider range of ONNX operators to support more use cases.
- Supporting the generation of other programming languages beyond C and intermediate representations (e.g. MLIR dialects).
- Implementation of optimizations and models compression techniques to the generated C code.
- Implementation of operator fusion techniques.

## Authors and Contacts

*NeuralCasting* is a project developed by [Alessandro Cerioli](https://dk.linkedin.com/in/alessandro-cerioli-26237231) during his Industrial PhD at [Jabra](https://www.jabra.dk/) and DTU ([Technical University of Denmark](https://www.dtu.dk/english/)) and is part of the European project [Convolve](https://convolve.eu/). For more information regarding the project or actively to contribute to the development of the repository, use the following contacts:

- **Jabra email**: alcerioli@jabra.com
- **DTU email**: alceri@dtu.dk