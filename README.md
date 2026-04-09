<p align="center">
  <img src="logo.png" width="300" alt="Logo">
</p>

# NeuralCasting

## Description

The **NeuralCasting repository** provides a framework for converting neural network models from the ONNX format into efficient, standalone C code tailored for embedded systems. It includes support for the Patmos processor, a time-predictable architecture designed for real-time applications, enabling execution with predictable timing behavior.

The framework constructs an internal representation of the neural network and generates static C implementations of the supported operators, eliminating the need for heavyweight inference runtimes. It also offers quantization support (e.g., 8-bit and fixed-point arithmetic) to reduce memory usage and computational cost.

Overall, NeuralCasting is designed to deploy neural networks on resource-constrained and real-time systems, ensuring efficient execution and enabling worst-case execution time (WCET) analysis.

## Structure

The NeuralCasting repository is structured as followed:

- *common*: contains the shared code of the framework.
- *config*: contains the config files of the repository.
- *custom_ops*: contains the definition for custom operations not canonical in ONNX. These units can be used within rewrite patterns.
- *experiments*: contains the results of the experiments.
- *graph*: constains the definition of the NeuralCasting graph genertaed from the ONNX models.
- *mem*: contains tools for the memory analysis of the models using the NeuralCasting graph.
- *ncast_lib*: contains the C code library of NeuralCasting.
- *node_fusion*: contains the rewrite patterns to modify the NeuralCasting graph.
- *onnx*: contains the onnx models used for the experiments.
- *ops*: contains the single operators of the NeuralCasting framework that compose the NeuralCasting graph.
- *patmos*: contains the code for the Patmos benchmark of the different operators.
- *wcet*: contains the code for the WCET analysis with Platin of the different operators.


## Setup

Clone the repository:

```bash
git clone git@github.com:alecerio/NeuralCasting.git 
```

Go into NeuralCasting folder:

```bash
cd NeuralCasting 
```

Create and activate Python envinronment:
```bash
python3 -m venv ncast-env
source ncast-env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the setup of t-crest, see the [t-crest repository](https://github.com/t-crest/patmos).

Update config/config.py:
```python
BASE_DIR = "<your-path-to-neuralcasting>/NeuralCasting"
T_CREST_PATH = "<your-path-to-tcrest>/t-crest"
```

## Tests

Run all the tests in NeuralCasting for the different operators:

```bash
PYTHONPATH="<your-path-to-neuralcasting>/NeuralCasting" pytest
```

These tests compare the results of the baseline in float32, with:
- Quantized int8 simulation in Python with scaling factors in floating point.
- Quantized int8 simulation in Python with scaling factors in fixed point.
- Quantized int8 in C with scaling factors in fixed point.

It is possible to run the tests for the single operators. For example, to run the tests related to qgemm (quantized generalized matrix multiplication):

```bash
PYTHONPATH="<your-path-to-neuralcasting>/NeuralCasting" pytest tests/qgemm/test_qgemm.py
```

## Patmos

To run the benchmark on Patmos, first connect the FPGA to your computer.

Configure the FPGA with Patmos, using the command:



## WCET Analysis

To run the WCET analysis of the different neural networks:

```bash
PYTHONPATH="<your-path-to-neuralcasting>/NeuralCasting" python3 wcet/model/<model-name>.py
```

For example, for the NSNet2:

```bash
PYTHONPATH="<your-path-to-neuralcasting>/NeuralCasting" python3 wcet/model/wcet_nsnet2.py
```

## Open Works

## Authors and Contacts

*NeuralCasting* is a project started by [Alessandro Cerioli](https://dk.linkedin.com/in/alessandro-cerioli-26237231) during his PhD at DTU ([Technical University of Denmark](https://www.dtu.dk/english/)) and funded by the European project [Convolve](https://convolve.eu/). For more information regarding the project or to actively contribute to the development of the repository, use the following contacts:

- **DTU email**: alceri@dtu.dk

## License

This project is developed according to *apache 2* license (see *LICENSE*).