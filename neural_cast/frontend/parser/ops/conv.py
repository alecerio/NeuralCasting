import math
from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.common.common import fix_identifier
from neural_cast.frontend.parser.ops.common.common import node_shape
from neural_cast.frontend.parser.ops.common.common import gen_define_connected_output

class Conv(OpNode):
    def __init__(self, name : str, kernel_size : int, padding : int, stride : int):
        super().__init__(name)
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def __str__(self):
        return super.__str__()

    def generate_code(self) -> str:
        #parallel : str = CompilerConfig()['parallel']
        name : str = fix_identifier(self.get_name())
        [_, input_channels, input_height, input_width] = self._inputs[0].infer_output_shape()
        output_channels = self._inputs[2].infer_output_shape()[-1]
        kernel_size : int = self.kernel_size
        padding : int = self.padding
        stride : int = self.stride
        output_name : str = fix_identifier(self._output_varnames[0])
        input_name : str = fix_identifier(self._input_varnames[0])
        weights_name : str = fix_identifier(self._input_varnames[1])
        bias_name : str = fix_identifier(self._input_varnames[2])
        output_height : int = int((input_height - kernel_size + 2 * padding) / stride + 1)
        output_width : int = int((input_width - kernel_size + 2 * padding) / stride + 1)
        output_size = output_channels * output_width * output_height
        

        #if parallel == 'omp':
        #    for_loop_begin = gen_introduce_omp_in_for_loop_elem_by_elem(for_loop_begin, input_name, output_name)

        code : str = self._read_template_c("Conv.c")

        code = self._expand_pattern(code, "$(NAME)", name)
        code = self._expand_pattern(code, "$(INPUT_NAME)", input_name)
        code = self._expand_pattern(code, "$(OUTPUT_NAME)", output_name)
        code = self._expand_pattern(code, "$(WEIGHTS_NAME)", weights_name)
        code = self._expand_pattern(code, "$(BIASES_NAME)", bias_name)
        code = self._expand_pattern(code, "$(INPUT_HEIGHT)", str(input_height))
        code = self._expand_pattern(code, "$(KERNEL_SIZE)", str(kernel_size))
        code = self._expand_pattern(code, "$(PADDING)", str(padding))
        code = self._expand_pattern(code, "$(STRIDE)", str(stride))
        code = self._expand_pattern(code, "$(INPUT_WIDTH)", str(input_width))
        code = self._expand_pattern(code, "$(OUTPUT_CHANNELS)", str(output_channels))
        code = self._expand_pattern(code, "$(OUTPUT_HEIGHT)", str(output_height))
        code = self._expand_pattern(code, "$(OUTPUT_WIDTH)", str(output_width))
        code = self._expand_pattern(code, "$(INPUT_CHANNELS)", str(input_channels))
        code = self._expand_pattern(code, "$(OUTPUT_SIZE)", str(output_size))

        return code
    
    def generate_declaration_code_c(self) -> str:
        define_connected_output : str = gen_define_connected_output(self, 0)
        output_name : str = fix_identifier(self._output_varnames[0])
        output_channels = self._inputs[2].infer_output_shape()[-1]
        [_, _, input_height, input_width] = self._inputs[0].infer_output_shape()
        output_height : int = int((input_height - self.kernel_size + 2 * self.padding) / self.stride + 1)
        output_width : int = int((input_width - self.kernel_size + 2 * self.padding) / self.stride + 1)
        output_size = output_channels * output_width * output_height

        code : str = self._read_template_c("Conv_decl.c")

        code = self._expand_pattern(code, "$(DEFINE_CONNECTED_OUTPUT)", define_connected_output)
        code = self._expand_pattern(code, "$(OUTPUT_NAME)", output_name)
        code = self._expand_pattern(code, "$(OUTPUT_SIZE)", str(output_size))

        return code
    
    def infer_output_shape(self) -> list[list[int]]:
        input : Node = self._inputs[0]
        shape : list[int] = node_shape(input)
        return shape
    
    def infer_output_type(self) -> int:
        input1 : Node = self._inputs[0]
        bias : Node = self._inputs[2]
        input1_shape = input1.infer_output_shape()
        bias_shape = bias.infer_output_shape()
        print(type(input1_shape))
        return [1, bias_shape[1], input1_shape[2], input1_shape[3]]
    
    def get_op_type(self) -> str:
        return "Conv"
    
    def generate_includes_code_c(self) -> str:
        return ""