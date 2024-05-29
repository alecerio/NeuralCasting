import math
from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.common.common import fix_identifier
from neural_cast.frontend.parser.ops.common.common import node_shape
from neural_cast.frontend.parser.ops.common.common import gen_define_connected_output

class MaxPool(OpNode):
    def __init__(self, name : str, kernel_size : int, stride : int):
        super().__init__(name)
        self.kernel_size = kernel_size
        self.stride = stride

    def __str__(self):
        return super.__str__()

    def generate_code(self) -> str:
        #parallel : str = CompilerConfig()['parallel']
        name : str = fix_identifier(self.get_name())
        input_name : str = fix_identifier(self._input_varnames[0])
        output_name : str = fix_identifier(self._output_varnames[0])
        in_height = self._inputs[0].infer_output_shape()[-2]
        in_width = self._inputs[0].infer_output_shape()[-1]
        out_height = int((in_height - self.kernel_size) / self.stride + 1)
        out_width = int((in_width - self.kernel_size) / self.stride + 1)
        out_shape = self.infer_output_shape()
        out_size = math.prod(out_shape)

        #if parallel == 'omp':
        #    for_loop_begin = gen_introduce_omp_in_for_loop_elem_by_elem(for_loop_begin, input_name, output_name)

        code : str = self._read_template_c("MaxPool.c")

        code = self._expand_pattern(code, "$(NAME)", name)
        code = self._expand_pattern(code, "$(OUTPUT_HEIGHT)", str(out_height))
        code = self._expand_pattern(code, "$(OUTPUT_WIDTH)", str(out_width))
        code = self._expand_pattern(code, "$(POOL_HEIGHT)", str(self.kernel_size))
        code = self._expand_pattern(code, "$(POOL_WIDTH)", str(self.kernel_size))
        code = self._expand_pattern(code, "$(STRIDE)", str(self.stride))
        code = self._expand_pattern(code, "$(INPUT_NAME)", input_name)
        code = self._expand_pattern(code, "$(OUTPUT_NAME)", output_name)
        code = self._expand_pattern(code, "$(OUTPUT_SIZE)", str(out_size))
        code = self._expand_pattern(code, "$(INPUT_WIDTH)", str(in_width))

        return code
    
    def generate_declaration_code_c(self) -> str:
        define_connected_output : str = gen_define_connected_output(self, 0)
        output_name : str = fix_identifier(self._output_varnames[0])
        out_shape = self.infer_output_shape()
        out_size = math.prod(out_shape)

        code : str = self._read_template_c("MaxPool_decl.c")

        code = self._expand_pattern(code, "$(DEFINE_CONNECTED_OUTPUT)", define_connected_output)
        code = self._expand_pattern(code, "$(OUTPUT_NAME)", output_name)
        code = self._expand_pattern(code, "$(OUTPUT_SIZE)", str(out_size))

        return code
    
    def infer_output_shape(self) -> list[list[int]]:
        input_shape = self._inputs[0].infer_output_shape()
        n_dims : int = len(input_shape)
        output_shape = []
        for _ in range(0, n_dims): 
            output_shape.append(1)
        in_height = self._inputs[0].infer_output_shape()[-2]
        in_width = self._inputs[0].infer_output_shape()[-1]
        out_height = int((in_height - self.kernel_size) / self.stride + 1)
        out_width = int((in_width - self.kernel_size) / self.stride + 1)
        output_shape[-1] = out_width
        output_shape[-2] = out_height
        return output_shape
    
    def infer_output_type(self) -> int:
        input1 : Node = self._inputs[0]
        bias : Node = self._inputs[2]
        input1_shape = input1.infer_output_shape()
        bias_shape = bias.infer_output_shape()
        print(type(input1_shape))
        return [1, bias_shape[1], input1_shape[2], input1_shape[3]]
    
    def get_op_type(self) -> str:
        return "MaxPool"
    
    def generate_includes_code_c(self) -> str:
        code : str = self._read_template_c("MaxPool_inc.c")
        return code