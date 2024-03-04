from neural_cast.frontend.common.common import CompilerConfig
from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.parser.ops.constant import Constant
from neural_cast.frontend.common.common import fix_identifier
import math
from neural_cast.frontend.parser.ops.common.common import node_shape
from neural_cast.frontend.parser.ops.common.common import gen_define_connected_output

class Squeeze(OpNode):
    def __init__(self, name : str):
        super().__init__(name)
    
    def __str__(self):
        return super().__str__()
    
    def generate_code(self) -> str:
        parallel : str = CompilerConfig()['parallel']
        name : str = self.get_name()
        define_connected_output : str = gen_define_connected_output(self, 0)
        output_name : str = fix_identifier(self._output_varnames[0])
        input_name : str = fix_identifier(self._input_varnames[0])
        in_shape : int = self.infer_output_shape()
        in_size : int = math.prod(in_shape)

        if parallel == 'omp':
            omp_parallel_for : str = self._gen_omp_parallel_for(input_name, output_name)
        else:
            omp_parallel_for : str = ""

        code : str = self._read_template_c("Squeeze.c")

        code = self._expand_pattern(code, "$NAME", name)
        code = self._expand_pattern(code, "$DEFINE_CONNECTED_OUTPUT", define_connected_output)
        code = self._expand_pattern(code, "$OUTPUT_NAME", output_name)
        code = self._expand_pattern(code, "$INPUT_SIZE", str(in_size))
        code = self._expand_pattern(code, "$INPUT_NAME", input_name)
        code = self._expand_pattern(code, "$OMP_PARALLEL_FOR", omp_parallel_for)

        return code
    
    def generate_includes_code_c(self) -> str:
        return ""

    def generate_declaration_code_c(self) -> str:
        return ""
    
    def infer_output_shape(self) -> list[list[int]]:
        input : Node = self._inputs[0]
        shape : list[int] = node_shape(input)

        if len(self._inputs) == 2:
            axes_input : Node = self._inputs[1]
            if isinstance(axes_input, Constant):
                axes = axes_input._tensor[0]
            else:
                axes = None
        else:
            axes = None
        
        shape_out = []
        if axes == None:
            for dim in shape:
                if dim != 1:
                    shape_out.append(dim)
        else:
            shape_out = shape
            shape_out.pop(axes)
        
        return shape_out
    
    def infer_output_type(self) -> int:
        input : Node = self._inputs[0]
        return input.infer_output_type()
    
    def get_op_type(self) -> str:
        return "Squeeze"
    
    def _gen_omp_parallel_for(self, input_name : str, output_name : str) -> str:
        code : str = '#pragma omp parallel for shared(tensor_' + input_name + ', tensor_' + output_name + ')'
        return code