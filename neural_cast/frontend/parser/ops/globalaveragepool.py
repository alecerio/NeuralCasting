import math
from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.common.common import fix_identifier
from neural_cast.frontend.parser.ops.common.common import gen_define_connected_output
from neural_cast.frontend.common.common import CompilerConfig

class GlobalAveragePool(OpNode):
    def __init__(self, name : str):
        super().__init__(name)

    def __str__(self):
        return super.__str__()

    def generate_code(self) -> str:
        parallel : str = CompilerConfig()['parallel']
        name : str = fix_identifier(self.get_name())
        output_name : str = fix_identifier(self._output_varnames[0])
        input_name : str = fix_identifier(self._input_varnames[0])
        [N, C, H, W] = self._inputs[0].infer_output_shape()
        out_shape = self.infer_output_shape()
        out_size = math.prod(out_shape)

        if parallel == 'omp':
            omp_pragma1 : str = '#pragma omp parallel for collapse(2)'

        code : str = self._read_template_c("GlobalAveragePool.c")

        code = self._expand_pattern(code, "$(NAME)", name)
        code = self._expand_pattern(code, "$(INPUT_NAME)", input_name)
        code = self._expand_pattern(code, "$(OUTPUT_NAME)", output_name)
        code = self._expand_pattern(code, "$(N)", str(N))
        code = self._expand_pattern(code, "$(C)", str(C))
        code = self._expand_pattern(code, "$(H)", str(H))
        code = self._expand_pattern(code, "$(W)", str(W))
        code = self._expand_pattern(code, "$(OUTPUT_SIZE)", str(out_size))
        
        if parallel == 'omp':
            code = self._expand_pattern(code, "$(OMP_PRAGMA1)", omp_pragma1)
        else:
            code = self._expand_pattern(code, "$(OMP_PRAGMA1)", '')

        return code
    
    def generate_declaration_code_c(self) -> str:
        name : str = fix_identifier(self.get_name())
        define_connected_output : str = gen_define_connected_output(self, 0)
        output_name : str = fix_identifier(self._output_varnames[0])
        out_shape = self.infer_output_shape()
        out_size = math.prod(out_shape)

        code : str = self._read_template_c("GlobalAveragePool_decl.c")

        code = self._expand_pattern(code, "$(NAME)", name)
        code = self._expand_pattern(code, "$(DEFINE_CONNECTED_OUTPUT)", define_connected_output)
        code = self._expand_pattern(code, "$(OUTPUT_NAME)", output_name)
        code = self._expand_pattern(code, "$(OUTPUT_SIZE)", str(out_size))

        return code
    
    def infer_output_shape(self) -> list[list[int]]:
        in_shape = self._inputs[0].infer_output_shape()
        out_shape = [in_shape[0], in_shape[1], 1, 1]
        return out_shape
    
    def infer_output_type(self) -> int:
        input : Node = self._inputs[0]
        return input.infer_output_type()
    
    def get_op_type(self) -> str:
        return "GlobalAveragePool"
    
    def generate_includes_code_c(self) -> str:
        return ""