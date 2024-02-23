from neural_cast.frontend.parser.node.op_node import OpNode
from neural_cast.frontend.parser.node.node import Node
from neural_cast.frontend.common.common import fix_identifier
import math
from neural_cast.frontend.parser.ops.common.common import node_shape
from neural_cast.frontend.parser.ops.common.common import gen_define_connected_output
from neural_cast.frontend.parser.ops.common.common import gen_for_loop_begin
from neural_cast.frontend.parser.ops.common.common import gen_for_loop_end
from neural_cast.frontend.parser.ops.common.common import gen_for_loop_index

class Transpose(OpNode):
    def __init__(self, name : str, perm : list[int]):
        super().__init__(name)
        self.perm = perm
    
    def __str__(self):
        return super().__str__()
    
    def generate_code(self) -> str:
        name : str = self.get_name()
        define_connected_output : str = gen_define_connected_output(self, 0)
        output_name : str = fix_identifier(self._output_varnames[0])
        input_name : str = fix_identifier(self._input_varnames[0])
        out_shape : int = self.infer_output_shape()
        out_size : int = math.prod(out_shape)
        
        input : Node = self._inputs[0]
        in_shape : list[int] = node_shape(input)

        for_loop_begin : str = gen_for_loop_begin(in_shape)
        for_loop_end : str = gen_for_loop_end(in_shape)
        index : str = gen_for_loop_index(in_shape)
        index_out : str = self._gen_index_out()

        code : str = self._read_template_c("Transpose.c")

        code = self._expand_pattern(code, "$NAME", name)
        code = self._expand_pattern(code, "$DEFINE_CONNECTED_OUTPUT", define_connected_output)
        code = self._expand_pattern(code, "$OUTPUT_NAME", output_name)
        code = self._expand_pattern(code, "$INPUT_SIZE", str(out_size))
        code = self._expand_pattern(code, "$FOR_LOOPS_BEGIN", for_loop_begin)
        code = self._expand_pattern(code, "$INDEX_OUT", index_out)
        code = self._expand_pattern(code, "$INDEX_IN", index)
        code = self._expand_pattern(code, "$INPUT_NAME", input_name)
        code = self._expand_pattern(code, "$FOR_LOOPS_END", for_loop_end)

        return code
    
    def generate_includes_code_c(self) -> str:
        return ""

    def generate_declaration_code_c(self) -> str:
        return ""
    
    def infer_output_shape(self) -> list[list[int]]:
        input : Node = self._inputs[0]
        shape : list[int] = node_shape(input)

        shape_out = []
        for i in range(len(shape)):
            shape_out.append(shape[self.perm[i]])

        return shape_out
    
    def infer_output_type(self) -> int:
        input : Node = self._inputs[0]
        return input.infer_output_type()
    
    def get_op_type(self) -> str:
        return "Tranpose"
    
    def _gen_index_out(self) -> str:
        index : str = ""

        out_shape : list[int] = self.infer_output_shape()
        perm : list[int] = self.perm

        for i in range(1, len(perm)):
            ix = perm[i]

            size : str = "(1*"
            for s in range(i+1, len(out_shape)):
                curr_size = out_shape[s]
                size += str(curr_size) + "*"
            size = size[:-1]
            size += ")"

            index += "i" + str(ix) + "*" + size + "+"
        
        index = index[:-1]
        
        return index