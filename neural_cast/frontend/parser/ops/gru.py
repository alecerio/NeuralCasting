from neural_cast.frontend.parser.node.op_node import OpNode
import math

class Gru(OpNode):
    def __init__(self, name : str):
        super().__init__(name)
    
    def __str__(self):
        return super.__str__()
    
    def generate_code(self) -> str:
        code : str = ""

        print(" { ------------------------- ")
        for i in range(len(self._input_varnames)):
            print(self._input_varnames[i], " , ", self._inputs[i].infer_output_shape())
        print(" ------------------------- } ")

        
        code += self._gen_code_WirXtBir()

        code_WhrHt1Bhr : str = self._read_template_c("Gemm.c")
        code_arg0 : str = self._read_template_c("Add.c")
        code_rt : str = self._read_template_c("Sigmoid.c")

        code_WizXtBiz : str = self._read_template_c("Gemm.c")
        code_WhzHt1Bhz : str = self._read_template_c("Gemm.c")
        code_arg1 : str = self._read_template_c("Add.c")
        code_zt : str = self._read_template_c("Sigmoid.c")

        code_WinXtBin : str = self._read_template_c("Gemm.c")
        code_WhnHt1Bhn : str = self._read_template_c("Gemm.c")
        code_rt_WhnHt1Bhn : str = self._read_template_c("Mul.c")
        code_arg2 : str = self._read_template_c("Add.c")
        code_nt : str = self._read_template_c("Tanh.c")

        return code
    
    def generate_declaration_code_c(self) -> str:
        return ""
    
    def generate_includes_code_c(self) -> str:
        return ""
    
    def get_op_type(self) -> str:
        return "Gru"
    
    def _gen_code_WirXtBir(self) -> str:
        _name : str = self.get_name() + "_WirXtBir"
        x_shape = self._inputs[0].infer_output_shape()
        x_size : int = math.prod(x_shape)
        x_name : str = self._input_varnames[0]
        h_shape = self._inputs[5].infer_output_shape()
        h_size : int = math.prod(h_shape)
        w_name : str = self._input_varnames[1]
        b_name : str = self._input_varnames[3]
        output_name : str =  self.get_name() + "_WirXtBir"
        output_init_code : str = self._gen_output_init_code(h_size)

        code_WirXtBir : str = self._read_template_c("Gemm.c")
        code_WirXtBir : str = self._expand_pattern(code_WirXtBir, "$DEFINE_CONNECTED_OUTPUT", "#define CONNECTED_OUTPUT")
        code_WirXtBir = self._expand_pattern(code_WirXtBir, "$INPUT_SIZE", str(x_size))
        code_WirXtBir = self._expand_pattern(code_WirXtBir, "$OUTPUT_SIZE", str(h_size))
        code_WirXtBir = self._expand_pattern(code_WirXtBir, "$NAME", _name)
        code_WirXtBir = self._expand_pattern(code_WirXtBir, "$INPUT_NAME_X", x_name)
        code_WirXtBir = self._expand_pattern(code_WirXtBir, "$INPUT_NAME_W", w_name)
        code_WirXtBir = self._expand_pattern(code_WirXtBir, "$INPUT_NAME_B", b_name)
        code_WirXtBir = self._expand_pattern(code_WirXtBir, "$OUTPUT_NAME", output_name)
        code_WirXtBir = self._expand_pattern(code_WirXtBir, "$OUTPUT_INIT", output_init_code)
        
        return code_WirXtBir
    
    def _gen_output_init_code(self, out_size : int) -> str:
        output_init_code = ""

        for _ in range(out_size):
            output_init_code = output_init_code + "0.0f, "
        
        return output_init_code

    



        