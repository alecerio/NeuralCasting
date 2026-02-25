from config.config import PATMOS_OPAPP_PATH, LIB_DIR, WCET_OUT_PATH
import random
import subprocess

def generate_wcet_analysis(unitname, generate_c_code):
    clear_compilation_path()
    generate_makefile(unitname)
    generate_c_code()
    run_wcet_analysis(unitname)

def generate_makefile(name):
    c_files = f"{PATMOS_OPAPP_PATH}/main.c {PATMOS_OPAPP_PATH}/input.c {LIB_DIR}/ncast_lib.c"
    inc_dirs = f"-I {LIB_DIR} -I {PATMOS_OPAPP_PATH}"

    opapp_folder = PATMOS_OPAPP_PATH.split('/')[-1]

    makefile = f"""
all:
	patmos-clang -O2 -mserialize-pml=simple.pml {c_files} -o {name}.elf {inc_dirs} -Wl,--defsym,__heap_end=0x900000,--defsym,_stack_cache_base=0x4000000,-defsym,_shadow_stack_base=0x3fff800 

wcet:
	platin wcet -i simple.pml -b {name}.elf --report

patmos:
	patmos-clang -O2 -mserialize-pml=simple.pml {c_files} -o {opapp_folder}.elf {inc_dirs} -Wl,--defsym,__heap_end=0x900000,--defsym,_stack_cache_base=0x4000000,-defsym,_shadow_stack_base=0x3fff800 
    
"""
    
    with open(f"{PATMOS_OPAPP_PATH}/Makefile", "w") as f:
        f.write(makefile)

def run_wcet_analysis(name):
    _run_make("all", f"{name}_all")
    _run_make("wcet", f"{name}_wcet")

def _run_make(command, name):
    result = subprocess.run(
        ["make", command],
        cwd=f"{PATMOS_OPAPP_PATH}",
        capture_output=True,
        text=True
    )
    print(result.stderr)
    with open(f"{WCET_OUT_PATH}/{name}.txt", "w") as f:
        f.write(result.stdout)

def generate_int32_str():
    n = random.randint(0, 2**31 - 1)
    s = str(n)
    return s

def generate_int16_str():
    n = random.randint(0, 2**16 - 1)
    s = str(n)
    return s

def generate_list_int8_str(size):
    numbers = [random.randint(-128, 127) for _ in range(size)]
    return ",".join(str(n) for n in numbers)

def clear_compilation_path():
    subprocess.run("rm -f *", cwd=f"{PATMOS_OPAPP_PATH}", shell=True)

def generate_main_wcet_template(nn_statement):
    main = f"""
#include "ncast_lib.h"
#include "input.h"

int main() {{
    {nn_statement}
    return 0;
}}

"""
    
    with open(f"{PATMOS_OPAPP_PATH}/main.c", "w") as f:
        f.write(main)

def generate_input_h_template(data):
    declarations = []
    for d in data:
        type: str = d[0]
        name: str = d[1]
        size: int = d[2]
        
        declaration = f"extern volatile {type} {name}[{size}];"
        declarations.append(declaration)
    declarations_code = '\n'.join(declarations)

    input_c = f"""
#include <stdint.h>

{declarations_code}
"""
    
    with open(f"{PATMOS_OPAPP_PATH}/input.h", "w") as f:
        f.write(input_c)

def generate_input_c_template(data):
    definitions = []
    for d in data:
        type: str = d[0]
        name: str = d[1]
        size: int = d[2]
        initialized: bool = d[3]
        
        initialization = ""
        if initialized:
            if type == 'int8_t':
                values = generate_list_int8_str(size)
                initialization = f" = {{ {values} }}"
            else:
                raise(f"type {type} not supported")

        
        definition = f"{type} volatile {name}[{size}]{initialization};"
        definitions.append(definition)
    definitions_code = '\n'.join(definitions)

    input_c = f"""
#include "input.h"

{definitions_code}
"""
    
    with open(f"{PATMOS_OPAPP_PATH}/input.c", "w") as f:
        f.write(input_c)