from wcet.common.common import clear_compilation_path, generate_makefile
from config.config import PATMOS_PATH, PATMOS_OPAPP_PATH, PATMOS_OUT_PATH
import subprocess

def generate_patmos_analysis(unitname, generate_c_code):
    clear_compilation_path()
    generate_makefile(unitname)
    generate_c_code()
    run_patmos_analysis(unitname)

def run_patmos_analysis(name):
    _compile_elf(name)
    _run_on_fpga(name)

def _compile_elf(name):
    result = subprocess.run(
        ["make", "patmos"],
        cwd=f"{PATMOS_OPAPP_PATH}",
        capture_output=True,
        text=True
    )
    print(result.stderr)
    with open(f"{PATMOS_OUT_PATH}/{name}_patmos.txt", "w") as f:
        f.write(result.stdout)

def _run_on_fpga(name):
    result = subprocess.run(
        ["make", "BOARD=altde2-115-sdram", "APP=nc-ops", "config", "app", "download"],
        cwd=f"{PATMOS_PATH}",
        capture_output=True,
        text=True
    )
    print(result.stderr)
    with open(f"{PATMOS_OUT_PATH}/{name}_patmos_out.txt", "w") as f:
        f.write(result.stdout)

def generate_main_patmos_template(nn_statement, n_exps):
    main = f"""
#include "ncast_lib.h"
#include "input.h"
#include "machine/rtc.h"
#include <stdio.h>

#define NUM_EXPS ({n_exps})

int main() {{
    unsigned long long tot = 0;
    for(int i=0; i<NUM_EXPS; i++) {{
        unsigned long long start = get_cpu_cycles();
        {nn_statement}
        unsigned long long end = get_cpu_cycles();
        tot += (end-start);
    }}
    unsigned long long avg = tot / NUM_EXPS;
    printf("cpu-cycles:%llu", avg);
    return 0;
}}

"""
    
    with open(f"{PATMOS_OPAPP_PATH}/main.c", "w") as f:
        f.write(main)
