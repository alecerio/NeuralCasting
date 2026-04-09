import re
import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np

#pattern_wcet = re.compile(r"cache-max-cycles:\s*(\d+)")
#pattern_wcet = re.compile(r"^\s*cycles:\s*(\d+)\s*$")
pattern_wcet = re.compile(r"(?m)^\s*cycles:\s*(-?\d+)\s*$")
pattern_patmos = re.compile(r"cpu-cycles:\s*(\d+)")

class RecordWCET():
    def __init__(self, name: str, type: str, cycles_wcet: int = None, cycles_patmos: int = None):
        self.name = name
        self.type = type
        self.cycles_wcet = cycles_wcet
        self.cycles_patmos = cycles_patmos
    
    def __str__(self):
        return f"{self.name};{self.type};{self.cycles_wcet};{self.cycles_patmos}"

def retrieve_cpu_cycles_wcet(full_directory: str, records : List[RecordWCET]):
    with open(full_directory, "r", encoding="utf-8") as f:
        content = f.read()

    #matches = pattern_wcet.findall(content)
    type = full_directory.split('/')[-1].split('.')[0].split('-')[0]
    name = full_directory.split('/')[-1].split('.')[0].split('-')[1][:-5]

    #if matches:
    #    for i, value in enumerate(matches, start=1):
    #        records.append(RecordWCET(name, type, int(value)))
    #else:
    #    print("No matches")

    match = re.search(r"^\s*cycles:\s*(\d+)\s*$", content, re.MULTILINE)

    if match:
        value = int(match.group(1))
        records.append(RecordWCET(name, type, int(value)))
    else:
        print(name)
        print("Nessun match")

def retrieve_cpu_cycles_patmos(full_directory: str, records : List[RecordWCET]):
    with open(full_directory, "r", encoding="utf-8") as f:
        content = f.read()

    matches = pattern_patmos.findall(content)
    type = full_directory.split('/')[-1].split('.')[0].split('-')[0]
    name = full_directory.split('/')[-1].split('.')[0].split('-')[1][:-11]

    

    if matches:
        for i, value in enumerate(matches, start=1):
            found = False
            for record in records:
                if record.name == name and record.type == type:
                    record.cycles_patmos = int(value)
                    found = True
                    break
            if found == False:
                pass
                #raise Exception("WCET record not found.")

    else:
        print("No matches")
        print(name)

def get_analysis_files(directory_wcet : str, directory_patmos : str, records : List[RecordWCET]):
    files = [
        f for f in os.listdir(directory_wcet)
        if os.path.isfile(os.path.join(directory_wcet, f))
    ]

    for file in files:
        filename = file.split('.')[0]
        if filename[-5:] == "_wcet":
            retrieve_cpu_cycles_wcet(f"{directory_wcet}/{file}", records)
    
    files = [
        f for f in os.listdir(directory_patmos)
        if os.path.isfile(os.path.join(directory_patmos, f))
    ]

    for file in files:
        filename = file.split('.')[0]
        retrieve_cpu_cycles_patmos(f"{directory_patmos}/{file}", records)


def plot_analysis_bar(records : List[RecordWCET]):
    operators = []
    measured = []
    wcet = []
    
    for record in records:
        operators.append(record.type)
        measured.append(record.cycles_patmos)
        wcet.append(record.cycles_wcet)

    x = np.arange(len(operators))
    width = 0.35

    plt.figure(figsize=(10,5))
    plt.bar(x - width/2, measured, width, label="Measured")
    plt.bar(x + width/2, wcet, width, label="WCET")

    plt.xticks(x, operators, rotation=90)
    plt.ylabel("CPU cycles")
    plt.yscale("log")
    plt.legend()

    plt.tight_layout()
    plt.show()


def add_prefix_records(records : List[RecordWCET], prefix : str):
    for record in records:
        record.name = f"{prefix}_{record.name}"

def plot_scatterplot(records : List[RecordWCET]):

    operators = []
    types = []

    measured_list = []
    wcet_list = []

    for record in records:
        operators.append(record.name)
        types.append(record.type)
        measured_list.append(record.cycles_patmos)
        wcet_list.append(record.cycles_wcet)

    measured = np.array(measured_list)
    wcet = np.array(wcet_list)

    ratio = measured / wcet
    x = np.arange(len(operators))

    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
        "magenta",
        "yellow",
        "black"
    ]

    color_map = {}
    set_types = set(types)
    for idx, type in enumerate(set_types):
        color_map[type] = colors[idx]
    
    print(color_map)
    print(len(set_types))
    colors_ = [color_map[t] for t in types]

    plt.figure(figsize=(10,5))

    plt.scatter(x, ratio, c=colors_)

    plt.axhline(1.0, linestyle="--")  # limite ideale

    plt.ylabel("Measured / WCET")
    plt.xlabel("Operator")
    plt.yscale("log")
    plt.xticks(x, operators, rotation=90)

    for t, c in color_map.items():
        plt.scatter([], [], c=c, label=t)
    plt.legend(title="Operator type")

    plt.tight_layout()
    plt.show()

def total_cycles_measured(records : List[RecordWCET]):
    tot_cycles = 0
    for record in records:
        tot_cycles += record.cycles_patmos
    return tot_cycles

def total_cycles_wcet(records : List[RecordWCET]):
    tot_cycles = 0
    for record in records:
        tot_cycles += record.cycles_wcet
    return tot_cycles

def plot_cycles_bar(records: List[RecordWCET]):

    operators = []
    types = []
    cycles_list = []

    for record in records:
        operators.append(record.name)
        types.append(record.type)
        cycles_list.append(record.cycles_patmos)

    cycles = np.array(cycles_list)
    x = np.arange(len(operators))

    colors = [
        "blue", "red", "green", "orange", "purple",
        "brown", "pink", "gray", "olive", "cyan",
        "magenta", "yellow", "black"
    ]

    unique_types = sorted(set(types))
    color_map = {t: colors[i] for i, t in enumerate(unique_types)}

    bar_colors = [color_map[t] for t in types]

    plt.figure(figsize=(12,6))

    plt.bar(x, cycles, color=bar_colors)

    plt.ylabel("Measured cycles")
    plt.yscale("log")
    plt.xlabel("Operator")
    plt.xticks(x, operators, rotation=90)

    # legenda
    for t, c in color_map.items():
        plt.bar(0, 0, color=c, label=t)

    plt.legend(title="Operator type")

    plt.tight_layout()
    plt.show()

def plot_combined(records: List[RecordWCET]):

    operators = []
    types = []
    measured_list = []
    wcet_list = []

    for record in records:
        if record.type != 'Unsqueeze':
            operators.append(record.name[10:])
            types.append(record.type)
            measured_list.append(record.cycles_patmos)
            wcet_list.append(record.cycles_wcet)


    REMOVE = [
        "_tcm_", "tcm_", "_tcm",
        "_conv1x1_", "_conv1x1", "conv1x1_",
        "_quant", "_PRelu", "_frontend_", "kend_", "ntend_", "tcn_"
    ]

    for idx, op in enumerate(operators):
        for r in REMOVE:
            op = op.replace(r, "")
        operators[idx] = op
    
    print("\n ------------------- \n")
    print(operators)

    op_sorted = ["MatMul_2","MatMul_4","MatMul_6","MatMul_8","MatMul_10","MatMul_12","MatMul","Add_2","Add_4","Add_6","Add_12","Add_14",
                 "Add_16","Add","MatMul_1","MatMul_3","MatMul_5","Add_1","Add_3","Add_5","Add_7","Add_8","Mul","Mul_2","Add_9","Sub","Tanh",
                 "Mul_1","Add_10","MatMul_7","MatMul_9","MatMul_11","Add_11","Add_13","Add_15","Add_17","Add_18","Mul_3","Mul_5","Add_19","Sub_1",
                 "Tanh_1","Mul_4","Add_20","MatMul_13","Add_21","MatMul_14","Add_22","MatMul_15","Add_23"]
    print("\n ------------------- \n")
    print(op_sorted)

    idx_map = {op: i for i, op in enumerate(operators)}
    perm = [idx_map[op] for op in op_sorted]
    operators = [operators[i] for i in perm]
    measured_list = [measured_list[i] for i in perm]
    wcet_list = [wcet_list[i] for i in perm]
    types = [types[i] for i in perm]

    measured = np.array(measured_list)
    wcet = np.array(wcet_list)
    cycles = measured

    ratio = measured / wcet
    x = np.arange(len(operators))

    colors = [
        "blue","red","green","orange","purple",
        "brown","pink","gray","olive","cyan",
        "magenta","yellow","black"
    ]

    unique_types = sorted(set(types))
    color_map = {t: colors[i] for i, t in enumerate(unique_types)}

    colors_ = [color_map[t] for t in types]

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12,8),
        sharex=True,
        gridspec_kw={"height_ratios":[2,1]}
    )

    # ---- BAR PLOT ----
    ax1.bar(x, cycles, color=colors_)
    ax1.set_ylabel("Measured cycles", fontsize=20)
    ax1.set_yscale("log")

    # legenda
    for t, c in color_map.items():
        ax1.bar(0, 0, color=c, label=t)

    ax1.legend(title="Operator type")

    # ---- SCATTER PLOT ----
    ax2.scatter(x, ratio, c=colors_)
    ax2.axhline(1.0, linestyle="--")

    ax2.set_ylabel("Measured / WCET", fontsize=20)
    ax2.set_xlabel("Operator", fontsize=20)
    ax2.set_yscale("log")

    ax2.set_xticks(x)
    ax2.set_xticklabels(operators, rotation=90, fontsize=18)

    ax1.legend(fontsize=16)
    #plt.yticks(fontsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)

    plt.tight_layout()
    plt.show()

from typing import List
import numpy as np
import matplotlib.pyplot as plt

def plot_combined2(records_left: List[RecordWCET], records_right: List[RecordWCET], title_left, title_right):

    def prepare_data(records: List[RecordWCET], shift_name):
        operators = []
        types = []
        measured_list = []
        wcet_list = []

        for record in records:
            if record.type != 'Unsqueeze':
                opname = record.name[shift_name:]
                if opname == "scaled_dot_product_attention_quant":
                    opname = "attention_quant"
                operators.append(opname)
                types.append(record.type)
                measured_list.append(record.cycles_patmos)
                wcet_list.append(record.cycles_wcet)

        measured = np.array(measured_list)
        wcet = np.array(wcet_list)
        ratio = measured / wcet
        x = np.arange(len(operators))

        return operators, types, measured, wcet, ratio, x

    operators_l, types_l, measured_l, wcet_l, ratio_l, x_l = prepare_data(records_left, 15)
    operators_r, types_r, measured_r, wcet_r, ratio_r, x_r = prepare_data(records_right, 19)

    colors = [
        "blue", "red", "green", "orange", "purple",
        "brown", "pink", "gray", "olive", "cyan",
        "magenta", "yellow", "black"
    ]

    all_types = sorted(set(types_l) | set(types_r))
    color_map = {t: colors[i % len(colors)] for i, t in enumerate(all_types)}

    colors_l = [color_map[t] for t in types_l]
    colors_r = [color_map[t] for t in types_r]

    fig, axs = plt.subplots(
        2, 2,
        figsize=(16, 8),
        sharex='col',
        gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1 = axs[0, 0]
    ax2 = axs[1, 0]
    ax3 = axs[0, 1]
    ax4 = axs[1, 1]

    # ---- LEFT COLUMN ----
    ax1.bar(x_l, measured_l, color=colors_l)
    ax1.set_ylabel("Measured cycles", fontsize=20)
    ax1.set_yscale("log")
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    ax1.set_title(title_left, fontsize=20)

    for t, c in color_map.items():
        ax1.bar(0, 0, color=c, label=t)

    ax1.legend(title="Operator type")

    ax2.scatter(x_l, ratio_l, c=colors_l)
    ax2.axhline(1.0, linestyle="--")
    ax2.set_ylabel("Measured / WCET", fontsize=20)
    ax2.set_xlabel("Operator", fontsize=20)
    ax2.set_yscale("log")
    ax2.set_xticks(x_l)
    ax2.tick_params(axis='x')
    ax2.tick_params(axis='y')
    ax2.set_xticklabels(operators_l, rotation=90, fontsize=18)

    # ---- RIGHT COLUMN ----
    ax3.bar(x_r, measured_r, color=colors_r)
    ax3.set_ylabel("Measured cycles", fontsize=20)
    ax3.set_yscale("log")
    ax3.tick_params(axis='x')
    ax3.tick_params(axis='y')
    ax3.set_title(title_right, fontsize=20)

    ax4.scatter(x_r, ratio_r, c=colors_r)
    ax4.axhline(1.0, linestyle="--")
    ax4.set_ylabel("Measured / WCET", fontsize=20)
    ax4.set_xlabel("Operator", fontsize=20)
    ax4.set_yscale("log")
    ax4.set_xticks(x_r)
    ax4.tick_params(axis='x')
    ax4.tick_params(axis='y')
    ax4.set_xticklabels(operators_r, rotation=90, fontsize=18)

    ax1.tick_params(axis='y', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)
    ax3.tick_params(axis='y', labelsize=18)
    ax4.tick_params(axis='y', labelsize=18)
    ax1.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    return ax1, ax2, ax3, ax4

def generate_table(records: List[RecordWCET]):
    rows = []
    header = "name;type;wcet-cycles;patmos-cycles;ratio"
    rows.append(header)
    for record in records:
        name = record.name
        type = record.type
        wcetc = record.cycles_wcet
        patc = record.cycles_patmos
        ratio = patc / wcetc
        row = f"{name};{type};{wcetc};{patc};{ratio}"
        rows.append(row)
    result = "\n".join(rows)
    return result

if __name__ == "__main__":
    records_nsnet2 : List[RecordWCET] = []
    base = "/home/alessandro/Desktop/new-experiments/"
    get_analysis_files(f"{base}/nsnet2/wcet", f"{base}/nsnet2/patmos", records_nsnet2)
    add_prefix_records(records_nsnet2, "nsnet2")

    records_tcn : List[RecordWCET] = []
    get_analysis_files(f"{base}/tcn/wcet", f"{base}/tcn/patmos", records_tcn)
    add_prefix_records(records_tcn, "tcn")

    records_ops : List[RecordWCET] = []
    get_analysis_files(f"{base}/ops/wcet", f"{base}/ops/patmos", records_ops)
    add_prefix_records(records_ops, "ops")

    records_resnet8 : List[RecordWCET] = []
    get_analysis_files(f"{base}/resnet8/wcet", f"{base}/resnet8/patmos", records_resnet8)
    add_prefix_records(records_resnet8, "resnet8")

    records_transformer : List[RecordWCET] = []
    get_analysis_files(f"{base}/transformer/wcet", f"{base}/transformer/patmos", records_transformer)
    add_prefix_records(records_transformer, "transformer")

    #plot_scatterplot(records_tcn)
    #plot_cycles_bar(records_tcn)
    
    #plot_combined(records_tcn)
    plot_combined(records_nsnet2)
    #plot_combined(records_ops)
    #plot_combined(records_resnet8)
    #plot_combined(records_transformer)
    plot_combined2(records_resnet8, records_transformer, "ResNet-8", "Transformer Encoder Layer")

    fig, axs = plt.subplots(2, 2, sharex='col', figsize=(10, 6))

    ax1 = axs[0, 0]
    ax2 = axs[1, 0]

    ax3 = axs[0, 1]
    ax4 = axs[1, 1]

    print("total cycles nsnet2")
    print(total_cycles_measured(records_nsnet2))
    print(total_cycles_wcet(records_nsnet2))
    print(" ----------------------- ")
    print()

    print("total cycles TCN")
    print(total_cycles_measured(records_tcn))
    print(total_cycles_wcet(records_tcn))
    print(" ----------------------- ")
    print()

    print("table nsnet2")
    nsnet2_table = generate_table(records_nsnet2)
    print(nsnet2_table)
    print(" ----------------------- ")
    print()

    print("table tcn")
    tcn_table = generate_table(records_tcn)
    print(tcn_table)
    print(" ----------------------- ")
    print()

    print("table ops")
    ops_table = generate_table(records_ops)
    print(ops_table)
    print(" ----------------------- ")
    print()

    print("table resnet8")
    resnet8_table = generate_table(records_resnet8)
    print(resnet8_table)
    print(" ----------------------- ")
    print()

    print("table transformer")
    transformer_table = generate_table(records_transformer)
    print(transformer_table)
    print(" ----------------------- ")
    print()

    n_cycles = 0
    for rnr in records_transformer:
        n_cycles += rnr.cycles_patmos
    print(f"Cycles: {n_cycles}")
