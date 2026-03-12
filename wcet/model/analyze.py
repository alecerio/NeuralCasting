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
    ax1.set_ylabel("Measured cycles")
    ax1.set_yscale("log")

    # legenda
    for t, c in color_map.items():
        ax1.bar(0, 0, color=c, label=t)

    ax1.legend(title="Operator type")

    # ---- SCATTER PLOT ----
    ax2.scatter(x, ratio, c=colors_)
    ax2.axhline(1.0, linestyle="--")

    ax2.set_ylabel("Measured / WCET")
    ax2.set_xlabel("Operator")
    ax2.set_yscale("log")

    ax2.set_xticks(x)
    ax2.set_xticklabels(operators, rotation=90)

    plt.tight_layout()
    plt.show()

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

    #plot_scatterplot(records_tcn)
    #plot_cycles_bar(records_tcn)
    plot_combined(records_tcn)
    plot_combined(records_nsnet2)
    plot_combined(records_ops)

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


