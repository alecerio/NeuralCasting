

with open("nsnet_out.txt", "r") as file:
    content = file.read()

contents = content.split(" ########################### ")

for subcont in contents:
    sep = subcont.split("\n")
    if len(sep) == 4:
        name = sep[1]
        val = sep[2]
        vals = val.split(" ")
        vals = vals[:-1]
        with open("debug_outputs_actual/" + name + ".txt", "w") as file:
            file.write(str(vals))
        