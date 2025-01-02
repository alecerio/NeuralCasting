import os

def mean_square_error(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("The lists must have the same length.")
    mse = sum((float(a) - float(b))**2 for a, b in zip(list1, list2)) / len(list1)
    return mse


debug_outputs = 'debug_outputs_actual/'
file_names = os.listdir(debug_outputs)


for filename in file_names:
    if os.path.isfile(f"debug_outputs_actual/{filename}") and os.path.isfile(f"debug_outputs_expected/{filename}"):
        with open(f"debug_outputs_actual/{filename}", "r") as file:
            actual = file.read()
            actual = actual.replace(" ", "")
            actual = actual.replace("'", "")
            actual = actual.replace("[", "")
            actual = actual.replace("]", "")
            actual = actual.split(',')
        with open(f"debug_outputs_expected/{filename}", "r") as file:
            expected = file.read()
            expected = expected.split(',')
        if len(expected) != len(actual):
            print(f"{filename}: not equal length expected {len(expected)} and actual {len(actual)}")
        else:
            mse = mean_square_error(actual, expected)
            print(f"mean square error for {filename} is {mse}")
    else:
        print(f"file debug_outputs_actual/{filename} not found")        
