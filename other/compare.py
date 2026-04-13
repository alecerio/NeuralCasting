import numpy as np

a = np.load("other/activation_268.npy")
b = np.load("other/output_268.npy")

mse = np.mean((a - b) ** 2)
print(mse)