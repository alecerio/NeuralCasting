import numpy as np


a = np.array([[-0.75, -0.5, 0.0], [0.25, 0.5, 0.75]])
b = np.array([[0.75, 0.5], [0.0, -0.25], [-0.5, -0.75]])

c = np.matmul(a, b)

print(c)