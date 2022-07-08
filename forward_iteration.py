#forward interation last eigenform (largest eigenvalue)
import math
import numpy as np
from numpy.linalg import inv

ro = 7850
A = 0.05
L = 1
E = 210000000000
mass_matrix_coeff = (ro*A*L)/6
# M = mass_matrix_coeff * np.array([[2,1],[1,2]])
# K = np.array([[131250000, -65625000],[-65625000,-34375000]])
M = np.array([[6.5416, 1.6354],[1.6354,22.8958]])
K = np.array([[525000000, -262500000],[-262500000,362500000]])
x1 = np.array([[1],[1]])
TOL = 0.00001
calculated_tolerance = 1000
initial_eigenvalue = 1
y1 = np.matmul(K, x1)
while TOL < calculated_tolerance:
    x_bar = np.matmul(inv(M), y1)
    y_bar = np.matmul(K, x_bar)
    y = np.matmul(y_bar, 1 / (np.sqrt(np.matmul(x_bar.transpose(), y1))))
    eigenvalue = np.matmul(x_bar.transpose(), y_bar) / np.matmul(x_bar.transpose(), y1)
    calculated_tolerance = abs((eigenvalue - initial_eigenvalue) / (eigenvalue))
    y1 = y.copy()
    initial_eigenvalue = eigenvalue.copy()
T = 2*math.pi /(np.sqrt(eigenvalue))
print(T)
