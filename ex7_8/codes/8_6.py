import numpy as np
from scipy.optimize import linprog

def _print_(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    k = x[5]
    print(f"x1: {x1}, x2: {x2}, x3: {x3}, x4: {x4}, x5: {x5}, k: {k}")

A = np.array([[0, -1, -1, -1, 0, 0],
              [2 - 1.4, 2 - 1.4, 1 - 1.4, 1 - 1.4, 5 - 1.4, 0],
              [9 - 5, 15 - 5, 4 - 5, 3 - 5, 2 - 5, 0],
              [1, 1, 1, 1, 1, 0],
              [0, 0, 0, 0, 0, -1]])
b = np.array([-400, 0, 0, 1000, 0])
lb = np.array([0, 0, 0, 0, 0, 0])
f = -np.array([4.3, 5.4 * 0.5, 5.0 * 0.5, 4.4 * 0.5, 4.5, 0]) / 100

# First optimization problem
res1 = linprog(f, A, b, bounds=list(zip(lb, [None]*6)))
_print_(res1.x)
print("interest: ", -res1.fun)

# Second optimization problem
constrain = A.copy()
constrain[0, 5] = 0.0
constrain[1, 5] = 0.0
constrain[2, 5] = 0.0
constrain[3, 5] = -1.0
lb[5] = 0.0
ub = [1000, 1000, 1000, 1000, 1000, 100]
res = linprog(f, constrain, b, bounds=list(zip(lb, ub)))
_print_(res.x)
print("interest: ", -res.fun)

# Third optimization problem
f2 = f.copy()
f2[0] = -4.5 / 100
res = linprog(f2, A, b, bounds=list(zip(lb, [None]*6)))
_print_(res.x)
print("interest: ", -res.fun)

# Fourth optimization problem
f3 = f.copy()
f3[2] = -4.8 * 0.5 / 100
res = linprog(f3, A, b, bounds=list(zip(lb, [None]*6)))
_print_(res.x)
print("interest: ", -res.fun)