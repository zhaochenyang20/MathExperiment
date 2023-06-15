import numpy as np
from scipy.optimize import fsolve

# Define inputs
a = np.array([18.607, 15.841, 20.443, 19.293])
b = np.array([3643.31, 2755.64, 4628.96, 4117.07])
c = np.array([239.73, 219.16, 252.64, 227.44])
Q = np.array(
    [
        1.0,
        0.192,
        2.169,
        1.611,
        0.316,
        1.0,
        0.477,
        0.524,
        0.377,
        0.360,
        1.0,
        0.296,
        0.524,
        0.282,
        2.065,
        1.0,
    ]
).reshape((4, 4))
P = 760

# Define functions
def calculate_Qx(Q, X):
    """Calculates the product of Q and X."""
    return Q @ X


def calculate_y(X, x_T):
    """Calculates y."""
    Qx = calculate_Qx(Q, X)
    return X * (
        b * (1.0 / (x_T[3] + c))
        + np.log(Qx)
        + Q.T @ (X * (1 / Qx))
        - a
        + (np.log(P) - 1)
    )


def equations(x_T):
    """Calculates the system of equations to be solved."""
    X = np.zeros_like(x_T)
    X[0:3] = x_T[0:3]
    X[3] = 1 - np.sum(x_T[0:3])
    return calculate_y(X, x_T)


def solve_equations(x_T_0):
    """Solves the system of equations for x_T."""
    return fsolve(equations, x_T_0)


def print_results(x_T):
    """Prints the results of the calculations."""
    print(
        "[组分]",
        "{:.2f} {:.2f} {:.2f} {:.2f}".format(
            abs(x_T[0] * 100.0),
            abs(x_T[1] * 100.0),
            abs(x_T[2] * 100.0),
            abs((1.0 - np.sum(x_T[0:3])) * 100.0),
        ),
    )
    print("[温度]", round(x_T[3], 2))


# Solve equations and print results
x_T_0s = [
    [1, 0, 0, 60],
    [0, 1, 0, 60],
    [0, 0, 1, 60],
    [0, 0, 0, 60],
    [0.25, 0.25, 0.25, 50],
    [0, 0.33, 0.33, 50],
    [0, 0.5, 0, 50],
    [0.1, 0.2, 0.3, 50],
]
for x_T_0 in x_T_0s:
    x_T = solve_equations(x_T_0)
    print_results(x_T)
