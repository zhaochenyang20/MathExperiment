import numpy as np
from scipy.optimize import minimize

np.set_printoptions(precision=15)


def mix(vec):
    x = vec[0]
    y = vec[1]
    w = vec[2]
    z_1 = vec[3]
    z_2 = vec[4]
    c = np.zeros(2)
    c[0] = (0.03 * x + 0.01 * y) / (x + y) * w + 0.02 * z_1 - 2.5 / 100 * (w + z_1)
    c[1] = (
        (0.03 * x + 0.01 * y) / (x + y) * (x + y - w)
        + 0.02 * z_2
        - 1.5 / 100 * ((x + y - w) + z_2)
    )
    ceq = []
    return c


def profit1(vec):
    x = vec[0]
    y = vec[1]
    w = vec[2]
    z_1 = vec[3]
    z_2 = vec[4]
    F = 9 * (w + z_1) + 15 * ((x + y - w) + z_2) - 6 * x - 16 * y - 10 * (z_1 + z_2)
    F = -F
    return F


def profit2(vec):
    x = vec[0]
    y = vec[1]
    w = vec[2]
    z_1 = vec[3]
    z_2 = vec[4]
    F = 9 * (w + z_1) + 15 * ((x + y - w) + z_2) - 6 * x - 13 * y - 10 * (z_1 + z_2)
    F = -F
    return F


lb = np.array([0, 0, 0, 0, 0])
ub = np.array([500, 500, 1000, 500, 500])

A = np.array(
    [[0, 0, 0, +1, +1], [-1, -1, 1, 0, 0], [0, 0, +1, +1, 0], [+1, +1, -1, 0, +1]]
)
b1 = np.array([500, 0, 100, 200])
b2 = np.array([500, 0, 600, 200])

ans1 = 0
ansx1 = []
for iter_cnt in range(50):
    x0 = np.random.rand(5) * 100
    res = minimize(
        profit1,
        x0,
        method="SLSQP",
        bounds=list(zip(lb, ub)),
        constraints={"type": "ineq", "fun": lambda x: b1 - np.dot(A, x)},
        options={"maxiter": 1000},
    )
    if -res.fun > ans1:
        ans1 = -res.fun
        ansx1 = res.x

ans2 = 0
ansx2 = []
for iter_cnt in range(50):
    x0 = np.random.rand(5) * 100
    res = minimize(
        profit1,
        x0,
        method="SLSQP",
        bounds=list(zip(lb, ub)),
        constraints={"type": "ineq", "fun": lambda x: b2 - np.dot(A, x)},
        options={"maxiter": 1000},
    )
    if -res.fun > ans2:
        ans2 = -res.fun
        ansx2 = res.x

ans3 = 0
ansx3 = []
for iter_cnt in range(50):
    x0 = np.random.rand(5) * 100
    res = minimize(
        profit2,
        x0,
        method="SLSQP",
        bounds=list(zip(lb, ub)),
        constraints={"type": "ineq", "fun": lambda x: b1 - np.dot(A, x)},
        options={"maxiter": 1000},
    )
    if -res.fun > ans3:
        ans3 = -res.fun
        ansx3 = res.x

ans4 = 0
ansx4 = []
for iter_cnt in range(50):
    x0 = np.random.rand(5) * 100
    res = minimize(
        profit2,
        x0,
        method="SLSQP",
        bounds=list(zip(lb, ub)),
        constraints={"type": "ineq", "fun": lambda x: b2 - np.dot(A, x)},
        options={"maxiter": 1000},
    )
    if -res.fun > ans4:
        ans4 = -res.fun
        ansx4 = res.x

print(ans1)
print(ansx1)
print(ans2)
print(ansx2)
print(ans3)
print(ansx3)
print(ans4)
print(ansx4)
