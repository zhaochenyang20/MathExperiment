import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

list_t = np.array(
    [0.083, 0.167, 0.25, 0.50, 0.75, 1.0, 1.5, 2.25, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
)
list_ct = np.array(
    [10.9, 21.1, 27.3, 36.4, 35.5, 38.4, 34.8, 24.2, 23.6, 15.7, 8.2, 8.3, 2.2, 1.8]
)


def objective(x):
    b = x[0]
    k = x[1]
    k1 = x[2]
    F = np.empty_like(list_t)
    for i in range(14):
        t = list_t[i]
        ct = list_ct[i]
        F[i] = ct - b * k1 / (k1 - k) * (np.exp(-k * t) - np.exp(-k1 * t))
    return F


def compute_loss(x):
    b = x[0]
    k = x[1]
    k1 = x[2]
    tmp_ct = b * k1 / (k1 - k) * (np.exp(-k * list_t) - np.exp(-k1 * list_t))
    return np.sum((tmp_ct - list_ct) ** 2)


x0 = np.random.rand(3)
res = least_squares(objective, x0, bounds=([0, 0, 0], [100, 100, 100]))
x = res.x

resnorm = res.cost
print(x)
print(resnorm)

final_loss = compute_loss(x)
b = x[0]
k = x[1]
k1 = x[2]
x = np.arange(0, 12.01, 0.01)
y = b * k1 / (k1 - k) * (np.exp(-k * x) - np.exp(-k1 * x))
print(f"final_loss: {final_loss}")
print(f"b: {b}")
print(f"k: {k}")
print(f"k1: {k1}")
plt.plot(x, y)
plt.xlabel("time")
plt.ylabel("concentration")
plt.show()
