import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

d = 100
v1 = 1.0
v2 = 2.0
k = v1 / v2


def dx_dy(pos, y):
    x, y = pos
    dx_dy = [(v1 * np.sqrt(x ** 2 + (d - y) ** 2) - v2 * x) / (v2 * (d - y)), 1]
    return dx_dy


def dx_dy_2(pos, y):
    x = pos[0]
    dx_dy = [(v1 * np.sqrt(x ** 2 + (d - y) ** 2) - v2 * x) / (v2 * (d - y)), 1]
    return dx_dy


y = np.linspace(0, 100, 1000)
pos_init = [0, 0]
pos_new = odeint(dx_dy, pos_init, y)
x_pos = pos_new[:, 0]

analytic_x = (d - y) / 2.0 * (d ** k * (d - y) ** (-k) - d ** (-k) * (d - y) ** k)
plt.plot(x_pos, y, analytic_x, y, "r--", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["numerical solution", "analytic solution"])
plt.show()
