import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

d = 100
v1 = 2.0
v2 = 2.0
k = v1 / v2


def calc(pos, t):
    len = np.sqrt(pos[0] ** 2.0 + (d - pos[1]) ** 2.0)
    dpos_dt = [v1 - v2 * pos[0] / len, v2 * (d - pos[1]) / len]
    return dpos_dt


t = np.linspace(0, 120, 1000)
pos0 = [0, 0]
pos = odeint(calc, pos0, t)
pos[:, 0] = np.maximum(pos[:, 0], 0.0)
pos[:, 1] = np.minimum(pos[:, 1], d)

index = None

for i in range(len(t)):
    if d - pos[i, 1] < 1e-6:
        index = i
        print(i)
        print(t[i])
        break

if index is not None:
    pos[index:1000, 1] = d

x = pos[:, 0]
y = pos[:, 1]
plt.subplot(1, 2, 1)
plt.plot(t, x, t, y)
plt.legend(["x", "y"])

analytic_x = (d - y) / 2.0 * (d ** k * (d - y) ** (-k) - d ** (-k) * (d - y) ** k)
plt.subplot(1, 2, 2)
plt.plot(x, y, analytic_x, y, "r--", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["numerical solution", "analytic solution"])
plt.show()
