from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

r1 = 1.0
r2 = 1.0
n1 = 100.0
n2 = 100.0
s1 = 1.5
s2 = 1.9
x0 = 10.0
y0 = 17.0


def diff(xy, t):
    dxy_dt = [
        r1 * xy[0] * (1 - xy[0] / n1 - s1 * xy[1] / n2),
        r2 * xy[1] * (1 - s2 * xy[0] / n1 - xy[1] / n2),
    ]
    return dxy_dt


t = np.linspace(0, 20, 1000)
xy = odeint(diff, [x0, y0], t)
x = xy[:, 0]
y = xy[:, 1]

_, axs = plt.subplots(1, 2)
axs[0].plot(t, x, t, y)
axs[0].set_xlabel("time")
axs[0].set_ylabel("population")
axs[0].legend(["x", "y"])
axs[1].plot(x, x, x, y)
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].legend(["x", "y"])
plt.show()
