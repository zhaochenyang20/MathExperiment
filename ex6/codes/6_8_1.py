import numpy as np
import math
import matplotlib.pyplot as plt

THRESHOLD = 1e-5
MU = 4.8
D = 0.25
R = 0.3
SAVE_START = 1000
SAVE_DURATION = 256
TOTAL_ITER = SAVE_START + SAVE_DURATION


def equ(q, c):
    return (1 - R) * q + R / D * (c - math.atan(MU * q))


def compute_q_values():
    q = np.zeros((11000, TOTAL_ITER))
    index = 0
    for c in np.arange(0.0, 1.1, 0.0001):
        q[index, 0] = 1.0
        for n in range(1, TOTAL_ITER):
            q[index, n] = equ(q[index, n - 1], c)
        index += 1
    return q


def find_forks(q):
    cur_n = 1
    for index in range(q.shape[0]):
        n = 1
        conv_fail = False
        while n <= 128:
            conv_fail = True
            for i in range(n):
                if (
                    abs(q[index, SAVE_START + i] - q[index, SAVE_START + n + i])
                    >= THRESHOLD
                ):
                    conv_fail = False
                    break
            if conv_fail:
                if cur_n != n:
                    cur_n = n
                    print("c=", np.arange(0.0, 1.1, 0.0001)[index], "f=", n)
                break
            n = n * 2


def plot_q_values(q):
    x = np.arange(0.0, 1.1, 0.0001)
    plt.plot(x, q[:, SAVE_START:TOTAL_ITER])
    plt.xlabel("Parameter: c")
    plt.ylabel("Value: q")
    plt.show()


if __name__ == "__main__":
    q = compute_q_values()
    find_forks(q)
    plot_q_values(q)
