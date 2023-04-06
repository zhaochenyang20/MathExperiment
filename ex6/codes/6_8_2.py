import numpy as np
import math
import matplotlib.pyplot as plt

def equ(q, c, mu, r, d):
    return (1 - r) * q + r / d * (c - math.atan(mu * q))

def simulate(q0, c, mu, r, d, n=100):
    q = np.zeros(n)
    q[0] = q0
    for i in range(1, n):
        q[i] = equ(q[i - 1], c, mu, r, d)
    return q

def plot_q(q):
    x = np.arange(len(q))
    plt.plot(x, q)
    plt.xlabel('Iter')
    plt.ylabel('q')
    plt.show()

def main():
    mu = 4.8
    d = 0.25
    r = 0.3
    q0 = 1.0
    cs = [2, 1, 0.92, 0.9, 0.895]
    for c in cs:
        q = simulate(q0, c, mu, r, d)
        plot_q(q)

if __name__ == '__main__':
    main()