import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import norm


def objective_function1(m, l, sigma):
    """
    Return the expected waste for one steel bar.

    Parameters
    ----------
    m : float
        The mean length of the steel bars.
    l : float
        The target length of the steel bars.
    sigma : float
        The standard deviation of the steel bar lengths.

    Returns
    -------
    float
        The expected waste for one steel bar.
    """
    return m - l * (1 - norm.cdf((l - m) / sigma))


def objective_function2(m, l, sigma):
    """
    Return the expected waste for each obtained steel bar.

    Parameters
    ----------
    m : float
        The mean length of the steel bars.
    l : float
        The target length of the steel bars.
    sigma : float
        The standard deviation of the steel bar lengths.

    Returns
    -------
    float
        The expected waste for each obtained steel bar.
    """
    return m / (1 - norm.cdf((l - m) / sigma)) - l


def main():
    """
    Optimize the objective functions and plot the results.
    """
    l = 2.0
    sigma = 0.2

    # Optimize the function objective_function1 in the range [0, 4]
    res1 = minimize_scalar(objective_function1, args=(l, sigma), bounds=(0, 4))
    x1 = res1.x
    fval1 = res1.fun

    # Optimize the function objective_function2 in the range [0, 4]
    res2 = minimize_scalar(objective_function2, args=(l, sigma), bounds=(0, 4))
    x2 = res2.x
    fval2 = res2.fun

    print("First Optimization 1:")
    print("x1:", x1)
    print("fval1:", fval1)
    print()
    print("Second Optimization 2:")
    print("x2:", x2)
    print("fval2:", fval2)

    xrange = np.arange(2, 4, 0.01)
    y1 = objective_function1(xrange, l, sigma)
    y2 = objective_function2(xrange, l, sigma)

    plt.plot(xrange, y1, label="Expected waste for one steel bar")
    plt.plot(xrange, y2, label="Expected waste for each obtained steel bar")
    plt.legend()
    plt.xlabel("m")
    plt.ylabel("Objective function")
    plt.show()


if __name__ == "__main__":
    main()
