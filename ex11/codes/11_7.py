import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm


def calculate_n0(mu, sigma, A, K, b, c):
    """
    Calculate the number of customers that need to be served in order to maximize the profit.

    Parameters
    ----------
    mu : float
        The mean number of customers.
    sigma : float
        The standard deviation of the number of customers.
    A : float
        The cost of serving a customer.
    K : float
        The fixed cost.
    b : float
        The minimum profit.
    c : float
        The maximum profit.

    Returns
    -------
    float
        The number of customers that need to be served in order to maximize the profit.
    """

    equation = lambda n0: norm.cdf(n0, mu, sigma) - (A * (2 * n0 / K - 1) + b) / (b - c)
    n0_solution = fsolve(equation, mu)
    return n0_solution[0]


def calculate_second_derivative(mu, sigma, A, K, b, c, n0):
    """
    Calculate the second derivative of the profit function.

    Parameters
    ----------
    mu : float
        The mean number of customers.
    sigma : float
        The standard deviation of the number of customers.
    A : float
        The cost of serving a customer.
    K : float
        The fixed cost.
    b : float
        The minimum profit.
    c : float
        The maximum profit.

    Returns
    -------
    float
        The second derivative of the profit function.
    """

    second_derivative = 2 * A / K + (c - b) * norm.pdf(n0, mu, sigma)
    return second_derivative


# Set parameters
mu = 2000
sigma = 50
A = 0.5
K = 50000
b = 0.5
c = 0.35

# Calculate n0 and second derivative
n0 = calculate_n0(mu, sigma, A, K, b, c)
second_derivative = calculate_second_derivative(mu, sigma, A, K, b, c, n0)

# Print results
print(f"n0: {n0:.4f}")
print(f"g''(n): {second_derivative:.4f}")
