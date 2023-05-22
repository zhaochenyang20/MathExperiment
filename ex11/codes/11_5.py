import numpy as np
from scipy import integrate


def calculate_p(x, y, sigma_x, sigma_y, r):
    coefficient = 1.0 / (2.0 * np.pi * sigma_x * sigma_y * np.sqrt(1 - r**2))
    return coefficient * np.exp(
        -(
            x**2 / sigma_x**2
            - 2 * r * x * y / (sigma_x * sigma_y)
            + y**2 / sigma_y**2
        )
        / (2 * (1 - r**2))
    )


def integrate_exact(a, sigma_x, sigma_y, r):
    def integrand(y, x):
        return calculate_p(x, y, sigma_x, sigma_y, r)

    return (
        2.0
        * integrate.dblquad(
            integrand, -a, a, lambda x: 0.0, lambda x: np.sqrt(a**2 - x**2)
        )[0]
    )


def monte_carlo_simulation(a, sigma_x, sigma_y, r, n):
    p_max = calculate_p(
        0, 0, sigma_x, sigma_y, r
    )  # Maximum value of the probability density function

    def in_target_region(x, y):
        return x**2 + y**2 <= a**2

    hits = 0
    for _ in range(n):
        x, y = np.random.uniform(-a, a, size=2)
        if in_target_region(x, y):
            hits += calculate_p(x, y, sigma_x, sigma_y, r)

    return hits / n * (2 * a) ** 2


def run_experiment(a, sigma_x, sigma_y, r, sample_sizes):
    exact_result = integrate_exact(a, sigma_x, sigma_y, r)
    print("[integrate]:", exact_result)

    for n in sample_sizes:
        monte_carlo_result = monte_carlo_simulation(a, sigma_x, sigma_y, r, n)
        print(
            f"[MC {n}]",
            monte_carlo_result,
            " error:",
            abs(exact_result - monte_carlo_result),
        )


# Set parameters
a = 100
sigma_x = 80
sigma_y = 50
r = 0.4
sample_sizes = [100, 1000, 10000, 100000, 1000000, 10000000]

# Run the experiment
run_experiment(a, sigma_x, sigma_y, r, sample_sizes)
