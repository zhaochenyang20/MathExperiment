import numpy as np
from scipy.integrate import quad, simps, quadrature


def calculate_pi(method):
    # Define a function for calculating pi using different methods
    x = np.linspace(0.0, 1.0, 10**5)
    y = np.sqrt(1.0 - x**2)
    fun1 = lambda x: np.sqrt(1.0 - x**2)

    if method == "quad":
        return quad(fun1, 0.0, 1.0, epsabs=1e-15)[0] * 4.0
    elif method == "simps":
        return simps(y, x) * 4.0
    elif method == "quadrature":
        return quadrature(fun1, 0.0, 1.0, tol=1e-30, maxiter=1000)[0] * 4.0
    else:
        raise ValueError("Invalid method")


# Call the function with different methods and print the results
methods = ["quad", "simps", "quadrature"]
for method in methods:
    pi = calculate_pi(method)
    print(f"pi_{method} = {pi:.15f}")
