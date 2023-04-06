import numpy as np
from scipy.integrate import quad

fun1 = lambda x: np.sqrt(1.0 - x ** 2)
fun2 = lambda x: np.exp(-(x ** 2))
fun3 = lambda x: 1.0 / (1.0 + x ** 2)

pi_1 = quad(fun1, 0.0, 1.0, epsabs=1e-15)[0] * 4.0
pi_2 = (2.0 * quad(fun2, 0.0, np.inf, epsabs=1e-15)[0]) ** 2
pi_3 = quad(fun3, 0.0, 1.0, epsabs=1e-15)[0] * 4.0

print(f"pi_1 = {pi_1:.15f}")
print(f"pi_2 = {pi_2:.15f}")
print(f"pi_3 = {pi_3:.15f}")
