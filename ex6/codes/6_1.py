from scipy.optimize import fsolve
import numpy as np

np.set_printoptions(precision=15)

S = 150000
m = 1000
n = 180
q = fsolve(lambda q: (S - m / q) * (1 + q) ** n + m / q, 0.1)

S = 500000
m = 4500
n = 180
q1 = fsolve(lambda q: (S - m / q) * (1 + q) ** n + m / q, 0.1)

m = 45000
n = 20
q2 = fsolve(lambda q: (S - m / q) * (1 + q) ** n + m / q, 0.1) / 12

print(q)
print(q1)
print(q2)
