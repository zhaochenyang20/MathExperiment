from scipy.integrate import odeint
from sympy import symbols, Function, dsolve, solve, log
import numpy as np

F = 253.86549552
c = 1.16674
m = 239.245
alpha = F / m
beta = c / m

xspan = np.linspace(0, 91.44, 100)
V0 = 0


def dV_dx(V, x):
    return alpha - beta * np.sqrt(2.0 * V)


V = odeint(dV_dx, V0, xspan)
Answer1 = np.sqrt(2.0 * V[-1])[0]
print(Answer1)

#! TODO

x = symbols("x")
v = Function("v")(x)
eqn = v.diff(x) * v - alpha + beta * v
cond = {v.subs(x, 0): 0}
vSol = dsolve(eqn, ics=cond)
Answer2 = float(vSol.rhs.subs(x, 91.44))
print(Answer2)

v = symbols("v")
eqn = (
    alpha * 91.44
    + alpha / beta * v
    + (alpha ** 2.0) / (beta ** 2.0) * log(1 - beta / alpha * v)
)
Answer3 = float(solve(eqn, v)[0])
print(Answer3)
