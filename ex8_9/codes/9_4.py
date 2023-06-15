from scipy.optimize import minimize


# Define the function to minimize
def get_varible(t):
    x, y, w, z1, z2 = t
    return (x, y, w, z1, z2)


def f(t):
    x, y, w, z1, z2 = get_varible(t)
    # F=9(w+z_1)+15((x+y-w)+z_2)-6x-16y-10(z_1+z_2)
    F = 9 * (w + z1) + 15 * ((x + y - w) + z2) - 6 * x - 16 * y - 10 * (z1+ z2)
    return -F

def f2(t):
    x, y, w, z1, z2 = get_varible(t)
    # F=9(w+z_1)+15((x+y-w)+z_2)-6x-16y-10(z_1+z_2)
    F = 9 * (w + z1) + 15 * ((x + y - w) + z2) - 6 * x - 13 * y - 10 * (z1+ z2)
    return -F

def constraint1(t):
    # 0 \leq z1 + z2 \leq 500
    x, y, w, z1, z2 = get_varible(t)
    #! z1 + z2 - 500 <= 0
    return 500 - z1 + z2


def constraint2(t):
    x, y, w, z1, z2 = get_varible(t)
    #! w - x - y <= 0
    return x + y - w


def constraint3(t):
    x, y, w, z1, z2 = get_varible(t)
    return -(0.03 * x + 0.01 * y) * w / (x + y) - 0.02 * z1 + 2.5 * 10 ** (-2) * (w + z1)


def constraint4(t):
    x, y, w, z1, z2 = get_varible(t)
    return -(0.03 * x + 0.01 * y) * (x + y - w) / (x + y) - 0.02 * z2 + 1.5 * 10 ** (-2) * (x + y + z2 - w)



def constraint5(t):
    x, y, w, z1, z2 = get_varible(t)
    return -w  - z1 + 100

def constraint52(t):
    x, y, w, z1, z2 = get_varible(t)
    return -w  - z1 + 600

def constraint6(t):
    x, y, w, z1, z2 = get_varible(t)
    return 200 - (x + y - w + z2)


# Initial guess
t0 = [
    100,
    100,
    100,
    50,
    50,
]

# Define the bounds for the variables
bounds = [(0, 300), (0, 300), (0, 300),  (0, 300),  (0, 300)]

# Define the constraints
cons = [
    {"type": "ineq", "fun": constraint1},
    {"type": "ineq", "fun": constraint2},
    {"type": "ineq", "fun": constraint3},
    {"type": "ineq", "fun": constraint4},
    {"type": "ineq", "fun": constraint5},
    {"type": "ineq", "fun": constraint6},
]

# Call the minimize function
res = minimize(f, t0, method="SLSQP", bounds=bounds, constraints=cons)

print(res)

# Define the constraints
cons = [
    {"type": "ineq", "fun": constraint1},
    {"type": "ineq", "fun": constraint2},
    {"type": "ineq", "fun": constraint3},
    {"type": "ineq", "fun": constraint4},
    {"type": "ineq", "fun": constraint52},
    {"type": "ineq", "fun": constraint6},
]

# Call the minimize function
res = minimize(f2, t0, method="SLSQP", bounds=bounds, constraints=cons)

print(res)