import numpy as np
from scipy.optimize import minimize

# 定义规划
def objective(x):
    return -(9 * x[0] - x[1] - 6 * x[2] - x[3] + 5 * x[4])


def constraint1(x):
    return (x[0] + x[1]) * (5 * x[2] + x[3]) - 2 * x[2] * (3 * x[0] + x[1])


def constraint2(x):
    return (x[0] + x[1]) * (-3 * x[0] + x[1] - 3 * x[2] - x[4]) + 2 * x[2] * (
        3 * x[0] + x[1]
    )


def constraint3(x):
    return -x[0] - x[1] + x[2] - x[4] + 200


def constraint4(x):
    return -x[2] - x[3] + 100


def constraint5(x):
    return x[0] + x[1] - x[2]


def constraint6(x):
    return x[3] + x[4]


def constraint7(x):
    return -x[3] - x[4] + 500


bnds = ((0, 500), (0, 500), (0, None), (0, None), (0, None))

con1 = {"type": "ineq", "fun": constraint1}
con2 = {"type": "ineq", "fun": constraint2}
con3 = {"type": "ineq", "fun": constraint3}
con4 = {"type": "ineq", "fun": constraint4}
con5 = {"type": "ineq", "fun": constraint5}
con6 = {"type": "ineq", "fun": constraint6}
con7 = {"type": "ineq", "fun": constraint7}

cons = [con1, con2, con3, con4, con5, con6, con7]

# 求解并报告
def search(ntrials=1000):
    best_objective = 0
    best_sol = None
    for trial in range(ntrials):
        x0 = np.random.rand(5) * 500
        sol = minimize(objective, x0, method="SLSQP", bounds=bnds, constraints=cons)
        if -sol.fun > best_objective:
            best_sol = sol

    print("[Optimal solution] x =", best_sol.x.round(0))
    print("[Objective value ] f =", -best_sol.fun.round(2))


def random_search(ntrials=500):
    """
    随机搜索
    """
    x0s = np.random.rand(ntrials, 5) * 500
    search(x0s)


# 求解 (1)
print("===== (1) =====")
random_search()
