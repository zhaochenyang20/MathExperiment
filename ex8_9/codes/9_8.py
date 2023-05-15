import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import Bounds, minimize

price = np.array(
    [
        [1.300, 1.225, 1.149, 1.050],
        [1.103, 1.290, 1.260, 1.050],
        [1.216, 1.216, 1.419, 1.050],
        [0.954, 0.728, 0.922, 1.050],
        [0.929, 1.144, 1.169, 1.050],
        [1.056, 1.107, 0.965, 1.050],
        [1.038, 1.321, 1.133, 1.050],
        [1.089, 1.305, 1.732, 1.050],
        [1.090, 1.195, 1.021, 1.050],
        [1.083, 1.390, 1.131, 1.050],
        [1.035, 0.928, 1.006, 1.050],
        [1.176, 1.715, 1.908, 1.050],
    ]
)
conmission_rate = 0.01
x_0 = [0.5, 0.35, 0.15]
expectation = np.mean(price, axis=0)
convariance = np.cov(price, rowvar=False)
lower_earning_rate = 0.10
upper_earning_rate = np.max(expectation) - 1  # possible to be broken
step_earning_rate = 0.01
earning_rates = np.arange(lower_earning_rate, upper_earning_rate, step_earning_rate)


def x2var(x, n):
    return x.dot(convariance[:n, :n]).dot(x)


def func_con_bond_earning(x, nbonds, earning_rate):
    return np.sum(x * expectation[:nbonds]) - (1.0 + earning_rate)


def exchange(nbonds, earning_rate):
    cons_switch = [
        {
            "type": "ineq",
            "fun": lambda bs: func_con_switch_earning(bs, nbonds, earning_rate),
        },
        {"type": "ineq", "fun": func_con_switch},
        {"type": "ineq", "fun": convert_bs_to_x},
    ]

    bs_0 = np.zeros(6)
    bounds = Bounds(np.zeros(6), np.ones(6))
    result = minimize(
        bs2var,
        bs_0,
        args=(nbonds,),
        method="SLSQP",
        constraints=cons_switch,
        bounds=bounds,
    )
    return result.x, result.fun


def without_switch(nbonds, earning_rate):
    x_0 = np.random.rand(nbonds)
    con_normalize = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    con_earning_rate = {
        "type": "ineq",
        "fun": func_con_bond_earning,
        "args": [nbonds, earning_rate],
    }
    contraints = [con_normalize, con_earning_rate]
    bnds = Bounds(np.zeros(nbonds), np.ones(nbonds))
    res = minimize(
        x2var, x_0, args=(nbonds,), method="SLSQP", constraints=contraints, bounds=bnds
    )
    return res.x, res.fun


def divide_bs(bs):
    bs_reshaped = bs.reshape((2, 3))
    b = bs_reshaped[0]
    s = bs_reshaped[1]
    return b, s


def convert_bs_to_x(bs):
    b, s = divide_bs(bs)
    x = x_0 + (1 - conmission_rate) * b - s
    return x


def bs2var(bs, nbonds):
    x = convert_bs_to_x(bs)
    return x2var(x, nbonds)


def func_con_switch(bs):
    b, s = divide_bs(bs)
    return (1 - conmission_rate) * np.sum(s) - np.sum(b)


def func_con_switch_earning(bs, nbonds, earning_rate):
    x = convert_bs_to_x(bs)
    return func_con_bond_earning(x, nbonds, earning_rate) + func_con_switch(bs)


def optimization(nbonds=3, portfolio_func=without_switch):
    bond_decisions = []
    variances = []

    for earning_rate in earning_rates:
        bond_decision, variance = portfolio_func(
            nbonds=nbonds, earning_rate=earning_rate
        )
        bond_decisions.append(bond_decision)
        variances.append(variance)

    bond_decisions = np.array(bond_decisions)
    variances = np.array(variances)

    return bond_decisions, variances


def optimaize(nbonds, earning_rate, portfolio_func):
    x_opt, var_opt = portfolio_func(nbonds, earning_rate)
    return x_opt, var_opt


def plot_xs(xs, labels):
    fig, ax = plt.subplots()
    for i in range(xs.shape[1]):
        ax.plot(earning_rates, xs[:, i], label=labels[i])
    ax.legend()
    fig.show()


def plot_optimal_earnings(earning_rates, xs_list, vars_list, labels):
    _, ax = plt.subplots()
    for _, var, label in zip(xs_list, vars_list, labels):
        ax.plot(earning_rates, var, label=label)
    ax.legend()
    plt.show()


def optimaize(nbonds, earning_rate, portfolio_func):
    bond_decision, var = portfolio_func(nbonds, earning_rate)
    return bond_decision, var


def run_experiment():
    """Run an experiment and print the results.

    This code defines a function run_experiment that calls several other functions
    to get optimal solutions for different scenarios, divide them into two variables,
    and normalize them. It then calls plot_xs and plot_optimal_earnings to visualize
    the results. The code is likely part of a larger project that involves optimization
    and visualization of financial data.
    """
    x_A_B_C, var_A_B_C = optimaize(3, 0.15, without_switch)
    x_All, var_All = optimaize(4, 0.15, without_switch)
    bs_switch, var_switch = optimaize(3, 0.15, exchange)
    b_switch, s_switch = divide_bs(bs_switch)
    x_switch = convert_bs_to_x(bs_switch)
    xs_A_B_C, vars_A_B_C = optimization(3)
    xs_A_B_C_D, vars_A_B_C_D = optimization(4)
    bss_switch, vars_switch = optimization(3, exchange)
    xs_switch = np.array([convert_bs_to_x(bs) for bs in bss_switch])
    cost = 1 - np.sum(x_switch)

    print("================================================")
    print("x = ", x_A_B_C)
    print("v = ", var_A_B_C)
    plot_xs(xs_A_B_C, ["A", "B", "C"])

    print("================================================")
    print("x = ", x_All)
    print("v = ", var_All)
    plot_xs(xs_A_B_C_D, ["A", "B", "C", "D"])

    print("================================================")
    print("b = ", b_switch)
    print("s = ", s_switch)
    print("x = ", x_switch)
    print("v = ", var_switch)
    print("cost = ", cost)
    plot_xs(xs_switch, ["A", "B", "C"])

    plot_optimal_earnings(
        earning_rates,
        [xs_A_B_C],
        [vars_A_B_C],
        ["A+B+C"],
    )
    plot_optimal_earnings(
        earning_rates,
        [xs_A_B_C_D],
        [vars_A_B_C_D],
        ["A+B+C+D"],
    )
    plot_optimal_earnings(
        earning_rates,
        [xs_switch],
        [vars_switch],
        ["A+B+C (exchange)"],
    )


if __name__ == "__main__":
    run_experiment()
