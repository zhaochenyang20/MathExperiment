import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import Bounds, minimize

method = "SLSQP"


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

expectation = np.mean(price, axis=0)


convariance = np.cov(price, rowvar=False)


lower_earning_rate = 0.10
upper_earning_rate = np.max(expectation) - 1  # possible to be broken
step_earning_rate = 0.01
earning_rates = np.arange(lower_earning_rate, upper_earning_rate, step_earning_rate)



def x2var(x, n):
    # pdprint(type(n))
    return x.dot(convariance[:n, :n]).dot(x)


def func_con_bond_earning(x, nbonds, earning_rate):
    return np.sum(x * expectation[:nbonds]) - (1.0 + earning_rate)


def without_exchange(nbonds, earning_rate):
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
        x2var, x_0, args=(nbonds,), method=method, constraints=contraints, bounds=bnds
    )

    x = res.x
    var = res.fun
    return x, var


def devide_bs(bs):
    bs = bs.reshape((2, 3))
    b = bs[0]
    s = bs[1]
    return b, s


conmission_rate = 0.01
x_0 = [0.5, 0.35, 0.15]


def bs2x(bs):
    b, s = devide_bs(bs)
    x = x_0 + (1 - conmission_rate) * b - s
    return x


def bs2var(bs, nbonds):
    x = bs2x(bs)
    return x2var(x, nbonds)  # no need to normalize


def func_con_exchange(bs):
    b, s = devide_bs(bs)
    return (1 - conmission_rate) * np.sum(s) - np.sum(b)


def func_con_exchange_earning(bs, nbonds, earning_rate):
    x = bs2x(bs)
    return func_con_bond_earning(x, nbonds, earning_rate) + func_con_exchange(bs)


def exchange(nbonds, earning_rate):
    con_earning_rate = {
        "type": "ineq",
        "fun": func_con_exchange_earning,
        "args": [nbonds, earning_rate],
    }
    con_exchange = {"type": "ineq", "fun": func_con_exchange}
    con_ratio_not_negative = {"type": "ineq", "fun": bs2x}
    cons_exchange = [con_earning_rate, con_exchange, con_ratio_not_negative]

    bs_0 = np.zeros(6)
    bnds = Bounds(np.zeros(6), np.ones(6))
    res_exchange = minimize(
        bs2var,
        bs_0,
        args=(nbonds),
        method=method,
        constraints=cons_exchange,
        bounds=bnds,
    )
    bs = res_exchange.x
    var = res_exchange.fun
    return bs, var


def iterate_earning_rates(nbonds=3, portfolio_func=without_exchange):
    decisions = []
    vars = []

    for earning_rate in earning_rates:
        decision, var = portfolio_func(nbonds, earning_rate)
        decisions.append(decision)
        vars.append(var)

    decisions = np.array(decisions)
    vars = np.array(vars)
    return decisions, vars



def print_devider(idx):
    print(f"===== ({idx}) =====")


def plot_xs(xs, labels):
    fig, ax = plt.subplots()
    for i in range(xs.shape[1]):
        ax.plot(earning_rates, xs[:, i], label=labels[i])
    ax.legend()
    fig.show()

x_opt_3, var_opt_3 = without_exchange(3, 0.15)

x_opt_4, var_opt_4 = without_exchange(4, 0.15)

bs_opt_exchange, var_opt_exchange = exchange(3, 0.15)
bs_opt_exchange = bs_opt_exchange.round(2)
b_opt_exchange, s_opt_exchange = devide_bs(bs_opt_exchange)
x_opt_exchange = bs2x(bs_opt_exchange)
x_opt_exchange_norm = x_opt_exchange / np.sum(x_opt_exchange)

xs_opt_3, vars_opt_3 = iterate_earning_rates(3)
xs_opt_4, vars_opt_4 = iterate_earning_rates(4)
bss_opt_exchange, vars_opt_exchange = iterate_earning_rates(3, exchange)
xs_opt_exchange = np.array([bs2x(bs) for bs in bss_opt_exchange])


print_devider(1)

print("[Optimal Solution] x = ", x_opt_3)
print("[Optimal Objective] v = ", var_opt_3)
plot_xs(xs_opt_3, ["A", "B", "C"])


print_devider(2)

print("[Optimal Solution] x = ", x_opt_4)
print("[Optimal Objective] v = ", var_opt_4)
plot_xs(xs_opt_4, ["A", "B", "C", "D"])

print_devider(3)

print("[Optimal Solution] b = ", b_opt_exchange, ", s =", s_opt_exchange)
print(
    "[Optimal Solution] x_org = ",
    x_opt_exchange,
    "=> x_norm =",
    x_opt_exchange_norm,
)
print("[Optimal Objective] v = ", var_opt_exchange)
plot_xs(xs_opt_exchange, ["A", "B", "C"])

plt.rcParams.update({"lines.linestyle": "--"})
fig, ax = plt.subplots()
ax.plot(earning_rates, vars_opt_3, label="A+B+C")
ax.plot(earning_rates, vars_opt_4, label="A+B+C+D")
ax.plot(earning_rates, vars_opt_exchange, label="A+B+C (exchange)")
ax.legend()
fig.show()
