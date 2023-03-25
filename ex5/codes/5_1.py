import numpy as np

np.set_printoptions(precision=15)


def main(n):

    A_Vandermonde = np.zeros((n, n))
    A_Vandermonde[:, 0] = 1
    b_Vandermonde = np.zeros((n, 1))

    for i in range(n):
        for j in range(1, n):
            A_Vandermonde[i, j] = (1 + 0.1 * i) ** j

    for i in range(n):
        for j in range(n):
            b_Vandermonde[i, 0] += A_Vandermonde[i, j]

    A_Hilbert = np.array(
        [[1 / (i + j - 1) for j in range(1, n + 1)] for i in range(1, n + 1)]
    )
    b_Hilbert = np.zeros((n, 1))
    for i in range(n):
        for j in range(n):
            b_Hilbert[i, 0] += A_Hilbert[i, j]

    x_Vandermonde = np.linalg.solve(A_Vandermonde, b_Vandermonde)
    x_Hilbert = np.linalg.solve(A_Hilbert, b_Hilbert)
    x = np.ones((n, 1))
    print("x_Vandermonde:", x_Vandermonde)
    print("x_Hilbert:", x_Hilbert)

    cond_Vandermonde = np.linalg.cond(A_Vandermonde)
    cond_Hilbert = np.linalg.cond(A_Hilbert)
    print("cond_Vandermonde:", cond_Vandermonde)
    print("cond_Hilbert:", cond_Hilbert)

    print("pertub A")
    for epsilon in [10 ** (-10.0), 10 ** (-8.0), 10 ** (-6.0)]:
        print("epsilon:", epsilon)
        A1 = A_Vandermonde.copy()
        A1[n - 1, n - 1] += epsilon
        x1 = np.linalg.solve(A1, b_Vandermonde)
        true_error1 = np.linalg.norm(x1 - x) / np.linalg.norm(x)
        frac = epsilon / np.linalg.norm(A_Vandermonde)
        if cond_Vandermonde * frac < 1:
            upper_error1 = cond_Vandermonde * frac / (1 - cond_Vandermonde * frac)
        else:
            upper_error1 = cond_Vandermonde * frac
        A2 = A_Hilbert.copy()
        A2[n - 1, n - 1] += epsilon
        x2 = np.linalg.solve(A2, b_Hilbert)
        true_error2 = np.linalg.norm(x2 - x) / np.linalg.norm(x)
        frac = epsilon / np.linalg.norm(A_Hilbert)
        if cond_Hilbert * frac < 1:
            upper_error2 = cond_Hilbert * frac / (1 - cond_Hilbert * frac)
        else:
            upper_error2 = cond_Hilbert * frac / np.finfo(float).eps
        print("True error in x_Vandermonde:", true_error1)
        print("Upper bound on error in x_Vandermonde:", upper_error1)
        print("True error in x_Hilbert:", true_error2)
        print("Upper bound on error in x_Hilbert:", upper_error2)

    print("pertub b")
    for epsilon in [10 ** (-10.0), 10 ** (-8.0), 10 ** (-6.0)]:
        print(epsilon)
        b1 = b_Vandermonde.copy()
        b1[-1, 0] += epsilon
        x1 = np.linalg.solve(A_Vandermonde, b1)
        true_error1 = np.linalg.norm(x1 - x) / np.linalg.norm(x)
        upper_error1 = cond_Vandermonde * epsilon / np.linalg.norm(b_Vandermonde)
        b2 = b_Hilbert.copy()
        b2[-1, 0] += epsilon
        x2 = np.linalg.solve(A_Hilbert, b2)
        true_error2 = np.linalg.norm(x2 - x) / np.linalg.norm(x)
        upper_error2 = cond_Hilbert * epsilon / np.linalg.norm(b_Hilbert)
        print("True error in x_Vandermonde:", true_error1)
        print("Upper bound on error in x_Vandermonde:", upper_error1)
        print("True error in x_Hilbert:", true_error2)
        print("Upper bound on error in x_Hilbert:", upper_error2)


if __name__ == "__main__":
    for i in [5, 7, 9, 11]:
        print("n =", i)
        main(i)
