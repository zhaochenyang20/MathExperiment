import numpy as np
import matplotlib.pyplot as plt


def jacobi(D, L, U, x_0, b, error):
    inv_D = np.linalg.inv(D)
    k = 0
    xk = x_0
    while True:
        xk_1 = inv_D @ ((L + U) @ xk + b)
        if np.linalg.norm(xk - xk_1, np.inf) < error:
            return xk_1, k + 1
        k += 1
        xk = xk_1


def gauss_seidel(D, L, U, x_0, b, error):
    inv_D_minus_L = np.linalg.inv(D - L)
    k = 0
    xk = x_0
    while True:
        xk_1 = inv_D_minus_L @ (U @ xk + b)
        if np.linalg.norm(xk - xk_1, np.inf) < error:
            return xk_1, k + 1
        k += 1
        xk = xk_1


n = 20
A = (
    3 * np.eye(n)
    + np.diag(np.ones(n - 1) * -0.5, 1)
    + np.diag(np.ones(n - 1) * -0.5, -1)
    + np.diag(np.ones(n - 2) * -0.25, 2)
    + np.diag(np.ones(n - 2) * -0.25, -2)
)

D = np.diag(np.diag(A))
L = -np.tril(A, -1)
U = -np.triu(A, 1)
error = 1e-8

b = np.zeros((20, 1))
x_0 = np.zeros((20, 1))
result, iteration = jacobi(D, L, U, x_0, b, error)
print("jacobi      ", iteration)
result, iteration = gauss_seidel(D, L, U, x_0, b, error)
print("gauss_seidel", iteration)

x_0 = np.ones((20, 1))
result, iteration = jacobi(D, L, U, x_0, b, error)
print("jacobi      ", iteration)
result, iteration = gauss_seidel(D, L, U, x_0, b, error)
print("gauss_seidel", iteration)

b = np.arange(1, 21).reshape(20, 1)
x_0 = np.linalg.solve(A, b)
result, iteration = jacobi(D, L, U, x_0, b, error)
print("jacobi      ", iteration)
result, iteration = gauss_seidel(D, L, U, x_0, b, error)
print("gauss_seidel", iteration)

for perturb in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
    x_0 += perturb * np.ones((20, 1))
    result, iteration = jacobi(D, L, U, x_0, b, error)
    print("jacobi      ", perturb, iteration)
    result, iteration = gauss_seidel(D, L, U, x_0, b, error)
    print("gauss_seidel", perturb, iteration)

x_0 = np.zeros((20, 1))
result, iteration = jacobi(D, L, U, x_0, b, error)
print("jacobi      ", iteration)
result, iteration = gauss_seidel(D, L, U, x_0, b, error)
print("gauss_seidel", iteration)

jacobi_iterations = np.zeros(100)
gauss_seidel_iterations = np.zeros(100)
error = 1e-5
x_0 = np.zeros((20, 1))

for d in range(1, 101):
    A = (
        (3 * d) * np.eye(n)
        + np.diag(np.ones(n - 1) * -0.5, 1)
        + np.diag(np.ones(n - 1) * -0.5, -1)
        + np.diag(np.ones(n - 2) * -0.25, 2)
        + np.diag(np.ones(n - 2) * -0.25, -2)
    )
    D = np.diag(np.diag(A))
    result, iteration = jacobi(D, L, U, x_0, b, error)
    jacobi_iterations[d - 1] = iteration
    result, iteration = gauss_seidel(D, L, U, x_0, b, error)
    gauss_seidel_iterations[d - 1] = iteration

plt.plot(range(1, 101), jacobi_iterations, label="Jacobi")
plt.plot(range(1, 101), gauss_seidel_iterations, label="Gauss Seidel")
plt.xlabel("Multiplier")
plt.ylabel("Iterations")
plt.legend()
plt.show()
