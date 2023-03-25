import numpy as np

np.set_printoptions(precision=15)

n = 5
b = np.array([0, 0, 5, 3, 0])
s = np.array([0.4, 0.6, 0.6, 0.4])
h = np.array([0, 500, 400, 200, 100]).reshape((n, 1))

A = np.zeros((n, n))

for i in range(n):
    A[0, i] = b[i]

for i in range(1, n):
    A[i, i - 1] = s[i - 1]

print(np.linalg.cond(A - np.eye(n)))
x = np.linalg.solve(A - np.eye(n), h)
print(np.linalg.norm(x - (A.dot(x) - h)))
