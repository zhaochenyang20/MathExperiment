import numpy as np

def split_matrix(A):
    n = A.shape[0]  # 方阵的维度
    D = np.diag(np.diag(A))  # 对角矩阵
    L = -np.tril(A, k=-1)  # 严格下三角矩阵
    U = -np.triu(A, k=1)  # 严格上三角矩阵
    return D, L, U

def spectral_radius(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    radius = np.max(np.abs(eigenvalues))
    return radius

A = np.array([[5, -7, 0, 1],
             [-3, 22, 6, 2],
             [5, -1, 31, -1],
             [2, 1, 0, 23]])
D, L, U = split_matrix(A)
inv_D_minus_L = np.linalg.inv(D - L)
step_matrix = inv_D_minus_L @ U
print(spectral_radius(step_matrix))