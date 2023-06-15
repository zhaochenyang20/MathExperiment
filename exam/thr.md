# 三

1) 0.07434
2) x^(5) = [ 1.71597235  0.39264682 -0.1305711   0.13806124]
3) 收敛，因为谱半径约等于 0.4 < 1

## 3.1

### 分析

条件数估计解相对误差上界

注意这里是1-范数？

```python
import numpy as np

order = 1

A=np.array([
    [5, -7, 0, 1],
    [-3,22,6,2],
    [5,-1,31,-1],
    [2,1,0,23]
])
b = np.array([6,3,4,7])
pert_b = np.array([0,0,0,0.1])

cond_A = np.linalg.cond(A, p=order)
print(cond_A)

rel_pert_b = np.linalg.norm(pert_b, ord=order) / np.linalg.norm(b, ord=order)
print(rel_pert_b)

sup_rel_err_x = cond_A * rel_pert_b
print(sup_rel_err_x)
```

### 答案

0.07433906520893041

## 3.2

### 分析

G-S 迭代法

```python


def gauss_sedeil(A, b, x0, tol, max_iter):
    x = x0.copy()

    # 求 D_sub_L_inv
    D = np.diag(np.diag(A))
    U = -(np.triu(A) - D)
    L = -(np.tril(A) - D)
    D_sub_L_inv = np.linalg.inv(D - L)

    for i in range(max_iter):
        # print(f"x^({i}) = {x}")

        x_new = D_sub_L_inv @ (U @ x + b)
        err = np.linalg.norm(x_new - x)

        x = x_new
        if  err < tol:
            print(f"Converge after {i+1} iterations")
            break

    print(f"x^({i+1}) = {x}")
    print(f"relative error = {err}")
    return x, err

gauss_sedeil(A=A,b=b,x0=np.zeros(4),tol=1e-6,max_iter=5)
```

### 答案

x^(5) = [ 1.71597235  0.39264682 -0.1305711   0.13806124]

## 3.3

### 分析

收敛性

1. 理论说明：迭代系数矩阵 B 谱半径不大于 1

```python
def cal_B_G_S(A):
    """求 G-S 迭代的迭代矩阵"""
    D = np.diag(np.diag(A))
    U = -(np.triu(A) - D)
    L = -(np.tril(A) - D)
    D_sub_L_inv = np.linalg.inv(D - L)
    B_G_S = D_sub_L_inv @ U

    return B_G_S

B_G_S = cal_B_G_S(A)
# print(B_G_S)
# 求 B_G_S 的谱半径
rho_B_G_S = np.max(np.abs(np.linalg.eigvals(B_G_S)))
print(rho_B_G_S)
```

2. 实验验证：16 次迭代后收敛

```python
gauss_sedeil(A=A,b=b,x0=np.zeros(4),tol=1e-6,max_iter=100)
```

输出

```python
Converge after 16 iterations
x^(16) = [ 1.72603719  0.3953224  -0.1321869   0.1370697 ]
relative error = 6.135494817040176e-07
```

答案

收敛，因为谱半径约等于 0.4 < 1
