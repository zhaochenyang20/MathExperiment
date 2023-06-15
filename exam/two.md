# 1

```python
import numpy as np
from scipy.optimize import linprog

# Define the objective function
c = np.array([-64, -54])

# Define the inequality constraints
A_ub = np.array([[3, 0], [12, 8]])
b_ub = np.array([80, 480])

# Define the equality constraints
A_eq = np.array([[1, 1]])
b_eq = np.array([55])

# Define the bounds
bounds = [(0, 55), (0, 55)]

# Call the linprog function
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

print(res)
```

1. 模型：

$3x_1\leq80,x_1+x_2=55,12x_1+8x_2<=480$，求 $64x+54y$ 的 max。

2. 3070；没有工时限制，求得最大利润，除以工时，得到加班费 5.921；

```py
import numpy as np
from scipy.optimize import linprog

# Define the objective function
c = np.array([-64, -54])

# Define the inequality constraints
A_ub = np.array([[3, 0]])
b_ub = np.array([80])

# Define the equality constraints
A_eq = np.array([[1, 1]])
b_eq = np.array([55])

# Define the bounds
bounds = [(0, 55), (0, 55)]

# Call the linprog function
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

print(res)

-----

In [7]: y = 55 - 26.66

In [8]: x = 26.66

In [9]: 12 * x + 8 * y
Out[9]: 546.64

In [10]: 3236.66 / 546.64
Out[10]: 5.921008341870335
```

3. 不改变

```python
import numpy as np
from scipy.optimize import linprog

# Define the objective function
c = np.array([-78, -54])

# Define the inequality constraints
A_ub = np.array([[3, 0], [12, 8]])
b_ub = np.array([80, 480])

# Define the equality constraints
A_eq = np.array([[1, 1]])
b_eq = np.array([55])

# Define the bounds
bounds = [(0, 55), (0, 55)]

# Call the linprog function
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

print(res)
```

# 3

```python
import numpy as np

# 计算向量 v = [1, 2, 3] 的2-范数

b = np.array([6, 3, 4, 7])
A = np.array([[5, -7, 0, 1],
             [-3, 22, 6, 2],
             [5, -1, 31, -1],
             [2, 1, 0, 23]])

delta = np.array([0, 0, 0, 0.1])
norm_b = np.linalg.norm(b)
cond_arr = np.linalg.cond(A)
norm_delta = np.linalg.norm(delta)
print(cond_arr * norm_delta / norm_b)
```

误差：0.09422673612193182

```py
import numpy as np

def split_matrix(A):
    n = A.shape[0]  # 方阵的维度
    D = np.diag(np.diag(A))  # 对角矩阵
    L = -np.tril(A, k=-1)  # 严格下三角矩阵
    U = -np.triu(A, k=1)  # 严格上三角矩阵
    return D, L, U


def jacobi(A, x0, b, error=1e-6):
    D, L, U = split_matrix(A)
    inv_D = np.linalg.inv(D)
    k = 0
    xk = x0
    while True:
        k += 1
        xk_1 = inv_D @ ((L + U) @ xk + b)
        xk = xk_1
        if np.linalg.norm(xk - xk_1, np.inf) < error:
            return xk_1, k


def gauss_seidel(A, x0, b, error=1e-6):
    D, L, U = split_matrix(A)
    inv_D_minus_L = np.linalg.inv(D - L)
    k = 0
    xk = x0
    while True:
        k += 1
        xk_1 = inv_D_minus_L @ (U @ xk + b)
        step = np.linalg.norm(xk - xk_1, np.inf)
        xk = xk_1
        print(k)
        print(xk)
        print(f"step {step}")
        if k == 5:
            return xk, k
        if step < error:
            return xk_1, k

b = np.array([6, 3, 4, 7])
A = np.array([[5, -7, 0, 1],
             [-3, 22, 6, 2],
             [5, -1, 31, -1],
             [2, 1, 0, 23]])

x0 = np.array([0, 0, 0, 0])

x_5, k = gauss_seidel(A, x0, b)
```

```py
5
[ 1.71597235  0.39264682 -0.1305711   0.13806124]
step 0.014587358828084485
```

收敛，是对角占优矩阵。