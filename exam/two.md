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

