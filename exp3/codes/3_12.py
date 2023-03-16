import numpy as np
from scipy.interpolate import interp1d

x = (
    np.array(
        [
            0,
            2,
            4,
            5,
            6,
            7,
            8,
            9,
            10.5,
            11.5,
            12.5,
            14,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
        ]
    )
    * 60.0
)
y = np.array([2, 2, 0, 2, 5, 8, 25, 12, 5, 10, 12, 7, 9, 28, 22, 10, 9, 11, 8, 9, 3])
print(np.trapz(y, x))

new_x = np.arange(0, 1440.1, 0.1)
f = interp1d(x, y, kind="cubic")
new_y = f(new_x)
print(np.trapz(new_y, new_x))

import matplotlib.pyplot as plt

plt.plot(x, y, label="linear")
plt.plot(new_x, new_y, label="cubic")
plt.legend()
plt.show()
