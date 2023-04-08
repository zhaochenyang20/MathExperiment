import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.integrate import simps

np.set_printoptions(precision=15)

x = np.array([0, 3, 5, 7, 9, 11, 12, 13, 14, 15])
y1 = np.array([0, 1.8, 2.2, 2.7, 3.0, 3.1, 2.9, 2.5, 2.0, 1.6])
y2 = np.array([0, 1.2, 1.7, 2.0, 2.1, 2.0, 1.8, 1.2, 1.0, 1.6])

all_x = np.arange(0, 15.1, 0.1)
all_y1 = interp1d(x, y1, kind="cubic")(all_x)
all_y2 = interp1d(x, y2, kind="cubic")(all_x)

print("The area of the fitted area is " + str(np.trapz(all_y1 - all_y2, all_x)))
print("The area of the fitted area is " + str(simps(all_y1 - all_y2, all_x)))
