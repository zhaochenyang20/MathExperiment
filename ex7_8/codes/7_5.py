import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

num_iteration = 0

index = np.array(
    [
        [4, 1],
        [12, 1],
        [13, 1],
        [17, 1],
        [21, 1],
        [5, 2],
        [16, 2],
        [17, 2],
        [25, 2],
        [5, 3],
        [20, 3],
        [21, 3],
        [24, 3],
        [5, 4],
        [12, 4],
        [24, 4],
        [8, 6],
        [13, 6],
        [19, 6],
        [25, 6],
        [8, 7],
        [14, 7],
        [16, 7],
        [20, 7],
        [21, 7],
        [14, 8],
        [18, 8],
        [13, 9],
        [15, 9],
        [22, 9],
        [11, 10],
        [13, 10],
        [19, 10],
        [20, 10],
        [22, 10],
        [18, 11],
        [25, 11],
        [15, 12],
        [17, 12],
        [15, 13],
        [19, 13],
        [15, 14],
        [16, 14],
        [20, 16],
        [23, 16],
        [18, 17],
        [19, 17],
        [20, 19],
        [23, 19],
        [24, 19],
        [23, 21],
        [23, 22],
    ]
)

distance = np.array(
    [
        0.9607,
        0.4399,
        0.8143,
        1.3765,
        1.2722,
        0.5294,
        0.6144,
        0.3766,
        0.6893,
        0.9488,
        0.8,
        1.109,
        1.1432,
        0.4758,
        1.3402,
        0.7006,
        0.4945,
        1.0559,
        0.681,
        0.3587,
        0.3351,
        0.2878,
        1.1346,
        0.387,
        0.7511,
        0.4439,
        0.8363,
        0.3208,
        0.1574,
        1.2736,
        0.5781,
        0.9254,
        0.6401,
        0.2467,
        0.4727,
        1.384,
        0.4366,
        1.0307,
        1.3904,
        0.5725,
        0.766,
        0.4394,
        1.0952,
        1.0422,
        1.8255,
        1.4325,
        1.0851,
        0.4995,
        1.2277,
        1.1271,
        0.706,
        0.8052,
    ]
)

def construct_points(vector, length, selection, dimension):
    x_coords = []
    y_coords = []
    for i in range(length):
        if selection[i][dimension] == 1:
            x_coords.append(0)
            y_coords.append(0)
        elif selection[i][dimension] == 2:
            x_coords.append(0)
            y_coords.append(vector[0])
        else:
            x_coords.append(vector[2 * selection[i][dimension] - 5])
            y_coords.append(vector[2 * selection[i][dimension] - 4])
    return np.array(x_coords), np.array(y_coords)


def objective(X):
    global num_iteration
    num_iteration += 1
    xi, yi = construct_points(X, index.shape[0], index, 0)
    xj, yj = construct_points(X, index.shape[0], index, 1)
    distances = np.sqrt((xi - xj) ** 2.0 + (yi - yj) ** 2.0)
    F = np.sum((distances - distance) ** 2.0)
    return F

minX = np.zeros(25 * 2 - 2 - 1)
minVal = objective(minX)

methods = ["BFGS", "L-BFGS-B", "CG"]

from pathlib import Path

result_path = Path.cwd() / "7_5_result.npy"
if not (result_path.exists() and (Path.cwd() / "7_5_result.txt").exists()):
    results = []
    for epoch in range(50):
        print(f"epoch: {epoch}")
        result_dict = {}
        for method in methods:
            num_iteration = 0
            res = minimize(
                objective,
                np.random.rand(25 * 2 - 2 - 1),
                method=method,
                options={"maxiter": 10000},
            )
            if res.fun < minVal:
                minX = res.x
                minVal = res.fun
            print(f"{method}: {res.fun}, {num_iteration}")
            with open(Path.cwd() / "7_5_result.txt", "a") as f:
                f.write(f"{method}: {res.fun}, {num_iteration}\n")
            result_dict[method] = [res.fun, num_iteration]
        result_dict["epoch"] = epoch
        with open(Path.cwd() / "7_5_result.txt", "a") as f:
            f.write("epoch: " + str(epoch) + "\n")
            f.write("minX: " + str(minX) + "\n")
            f.write("minVal: " + str(minVal) + "\n")
        results.append(result_dict)
    all_results = {"results": results, "minX": minX, "minVal": minVal}
    np.save(result_path, all_results)
else:
    all_results = np.load(result_path, allow_pickle=True).item()
    results = all_results["results"]
    minX = all_results["minX"]
    minVal = all_results["minVal"]

x = np.zeros(25)
x[1:] = minX[::2]
y = np.zeros(25)
y[2:] = minX[1::2]

print(f"Minimum value: {minVal}")

plt.scatter(x, y)
plt.show()
