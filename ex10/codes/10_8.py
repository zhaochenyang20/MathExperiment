# Import necessary libraries
from pulp import *
import numpy as np


# Define the problem to be solved
def solve(half_limit=None):
    # Define the problem
    prob = LpProblem("Employment", LpMinimize)

    # Set the parameters
    work_time = list(range(1, 9))  # Work hours
    n = np.array([4, 3, 4, 6, 5, 6, 8, 8])  # Number of servers needed in each time slot
    s_f = 100  # Daily salary of full-time server
    s_h = 40  # Daily salary of part-time server

    # Define the constraints
    lunch_time = [4, 5]
    l = LpVariable.dicts("l", lunch_time, lowBound=0, cat="Integer")
    half_start_time = list(range(1, 6))
    h = LpVariable.dicts("h", half_start_time, lowBound=0, cat="Integer")

    # Define the objective function
    prob += lpSum(
        [s_f * l[i] for i in lunch_time] + [s_h * h[i] for i in half_start_time]
    )

    # Add the constraints
    for i in work_time:
        prob += (
            lpSum([l[j] for j in lunch_time if j != i])
            + lpSum([h[j] for j in half_start_time if j <= i <= j + 3])
            >= n[i - 1]
        )

    if half_limit is not None:
        prob += lpSum([h[i] for i in half_start_time]) <= half_limit

    # Solve the problem
    prob.solve()

    # Print the results
    print("Solution:")
    for i in lunch_time:
        print(f"l[{i}] = {value(l[i])}")
    for i in half_start_time:
        print(f"h[{i}] = {value(h[i])}")
    print("Objective function value: ", value(prob.objective))
    print("\n")


# Call the function to solve the problem with various parameters
solve(3)
solve(0)
solve(None)
