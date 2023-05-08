import numpy as np
from scipy.optimize import linprog


def solve_lp(Volumes, Prices, Scaling_factors, Constant, Quantity, A, b):
    """Solves a linear programming problem and calculates the total profit.

    Args:
        Volumes (numpy.ndarray): The volume of each product.
        Prices (numpy.ndarray): The price of each product.
        Scaling_factors (numpy.ndarray): Scaling factors for the constraints.
        Constant (int): A constant used in the profit calculation.
        Quantity (int): The quantity of products to produce.
        A (numpy.ndarray): The constraint matrix.
        b (numpy.ndarray): The constraint vector.

    Returns:
        tuple: A tuple containing the optimal solution and the total profit.
    """
    # Define the objective function coefficients for minimizing
    c_obj = [-Volumes[i] for i in range(1, len(Volumes))]
    # Define the upper bounds for the decision variables
    bounds = [(0, Prices[i]) for i in range(1, len(Prices))]
    # Solve the linear programming problem
    res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds)
    # Extract the optimal solution and objective function value
    x = res.x
    fval = res.fun
    # Calculate the total profit
    total_profit = Constant * (fval + np.dot(Volumes[1:], Prices[1:]))
    return x, total_profit


def define_constraints(Volumes, Prices, Scaling_factors, Quantity):
    """Defines the constraints for the linear programming problems.

    Args:
        Volumes (numpy.ndarray): The volume of each product.
        Prices (numpy.ndarray): The price of each product.
        Scaling_factors (numpy.ndarray): Scaling factors for the constraints.
        Quantity (int): The quantity of products to produce.

    Returns:
        list: A list of tuples containing the constraint matrix and vector for each problem.
    """
    # Define the first set of constraints
    A1 = np.array(
        [
            [Volumes[1], 0, 0],
            [Scaling_factors[0] * Volumes[1], Volumes[2], 0],
            [
                Scaling_factors[1] * Scaling_factors[0] * Volumes[1],
                Scaling_factors[1] * Volumes[2],
                Volumes[3],
            ],
        ]
    )
    b1 = np.array(
        [
            Quantity * (Volumes[0] + Volumes[1]) - Volumes[0] * Prices[0],
            Quantity * (Volumes[0] + Volumes[1] + Volumes[2])
            - Scaling_factors[0] * Volumes[0] * Prices[0],
            Quantity * (Volumes[0] + Volumes[1] + Volumes[2] + Volumes[3])
            - Scaling_factors[1] * Scaling_factors[0] * Volumes[0] * Prices[0],
        ]
    )

    # Define the second set of constraints
    A2 = np.array(
        [
            [Scaling_factors[0] * Volumes[1], 0, 0],
            [
                Scaling_factors[1] * Scaling_factors[0] * Volumes[1],
                Scaling_factors[1] * Volumes[2],
                0,
            ],
        ]
    )
    b2 = np.array(
        [
            Quantity * (Volumes[0] + Volumes[1])
            - Scaling_factors[0] * Volumes[0] * Prices[0],
            Quantity * (Volumes[0] + Volumes[1] + Volumes[2])
            - Scaling_factors[1] * Scaling_factors[0] * Volumes[0] * Prices[0],
        ]
    )

    return [(A1, b1), (A2, b2)]


# Initialize variables
Volumes = np.array([1000, 5, 5, 5])  # Volume
Prices = np.array([0.8, 100, 60, 50])  # Price
Scaling_factors = np.array([0.9, 0.6])  # Scaling factors

# Define parameters
Constant = 1
Quantity = 1

# Define constraints
constraints = define_constraints(Volumes, Prices, Scaling_factors, Quantity)

# Solve the linear programming problems and print the results
for i, (A, b) in enumerate(constraints):
    x, total_profit = solve_lp(
        Volumes, Prices, Scaling_factors, Constant, Quantity, A, b
    )
    print(f"Results of problem {i+1}:")
    print(f"x = {x}")
    print(f"Total profit = {total_profit}\n")
