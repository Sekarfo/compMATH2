import numpy as np

# Step 1: Define the system
A = np.array([
    [5, 1, 1],
    [1, 4, 2],
    [1, 1, 5]
], dtype=float)

b = np.array([10, 12, 15], dtype=float)
x_initial = np.array([0, 0, 0], dtype=float)
tolerance = 1e-5
max_iterations = 100


def relaxation_method(A, b, x_initial, omega, tolerance, max_iterations):
    n = len(b)
    x = np.copy(x_initial)
    iterations = 0
    table = []

    for k in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            # Compute the sum of known and remaining values
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            # Apply the relaxation formula
            x_new[i] = (1 - omega) * x[i] + (omega * (b[i] - sum1 - sum2)) / A[i, i]

        table.append((iterations + 1, *x_new))
        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            break

        x = x_new
        iterations += 1

    return x_new, iterations, table
# Step 2: Solve the system for ω = 1.1 and ω = 1.5
omega_values = [1.1, 1.5]
solutions = {}

for omega in omega_values:
    solution, iterations, table = relaxation_method(A, b, x_initial, omega, tolerance, max_iterations)
    solutions[omega] = {
        "solution": solution,
        "iterations": iterations,
        "table": table
    }
    print(f"Results for ω = {omega}:")
    print(f"Solution: {solution}")
    print(f"Iterations: {iterations}\n")
