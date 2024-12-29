import numpy as np
import pandas as pd
# Step 1: Define the system
A = np.array([
    [8, -3, 2],
    [4, 11, -1],
    [6, 3, 12]
], dtype=float)

b = np.array([20, 33, 36], dtype=float)

# Initial guess
x = np.array([0, 0, 0], dtype=float)
tolerance = 1e-5
max_iterations = 100
n = len(b)


# Step 2: Gauss-Seidel iteration
iteration = 0
table = []

while iteration < max_iterations:
    x_new = np.copy(x)
    for i in range(n):
        sum1 = np.dot(A[i, :i], x_new[:i])  # Sum of earlier updated values
        sum2 = np.dot(A[i, i + 1:], x[i + 1:])  # Sum of remaining values

        # Update the current variable
        x_new[i] = (b[i] - sum1 - sum2) / A[i, i]

    table.append((iteration + 1, *x_new))
    if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
        break

    x = x_new
    iteration += 1
# Step 3: Display the results

# Create a DataFrame for the table
df = pd.DataFrame(table, columns=["Iteration", "x1", "x2", "x3"])
print("Table of Iterations:")
print(df)
print("\nFinal Solution:")
print(x_new)
