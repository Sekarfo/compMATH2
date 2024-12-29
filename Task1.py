import numpy as np

# A is the 3x3 coefficient matrix
A = np.array([
    [10, -1, -2],
    [-2, 10, -1],
    [-1, -2, 10]
], dtype=float)

# b is the right-hand side vector
b = np.array([5, -6, 15], dtype=float)

# Jacobi parameters
max_iterations = 100
tolerance = 1e-6  # or something you choose, e.g. 1e-6S

# Initial guess x^(0)
x_old = np.array([0, 0, 0], dtype=float)

n = len(b)
x_new = np.zeros_like(x_old)


def jacobi_step(A, b, x_old):
    """Perform one Jacobi iteration step."""
    n = len(b)
    x_new = np.zeros_like(x_old)
    for i in range(n):
        # sum_over_j = sum(a_{i,j} * x_old_j for j != i)
        sum_over_j = 0
        for j in range(n):
            if j != i:
                sum_over_j += A[i, j] * x_old[j]
        x_new[i] = (b[i] - sum_over_j) / A[i, i]
    return x_new
iteration = 0

while iteration < max_iterations:
    x_new = jacobi_step(A, b, x_old)

    # Check convergence criterion: ||x_new - x_old|| < tolerance
    if np.linalg.norm(x_new - x_old, ord=np.inf) < tolerance:
        break

    x_old = x_new.copy()
    iteration += 1

print("Jacobi Method Solution:")
print("x =", x_new)
print("Number of iterations:", iteration)
