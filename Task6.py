import numpy as np

# Define the coefficient matrix A and right-hand side vector b
A = np.array([
    [1.001, 0.999],
    [1.002, 1.000]
], dtype=float)

b = np.array([2, 2.001], dtype=float)

# Solve for x
x = np.linalg.solve(A, b)

print("Coefficient matrix A:")
print(A)
print("RHS vector b:", b)
print("Solution x:", x)

# Let's also compute the condition number to illustrate how ill-conditioned it is
cond_number = np.linalg.cond(A)
print("Condition number of A:", cond_number)



# Copy the original matrix
A_perturbed = A.copy()
# Perturb one coefficient by a tiny amount
A_perturbed[0, 0] = 1.0015

# Solve the perturbed system
x_perturbed = np.linalg.solve(A_perturbed, b)

print("\nPerturbed coefficient matrix A_perturbed:")
print(A_perturbed)
print("Solution x_perturbed:", x_perturbed)
