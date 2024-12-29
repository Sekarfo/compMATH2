import numpy as np

# Step 1: Define the system
A = np.array([
    [2, 3, 1],
    [4, 11, -1],
    [-2, 1, 7]
], dtype=float)

b = np.array([10, 33, 15], dtype=float)
# Step 2: Perform Gaussian elimination with pivoting
n = len(b)

# Forward elimination
for k in range(n):
    # Pivoting: Find the row with the largest element in column k
    max_row = np.argmax(np.abs(A[k:, k])) + k
    if k != max_row:
        # Swap rows in A and b
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

    # Eliminate entries below the pivot
    for i in range(k + 1, n):
        factor = A[i, k] / A[k, k]
        A[i, k:] -= factor * A[k, k:]
        b[i] -= factor * b[k]
# Step 3: Back substitution to solve the system
x = np.zeros_like(b)

for i in range(n - 1, -1, -1):
    x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
# Step 4: Print the results
print("Upper Triangular Matrix A:")
print(A)
print("\nModified Vector b:")
print(b)
print("\nSolution Vector x:")
print(x)
