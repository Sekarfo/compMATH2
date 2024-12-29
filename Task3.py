import numpy as np

# Step 1: Define the augmented matrix
augmented_matrix = np.array([
    [1, 1, 1, 9],
    [2, -3, 4, 13],
    [3, 4, 5, 40]
], dtype=float)
# Step 2: Perform Gauss-Jordan elimination
n = len(augmented_matrix)

for i in range(n):
    # Pivoting: Make the diagonal element 1
    pivot = augmented_matrix[i, i]
    augmented_matrix[i] = augmented_matrix[i] / pivot

    # Eliminate all other entries in column i
    for j in range(n):
        if j != i:
            factor = augmented_matrix[j, i]
            augmented_matrix[j] -= factor * augmented_matrix[i]
# Step 3: Extract the solution
solution = augmented_matrix[:, -1]  # Last column contains the solution
# Step 4: Print the results
print("Final Diagonal Matrix:")
print(augmented_matrix)

print("\nSolution Vector (x, y, z):")
print(solution)
