import numpy as np
import matplotlib.pyplot as plt

# Create a simple grid of points
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
grid = np.vstack([X.ravel(), Y.ravel()])

# Linear transformation (2x2 matrix)
W_linear = np.array([[1.2, 0.5], [-0.3, 1.0]])
linear_transformed = W_linear @ grid


# Non-linear transformation: linear -> ReLU -> linear
def relu(z):
    return np.maximum(0, z)


# First linear layer
W1 = np.array([[1, -0.5], [0.5, 1]])
Z1 = W1 @ grid

# Apply ReLU
A1 = relu(Z1)

# Second linear layer
W2 = np.array([[1, 0.5], [-0.5, 1]])
nonlinear_transformed = W2 @ A1

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original grid
axes[0].scatter(grid[0, :], grid[1, :], color="gray", alpha=0.3, s=10)
axes[0].scatter(
    linear_transformed[0, :], linear_transformed[1, :], color="blue", alpha=0.5, s=10
)
axes[0].set_title("Linear Transformation")
axes[0].set_aspect("equal")
axes[0].grid(True)

# Non-linear grid
axes[1].scatter(grid[0, :], grid[1, :], color="gray", alpha=0.3, s=10)
axes[1].scatter(
    nonlinear_transformed[0, :],
    nonlinear_transformed[1, :],
    color="red",
    alpha=0.5,
    s=10,
)
axes[1].set_title("2-Layer Network (ReLU non-linearity)")
axes[1].set_aspect("equal")
axes[1].grid(True)

plt.show()
