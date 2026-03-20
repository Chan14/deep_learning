import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a simple grid of points
x = np.linspace(-3, 3, 30)
y = np.linspace(-3, 3, 30)
X, Y = np.meshgrid(x, y)
grid = np.vstack([X.ravel(), Y.ravel()])

# Linear and nonlinear transformations
W1 = np.array([[1, -0.5], [0.5, 1]])
W2 = np.array([[1, 0.5], [-0.5, 1]])


def relu(z):
    return np.maximum(0, z)


# Store transformations step by step
Z1 = W1 @ grid
A1 = relu(Z1)
Z2 = W2 @ A1

frames = [grid, Z1, A1, Z2]
colors = ["gray", "blue", "green", "red"]
titles = ["Original Grid", "After Linear Layer 1", "After ReLU", "After Linear Layer 2"]

fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(frames[0][0, :], frames[0][1, :], c=colors[0], s=30, alpha=0.6)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect("equal")
ax.grid(True)
title = ax.set_title(titles[0])


def update(frame):
    sc.set_offsets(frames[frame].T)
    sc.set_color(colors[frame])
    title.set_text(titles[frame])
    return sc, title


anim = FuncAnimation(fig, update, frames=4, interval=1000, blit=True)
plt.close()  # Prevents static plot display in notebook output
anim
