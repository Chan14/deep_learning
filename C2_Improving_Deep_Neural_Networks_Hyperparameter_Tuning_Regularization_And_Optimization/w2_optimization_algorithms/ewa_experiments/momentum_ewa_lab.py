"""
================================================================================
EWA + MOMENTUM LAB  —  seeing the cancellation, step by step
================================================================================

Goal
----
Watch *exactly* what momentum does to a single gradient step. We will:

  1. Build a tiny problem: 5 training examples, 2 weights, a NARROW RAVINE
     (a loss valley that is steep across and shallow along).
  2. Run three optimizers on the SAME problem:
        RUN 1  Batch Gradient Descent      (full gradient, smooth baseline)
        RUN 2  Vanilla SGD (no momentum)   (one example at a time, noisy baseline)
        RUN 3  SGD WITH MOMENTUM           (THE EXPERIMENT)
  3. For RUN 3, at every step print the EWA update with real numbers, and split
     the new velocity into its two pieces so we can SEE which pieces cancel and
     which pieces add up.
  4. Make three plots:
        PLOT 1  The ravine (contour) + all three paths
        PLOT 2  The ravine in 3D, zoomed near one chosen step, with the vectors
        PLOT 3  The vector decomposition at that step, with the math written on it

Notation (every symbol defined up front)
----------------------------------------
  N            number of training examples            = 5
  n            number of weights (parameters)         = 2
  w = [w1, w2] the weight vector we are optimizing
  x^(i)        feature vector of example i            (length 2)
  y^(i)        target value of example i              (a number)
  g            a gradient vector                      (length 2, same shape as w)
  v            the VELOCITY = the EWA of past gradients
  alpha (a)    learning rate (step size)
  beta  (b)    EWA decay for momentum                 = 0.9  (Andrew Ng's default)

The momentum update (Ng's "EWA of gradients" form):
        v  <-  beta * v  +  (1 - beta) * g          # velocity = running average of gradients
        w  <-  w  -  alpha * v                        # step along the velocity, not the raw gradient
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")                 # headless: save figures to files
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

np.set_printoptions(precision=4, suppress=True)

# ------------------------------------------------------------------------------
# 1) THE PROBLEM  —  5 examples, 2 weights, deliberately a narrow ravine
# ------------------------------------------------------------------------------
# Feature 1 has large values  -> the loss is STEEP along w1  (the "walls")
# Feature 2 has small values   -> the loss is SHALLOW along w2 (the "floor")
X = np.array([
    [3.0,  0.55],
    [3.4, -0.48],
    [2.6,  0.52],
    [3.2, -0.42],
    [2.8,  0.40],
])
# Targets: a clean linear signal plus alternating noise so consecutive examples
# disagree about which way to push -> vanilla SGD will zig-zag.
y = X @ np.array([1.0, 2.0]) + np.array([0.5, -0.5, 0.5, -0.5, 0.5])

N = X.shape[0]            # 5 examples
n = X.shape[1]            # 2 weights

# Loss is  L(w) = (1 / 2N) * sum_i ( w . x^(i) - y^(i) )^2
def loss(w):
    residuals = X @ w - y                 # length-5 vector of (prediction - target)
    return 0.5 * np.mean(residuals ** 2)

# Gradient from ONE example i:   g^(i) = ( w . x^(i) - y^(i) ) * x^(i)
def example_gradient(w, i):
    residual_i = X[i] @ w - y[i]          # a single number
    return residual_i * X[i]              # length-2 vector

# Gradient from ALL examples (full batch): average of the per-example gradients
def batch_gradient(w):
    residuals = X @ w - y                 # length-5
    return (X.T @ residuals) / N          # length-2

# --- confirm the ravine geometry ---------------------------------------------
H = (X.T @ X) / N                          # curvature matrix (the Hessian of L)
eigvals, eigvecs = np.linalg.eigh(H)       # small eigenvalue = floor, large = wall
w_star = np.linalg.solve(X.T @ X, X.T @ y) # the true optimum (least squares)

print("=" * 78)
print("THE RAVINE")
print("=" * 78)
print(f"curvature eigenvalues : {eigvals}   (small = shallow floor, large = steep wall)")
print(f"condition number      : {eigvals[-1] / eigvals[0]:.1f}   (how narrow the ravine is)")
print(f"optimum  w*           : {w_star}   (where we want to end up)")
print()

# Shared settings
w0    = np.array([3.5, -1.5])   # starting weights (up on a wall, far down the floor)
alpha = 0.12                    # learning rate, shared by all three runs
beta  = 0.9                     # EWA decay for momentum


# ------------------------------------------------------------------------------
# RUN 1 — BATCH GRADIENT DESCENT  (baseline: the smooth, full-gradient path)
# ------------------------------------------------------------------------------
def run_batch_gd(steps):
    w = w0.copy()
    path = [w.copy()]
    for _ in range(steps):
        g = batch_gradient(w)       # full gradient
        w = w - alpha * g
        path.append(w.copy())
    return np.array(path)

# ------------------------------------------------------------------------------
# RUN 2 — VANILLA SGD  (baseline: one example per step, no memory -> noisy)
# ------------------------------------------------------------------------------
def run_vanilla_sgd(steps):
    w = w0.copy()
    path = [w.copy()]
    for t in range(steps):
        i = t % N                   # cycle through examples 0,1,2,3,4,0,1,...
        g = example_gradient(w, i)
        w = w - alpha * g
        path.append(w.copy())
    return np.array(path)

# ------------------------------------------------------------------------------
# RUN 3 — SGD WITH MOMENTUM  (THE EXPERIMENT: track the velocity, print the math)
# ------------------------------------------------------------------------------
def run_momentum(steps, verbose=True):
    w = w0.copy()
    v = np.zeros(n)                 # velocity starts at zero
    path = [w.copy()]
    records = []                    # we store the full state of every step

    if verbose:
        print("=" * 78)
        print("RUN 3 — SGD WITH MOMENTUM   (velocity v is the EWA of gradients)")
        print("formula every step:   v = beta*v_prev + (1-beta)*g   then   w = w - alpha*v")
        print(f"                      beta={beta}   alpha={alpha}")
        print("=" * 78)

    for t in range(steps):
        i = t % N
        g = example_gradient(w, i)          # raw gradient (what vanilla SGD would use)
        v_prev = v.copy()                   # velocity coming IN
        memory = beta * v_prev              # the "remember the past" piece
        fresh  = (1 - beta) * g             # the "new information" piece
        v = memory + fresh                  # velocity going OUT  (the EWA)
        w = w - alpha * v                   # take the step along the velocity

        records.append(dict(t=t, i=i, g=g.copy(), v_prev=v_prev.copy(),
                            memory=memory.copy(), fresh=fresh.copy(),
                            v_new=v.copy(), w_after=w.copy()))
        path.append(w.copy())

        if verbose:
            # The EWA formula printed WITH the actual numbers, every step:
            print(f"step {t:3d} | ex {i} | "
                  f"v = {beta}*[{v_prev[0]:+.3f},{v_prev[1]:+.3f}] "
                  f"+ {1-beta:.1f}*[{g[0]:+.3f},{g[1]:+.3f}] "
                  f"= [{v[0]:+.3f},{v[1]:+.3f}] | "
                  f"raw g=[{g[0]:+.3f},{g[1]:+.3f}]  ->  update w -= {alpha}*v")

    return np.array(path), records


# ------------------------------------------------------------------------------
# Run everything
# ------------------------------------------------------------------------------
BATCH_STEPS = 50
SGD_STEPS   = 50
MOM_STEPS   = 120     # momentum needs more steps: the (1-beta)=0.1 factor shrinks
                      # each step ~10x, so it takes longer to build up speed.

path_batch = run_batch_gd(BATCH_STEPS)
path_sgd   = run_vanilla_sgd(SGD_STEPS)
path_mom, records = run_momentum(MOM_STEPS, verbose=True)

# The one step we put under the microscope in PLOT 2 and PLOT 3:
FEATURE = 64
rec = records[FEATURE]
pos = path_mom[FEATURE]            # the weights AT which step 64's gradient is taken

print()
print("=" * 78)
print(f"ZOOM ON STEP {FEATURE}   (example {rec['i']})  —  the decomposition")
print("=" * 78)
print(f"position w           : [{pos[0]:+.4f}, {pos[1]:+.4f}]")
print(f"raw gradient g       : [{rec['g'][0]:+.4f}, {rec['g'][1]:+.4f}]   (what vanilla SGD would follow)")
print(f"velocity in  v_prev  : [{rec['v_prev'][0]:+.4f}, {rec['v_prev'][1]:+.4f}]")
print(f"memory  beta*v_prev  : [{rec['memory'][0]:+.4f}, {rec['memory'][1]:+.4f}]")
print(f"fresh   (1-b)*g      : [{rec['fresh'][0]:+.4f}, {rec['fresh'][1]:+.4f}]")
print(f"velocity out v_new   : [{rec['v_new'][0]:+.4f}, {rec['v_new'][1]:+.4f}]")
print("-" * 78)
print(f"  STEEP axis w1 :  memory {rec['memory'][0]:+.3f}  +  fresh {rec['fresh'][0]:+.3f}  "
      f"=  {rec['v_new'][0]:+.3f}   <-- opposite signs => THEY CANCEL")
print(f"  FLOOR axis w2 :  memory {rec['memory'][1]:+.3f}  +  fresh {rec['fresh'][1]:+.3f}  "
      f"=  {rec['v_new'][1]:+.3f}   <-- same signs => THEY ADD UP")
print("=" * 78)


# ==============================================================================
# PLOT 1 — the ravine (contour) with all three paths
# ==============================================================================
pad = 0.6
w1_min = min(path_batch[:,0].min(), path_sgd[:,0].min(), path_mom[:,0].min(), w_star[0]) - pad
w1_max = max(path_batch[:,0].max(), path_sgd[:,0].max(), path_mom[:,0].max(), w_star[0]) + pad
w2_min = min(path_batch[:,1].min(), path_sgd[:,1].min(), path_mom[:,1].min(), w_star[1]) - pad
w2_max = max(path_batch[:,1].max(), path_sgd[:,1].max(), path_mom[:,1].max(), w_star[1]) + pad

g1 = np.linspace(w1_min, w1_max, 220)
g2 = np.linspace(w2_min, w2_max, 220)
G1, G2 = np.meshgrid(g1, g2)
LOSS = np.zeros_like(G1)
for a_idx in range(G1.shape[0]):
    for b_idx in range(G1.shape[1]):
        LOSS[a_idx, b_idx] = loss(np.array([G1[a_idx, b_idx], G2[a_idx, b_idx]]))

fig1, ax1 = plt.subplots(figsize=(9, 7))
cs = ax1.contour(G1, G2, LOSS, levels=np.logspace(-1.0, 2.2, 22), cmap="Greys", linewidths=0.8)
ax1.plot(path_batch[:,0], path_batch[:,1], "-o", ms=3, lw=1.4, color="#1f77b4", label="Batch GD")
ax1.plot(path_sgd[:,0],   path_sgd[:,1],   "-o", ms=3, lw=1.4, color="#ff7f0e", label="Vanilla SGD")
ax1.plot(path_mom[:,0],   path_mom[:,1],   "-o", ms=2, lw=1.4, color="#2ca02c", label="SGD + Momentum")
ax1.plot(*w_star, "*", ms=20, color="crimson", label="optimum w*")
ax1.plot(*w0, "ks", ms=8, label="start")
ax1.scatter(*pos, s=120, facecolors="none", edgecolors="black", linewidths=1.8, zorder=5)
ax1.annotate(f"step {FEATURE}", pos, textcoords="offset points", xytext=(8, 8), fontsize=9)
ax1.set_xlabel("w1   (steep axis — the ravine walls)")
ax1.set_ylabel("w2   (shallow axis — the ravine floor)")
ax1.set_title("PLOT 1 — the ravine and three paths\n"
              "SGD bounces across the walls; momentum glides down the floor")
ax1.legend(loc="upper right", framealpha=0.95)
fig1.tight_layout()
fig1.savefig("plot1_ravine_paths.png", dpi=130)

# ==============================================================================
# PLOT 2 — 3D surface, zoomed near the chosen step, with the vectors on it
# ==============================================================================
zoom = 0.9
z1 = np.linspace(pos[0] - zoom, pos[0] + zoom, 60)
z2 = np.linspace(pos[1] - zoom, pos[1] + zoom, 60)
Z1, Z2 = np.meshgrid(z1, z2)
ZL = np.zeros_like(Z1)
for a_idx in range(Z1.shape[0]):
    for b_idx in range(Z1.shape[1]):
        ZL[a_idx, b_idx] = loss(np.array([Z1[a_idx, b_idx], Z2[a_idx, b_idx]]))

fig2 = plt.figure(figsize=(10, 7.5))
ax2 = fig2.add_subplot(111, projection="3d")
ax2.plot_surface(Z1, Z2, ZL, cmap="viridis", alpha=0.55, linewidth=0, antialiased=True)

z_here = loss(pos)                                   # height of the surface at our point
ax2.scatter([pos[0]], [pos[1]], [z_here], color="black", s=60, label="current weights")

# The raw gradient is ~8x longer than the velocity, so if we drew true lengths the
# velocity arrow would be invisible. Here we draw each arrow at the SAME display
# length to compare DIRECTIONS, and put the true length in the label.
disp = 0.7   # display length for every arrow
def draw3d_dir(vec, color, label):
    u = vec / (np.linalg.norm(vec) + 1e-12)
    ax2.quiver(pos[0], pos[1], z_here, u[0]*disp, u[1]*disp, 0,
               color=color, lw=3.0, arrow_length_ratio=0.25, label=label)

draw3d_dir(rec["g"],     "#d62728", f"raw gradient g  (|g|={np.linalg.norm(rec['g']):.2f}) -> up the wall")
draw3d_dir(rec["v_prev"],"#9467bd", f"velocity in  v_prev (|v|={np.linalg.norm(rec['v_prev']):.2f})")
draw3d_dir(rec["v_new"], "#2ca02c", f"velocity out v_new (|v|={np.linalg.norm(rec['v_new']):.2f}) -> down the floor")

ax2.set_xlabel("w1 (steep)")
ax2.set_ylabel("w2 (floor)")
ax2.set_zlabel("loss")
ax2.set_title(f"PLOT 2 — the valley in 3D near step {FEATURE}\n"
              "raw gradient points up a wall; momentum's velocity points down the floor")
ax2.legend(loc="upper left", fontsize=8)
ax2.view_init(elev=38, azim=-60)
fig2.tight_layout()
fig2.savefig("plot2_valley_3d.png", dpi=130)

# ==============================================================================
# PLOT 3 — the vector decomposition at the chosen step, math written on it
# ==============================================================================
fig3, ax3 = plt.subplots(figsize=(9.5, 8.5))

O = np.array([0.0, 0.0])    # we draw all vectors from a common origin
def arrow(ax, start, vec, color, label, lw=2.6, ls="-", alpha=1.0):
    ax.annotate("", xy=(start[0]+vec[0], start[1]+vec[1]), xytext=(start[0], start[1]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, alpha=alpha))
    ax.plot([], [], color=color, lw=lw, ls=ls, label=label)   # for legend

memory = rec["memory"]
fresh  = rec["fresh"]
v_new  = rec["v_new"]
g      = rec["g"]

# the two pieces of the EWA, drawn TIP-TO-TAIL so they add up to v_new:
arrow(ax3, O,      memory, "#1f77b4", "memory  beta*v_prev")
arrow(ax3, memory, fresh,  "#ff7f0e", "fresh  (1-beta)*g")
# the resultant velocity (the diagonal):
arrow(ax3, O, v_new, "#2ca02c", "velocity out  v_new", lw=3.4)

# axis through origin so we can read the two axes (steep w1 / floor w2)
ax3.axhline(0, color="gray", lw=0.6)
ax3.axvline(0, color="gray", lw=0.6)

# math written on the figure
txt = (
    r"$v_{new} = \beta\, v_{prev} + (1-\beta)\, g$" "\n"
    f"$\\beta$ = {beta},   raw $g$ = [{g[0]:+.2f}, {g[1]:+.2f}]\n"
    f"memory  $\\beta v_{{prev}}$ = [{memory[0]:+.3f}, {memory[1]:+.3f}]\n"
    f"fresh  $(1-\\beta)g$ = [{fresh[0]:+.3f}, {fresh[1]:+.3f}]\n"
    f"$v_{{new}}$ = [{v_new[0]:+.3f}, {v_new[1]:+.3f}]\n"
    "\n"
    f"STEEP w1: {memory[0]:+.3f} + ({fresh[0]:+.3f}) = {v_new[0]:+.3f}  (cancel)\n"
    f"FLOOR w2: {memory[1]:+.3f} + ({fresh[1]:+.3f}) = {v_new[1]:+.3f}  (add up)\n"
    "\n"
    f"(raw g is {np.linalg.norm(g)/np.linalg.norm(v_new):.0f}x longer than v_new\n"
    " and points off-frame -- momentum shrank it)"
)
ax3.text(0.02, 0.98, txt, transform=ax3.transAxes, va="top", ha="left",
         fontsize=10, family="monospace",
         bbox=dict(boxstyle="round", fc="#fffbe6", ec="gray"))

lim = 0.6
ax3.set_xlim(-lim, lim)
ax3.set_ylim(-lim, lim)
ax3.set_aspect("equal")
ax3.set_xlabel("w1 component  (steep axis)")
ax3.set_ylabel("w2 component  (floor axis)")
ax3.set_title(f"PLOT 3 — velocity decomposition at step {FEATURE}\n"
              "blue + orange (tip-to-tail) = green.  Watch the steep (w1) pieces cancel.")
ax3.legend(loc="lower right", fontsize=9, framealpha=0.95)
ax3.grid(True, alpha=0.25)
fig3.tight_layout()
fig3.savefig("plot3_vector_decomposition.png", dpi=130)

print("\nSaved: plot1_ravine_paths.png, plot2_valley_3d.png, plot3_vector_decomposition.png")
