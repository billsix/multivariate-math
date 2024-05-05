# %% [markdown]
# Plotting
# --------
# %%
import sympy
from galgebra.printer import latex
import matplotlib.pyplot as plt
import numpy as np


# tell sympy to use our printing by default
sympy.init_printing(latex_printer=latex, use_latex="mathjax")


# %% [markdown]
# Make test for if this is notebook
# ---------------------------------
# %%
def is_notebook() -> bool:
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


# %% [markdown]
# Do plot in notebook
# -------------------
# %%
fig, ax = plt.subplots()

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2 * np.pi * t)
(line,) = ax.plot(t, s, lw=2)

ax.annotate(
    "local max",
    xy=(2, 1),
    xytext=(3, 1.5),
    arrowprops=dict(facecolor="black", shrink=0.05),
)
ax.set_ylim(-2, 2)
plt.show()

# %% [markdown]
# Do vector
# ---------
# %%
# Vector components
x = 2
y = 4
z = 6

# Create a figure and axis for the number lines
fig, ax = plt.subplots()

# Plot the x-component on its number line (top-most)
ax.hlines(2, 0, x, colors="r", label=f"x = {x}")

# Plot the y-component on its number line (middle)
ax.hlines(1, 0, y, colors="g", label=f"y = {y}")

# Plot the z-component on its number line (bottom-most)
ax.hlines(0, 0, z, colors="b", label=f"z = {z}")
ax.set_xlim(0, max(x, y, z) + 1)  # Set the x-axis limit

# Add tick marks for every 1 unit and label them
for i in range(int(max(x, y, z)) + 1):
    ax.vlines(i, -0.1, 0.1, colors="k", alpha=0.7)
    ax.vlines(i, 0.9, 1.1, colors="k", alpha=0.7)
    ax.vlines(i, 1.9, 2.1, colors="k", alpha=0.7)
    ax.text(i, -0.3, str(i), ha="center", va="center", fontsize=10)  # Label tick marks

# Set y-axis ticks and labels
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["Z", "Y", "X"])

# Set the title
ax.set_title("Vector")

# Move the legend outside of the plot
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

# Remove x-axis ticks and labels
ax.set_xticks([])
ax.set_xticklabels([])

# Show the plot
plt.grid()
plt.show()
# %% [markdown]
# Animated Plot (for non-notebook)
# --------------------------------
# %%
if not is_notebook():
    if __name__ == "__main__":
        import matplotlib.pyplot as plt
        import numpy as np

        import matplotlib.animation as animation

        fig, ax = plt.subplots()

        x = np.arange(0, 2 * np.pi, 0.01)
        (line,) = ax.plot(x, np.sin(x))

        def animate(i):
            line.set_ydata(np.sin(x + i / 50))  # update the data.
            return (line,)

        ani = animation.FuncAnimation(fig, animate, interval=20, blit=True, save_count=50)

        plt.show()
