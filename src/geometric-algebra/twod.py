# %%


# %% [markdown]
# Geometric Algebra Notes
# -----------------------

# %% [markdown]
# Import a geometric algebra package


# %%
import sympy
from galgebra.ga import Ga


# %% [markdown]
# Define the coordinate system
# ----------------------------
#
# \\[ \vec{rotate}_{e_{first}}^{e_{second}}(\begin{bmatrix} {a_1} \\
#   {a_2}
#   \end{bmatrix}; \theta ) = \begin{bmatrix}
#      cos(\theta) \\
#      sin(\theta) \\
# \end{bmatrix}  * {a_1}  + \begin{bmatrix}
#      -sin(\theta) \\
#      cos(\theta) \\
# \end{bmatrix}  * {a_2}
# \\]
#
# \\[  = \begin{bmatrix}
#      cos(\theta) * {a_1} + -sin(\theta) * {a_2}  \\
#      sin(\theta) * {a_1} + cos(\theta) * {a_2}  \\
# \end{bmatrix}
# \\]
#
# \\[ = \begin{bmatrix}
#      {a_1} \\
#      {a_2} \\
# \end{bmatrix}  * cos(\theta)  + \begin{bmatrix}
#      -{a_2} \\
#      {a_1} \\
# \end{bmatrix}  * sin(\theta)
# \\]
#
# \\[ = \begin{bmatrix}
#      {a_1} \\
#      {a_2} \\
# \end{bmatrix}  * cos(\theta)  + \begin{bmatrix}
#      -{a_2} \\
#      0 \\
# \end{bmatrix}  * sin(\theta) + \begin{bmatrix}
#      0 \\
#      {a_1} \\
# \end{bmatrix}  * sin(\theta)
# \\]
#
# \\[ = \begin{bmatrix}
#      {a_1} \\
#      {a_2} \\
# \end{bmatrix}  * cos(\theta)  + \begin{bmatrix}
#      1 \\
#      0 \\
# \end{bmatrix}  * -{a_2}  * sin(\theta) + \begin{bmatrix}
#      0 \\
#      1 \\
# \end{bmatrix}  * {a_1}   * sin(\theta)
# \\]
#
# \\[ = \begin{bmatrix}
#      {a_1} \\
#      {a_2} \\
# \end{bmatrix}  * cos(\theta)  + {e_2} \wedge {e_1}  * -{a_2}  * sin(\theta) + {e_1} \wedge {e_2} * {a_1}   * sin(\theta)
# \\]
#
# \\[ = \begin{bmatrix}
#      {a_1} \\
#      {a_2} \\
# \end{bmatrix}  * cos(\theta)  + {e_1} \wedge {e_2}  * {a_2}  * sin(\theta) + {e_1} \wedge {e_2} * {a_1}   * sin(\theta)
# \\]


# %%
xyz = (x, y, z) = sympy.symbols("x y z", real=True)
o3d = Ga("e_x e_y e_z", g=[1, 1, 1], coords=xyz)
e_x, e_y, e_z = o3d.mv()
