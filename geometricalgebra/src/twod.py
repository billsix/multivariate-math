# %% Rotate [cell type] key="value"

#
# Copyright (c) 2023 William Emerison Six
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# %% [markdown]
# Geometric Algebra Notes
# -----------------------

# %% [markdown]
# Import a geometric algebra package


# %%
import sympy
import galgebra
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


# %%
xyz = (x, y, z) = sympy.symbols("x y z", real=True)
o3d = Ga("e_x e_y e_z", g=[1, 1, 1], coords=xyz)
e_x, e_y, e_z = o3d.mv()

