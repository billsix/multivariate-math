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

# %%
xyz = (x, y, z) = sympy.symbols("x y z", real=True)
o3d = Ga("e_x e_y e_z", g=[1, 1, 1], coords=xyz)
e_x, e_y, e_z = o3d.mv()


# %% [markdown]
# Make Constants
# --------------
# %%
a1a2a3 = (a_1, a_2, a_3) = sympy.symbols("a_1 a_2 a_3", real=True)
b1b2b3 = (b_1, b_2, b_3) = sympy.symbols("b_1 b_2 b_3", real=True)


# %% [markdown]
# Define vectors with numbers
# ---------------------------
# %%
vec1 = 3 * e_x + 4 * e_y - 12 * e_z
vec2 = -3 * e_x + 12 * e_y + 4 * e_z

# %% [markdown]
# Define vectors with symbols
# ---------------------------
# %%
symvec1 = a_1 * e_x + a_2 * e_y + a_3 * e_z
symvec2 = b_1 * e_x + b_2 * e_y + b_3 * e_z


# %%
symvec1


# %% [markdown]
# Define basic operations on vectors/multivectors
# -----------------------------------------------
# %%
def project(self, onto):
    return (self | onto) / onto


galgebra.mv.Mv.project = project


# %%
def reject(self, away_from):
    return (self ^ away_from) / away_from


galgebra.mv.Mv.reject = reject


# %% [markdown]
# Define rotate in 3D
# -------------------
# %%
def rotate(self, from_vec, to_vec):
    plane = from_vec ^ to_vec

    parallel = self.project(onto=plane)
    perp = self.reject(away_from=plane)

    rotated_parallel = parallel * from_vec * to_vec / abs(from_vec) / abs(to_vec)
    return rotated_parallel + perp


galgebra.mv.Mv.rotate = rotate


# %% [markdown]
# Define rotate in 3D
# -------------------
#
# Inline definitions


# %%
def rotate(self, from_vec, to_vec):
    return self.project(onto=from_vec ^ to_vec) * from_vec * to_vec / abs(from_vec) / abs(to_vec) + self.reject(
        away_from=from_vec ^ to_vec
    )


galgebra.mv.Mv.rotate = rotate


# %% [markdown]
# Define rotate in 3D
# -------------------
#
# Inline definitions


# %%
def rotate(self, from_vec, to_vec):
    return (self | (from_vec ^ to_vec)) / (from_vec ^ to_vec) * from_vec * to_vec / abs(from_vec) / abs(to_vec) + (
        self ^ (from_vec ^ to_vec)
    ) / (from_vec ^ to_vec)


galgebra.mv.Mv.rotate = rotate


# %% [markdown]
# Define rotate in 3D
# -------------------
#
# Inline definitions


# %%
def rotate(self, from_vec, to_vec):
    return (self | (from_vec ^ to_vec)) / (from_vec ^ to_vec) * from_vec * to_vec / abs(from_vec) / abs(to_vec) + (
        self * (from_vec ^ to_vec) - (self | (from_vec ^ to_vec))
    ) / (from_vec ^ to_vec)


galgebra.mv.Mv.rotate = rotate


# %% [markdown]
# Define rotate in 3D
# -------------------
#
# Inline definitions


# %%
def rotate(self, from_vec, to_vec):
    return (self | (from_vec ^ to_vec)) / (from_vec ^ to_vec) * from_vec * to_vec / abs(from_vec) / abs(to_vec) + (
        self * (from_vec ^ to_vec) / (from_vec ^ to_vec) - (self | (from_vec ^ to_vec))
    ) / (from_vec ^ to_vec)


galgebra.mv.Mv.rotate = rotate


# %% [markdown]
# Define rotate in 3D
# -------------------
#
# Inline definitions


# %%
def rotate(self, from_vec, to_vec):
    return self + (self | (from_vec ^ to_vec)) / (from_vec ^ to_vec) * (
        from_vec * to_vec / abs(from_vec) / abs(to_vec) - 1
    )


galgebra.mv.Mv.rotate = rotate


# %% [markdown]
# Define rotate in 3D
# -------------------
#
# Inline definitions


# %%
def rotate(self, from_vec, to_vec):
    plane = from_vec ^ to_vec
    return self + self.project(onto=plane) * (
        from_vec * to_vec / abs(from_vec) / abs(to_vec) - self.project(onto=plane)
    )


galgebra.mv.Mv.rotate = rotate


# %% [markdown]
# Another Markdown cell
# %%
def delta(x, f):
    return f(x) - x


def rotate(self, from_vec, to_vec):
    plane = from_vec ^ to_vec
    return self + delta(
        x=self.project(onto=plane),
        f=lambda x: x * (from_vec * to_vec / abs(from_vec) / abs(to_vec)),
    )


galgebra.mv.Mv.rotate = rotate
# %% [markdown]
# Define the cross product
# ------------------------
# %%
cross_product = symvec2.rotate(from_vec=symvec1, to_vec=e_x).project(onto=e_y ^ e_z).rotate(
    from_vec=e_y, to_vec=e_z
).rotate(from_vec=e_x, to_vec=symvec1) * abs(symvec1)
# %%
cross_product
