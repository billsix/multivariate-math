# %% [markdown]
# Cross Product
# -------------
# %%

import sympy
import galgebra
from galgebra.ga import Ga
from galgebra.printer import latex
from IPython.display import Math

# tell sympy to use our printing by default
sympy.init_printing(latex_printer=latex, use_latex="mathjax")

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

# %%
Math("vec1 = %s" % latex(vec1))
# %%
Math("vec2 = %s" % latex(vec2))

# %% [markdown]
# Define vectors with symbols
# ---------------------------
# %%
symvec1 = a_1 * e_x + a_2 * e_y + a_3 * e_z
symvec2 = b_1 * e_x + b_2 * e_y + b_3 * e_z

Math("symvec1 = %s" % latex(symvec1))

# %%
Math("symvec2 = %s" % latex(symvec2))


# %% [markdown]
# Define Basic Operations on Vectors/Multivectors
# -----------------------------------------------
# %%
def project(self, onto):
    return (self | onto) / onto


galgebra.mv.Mv.project = project


# %%
def reject(self, away_from):
    return (self ^ away_from) / away_from


galgebra.mv.Mv.reject = reject


# %%
def rotate(self, from_vec, to_vec):
    plane = from_vec ^ to_vec

    parallel = self.project(onto=plane)
    perp = self.reject(away_from=plane)

    rotated_parallel = parallel * from_vec * to_vec / abs(from_vec) / abs(to_vec)
    return rotated_parallel + perp


galgebra.mv.Mv.rotate = rotate

# %%
cross_product = symvec2.rotate(from_vec=symvec1, to_vec=e_x).project(onto=e_y ^ e_z).rotate(
    from_vec=e_y, to_vec=e_z
).rotate(from_vec=e_x, to_vec=symvec1) * abs(symvec1)
Math("cross\_product = %s" % latex(cross_product))
