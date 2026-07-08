# Copyright (c) 2018-2026 William Emerison Six
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

"""Runtime TeX *billboard* labels for the cross-product demo.

Ported from modelviewprojection's ``mathdemos/_labels.py``.  The labels are
generated at runtime by ``texExpToPng`` (the C wrapper around ``latex`` +
``dvipng``), loaded as RGBA textures, and drawn as camera-facing quads (see
``billboard.vert``).

**Graceful degradation (the whole point of this module).**  ``texExpToPng`` is
on ``PATH`` in the podman image (the Dockerfile builds it from a SHA-pinned
clone).  On a host without it, ``shutil.which`` returns ``None``,
``LabelRenderer.available`` is ``False``, and every method is a no-op: the
demo runs exactly as before, just without label textures.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile

import numpy as np
import OpenGL.GL as GL
from OpenGL.GL import shaders as gl_shaders
from PIL import Image

# Resolved once: the texExpToPng executable, or None if it is not installed.
TEXEXP = shutil.which("texExpToPng")
AVAILABLE = TEXEXP is not None

_FLOATS_PER_VERT = 7  # center(3) + offset(2) + uv(2)
_VERTS = 6  # two triangles


class LabelRenderer:
    """Generates + draws TeX billboard labels, or no-ops if texExpToPng is absent.

    Usage per frame::

        if labels.available:
            labels.begin(view_M, proj_M, (width, height))
            labels.draw(r"\\vec{a}", center_world_xyz)
            ...
            labels.end()
    """

    def __init__(self, shader_dir: str, *, dpi: int = 600, fg: str = "rgb 1 1 1"):
        self.available = AVAILABLE
        self._dpi = dpi
        self._fg = fg
        self._tex_cache: dict = {}  # latex -> (tex_id, w, h) | None (gen failed)
        self._cachedir = None
        if not self.available:
            return
        self._cachedir = tempfile.mkdtemp(prefix="mvm_xprod_labels_")
        self._program = self._compile(shader_dir)
        self._u_v = GL.glGetUniformLocation(self._program, "vMatrix")
        self._u_p = GL.glGetUniformLocation(self._program, "pMatrix")
        self._u_tex = GL.glGetUniformLocation(self._program, "tex")
        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            _VERTS * _FLOATS_PER_VERT * 4,
            None,
            GL.GL_DYNAMIC_DRAW,
        )
        stride = _FLOATS_PER_VERT * 4
        for loc, size, off in ((0, 3, 0), (1, 2, 3), (2, 2, 5)):
            GL.glEnableVertexAttribArray(loc)
            GL.glVertexAttribPointer(
                loc,
                size,
                GL.GL_FLOAT,
                GL.GL_FALSE,
                stride,
                ctypes.c_void_p(off * 4),
            )
        GL.glBindVertexArray(0)

    # -- shader -----------------------------------------------------------
    def _compile(self, shader_dir: str):
        def read(name: str) -> str:
            with open(os.path.join(shader_dir, name)) as f:
                return f.read()

        vs = gl_shaders.compileShader(read("billboard.vert"), GL.GL_VERTEX_SHADER)
        fs = gl_shaders.compileShader(read("billboard.frag"), GL.GL_FRAGMENT_SHADER)
        return gl_shaders.compileProgram(vs, fs)

    # -- texture generation / caching ------------------------------------
    def _texture_for(self, latex: str):
        if latex in self._tex_cache:
            return self._tex_cache[latex]
        result = None
        png = self._generate(latex)
        if png is not None:
            result = self._load_texture(png)
        self._tex_cache[latex] = result
        return result

    def _generate(self, latex: str):
        if TEXEXP is None or self._cachedir is None:
            return None
        key = hashlib.sha1(f"{latex}|{self._dpi}|{self._fg}".encode()).hexdigest()[:16]
        out = os.path.join(self._cachedir, key + ".png")
        if os.path.exists(out):
            return out
        try:
            subprocess.run(
                [
                    TEXEXP,
                    "--exp",
                    f"${latex}$",
                    "--size",
                    str(self._dpi),
                    "--fg",
                    self._fg,
                    "--bg",
                    "Transparent",
                    "--output",
                    out,
                ],
                cwd=self._cachedir,  # texExpToPng leaves formula.tex/.dvi in CWD
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return None
        return out if os.path.exists(out) else None

    def _load_texture(self, png: str):
        # Flip vertically: PIL is top-left origin, GL texture (0,0) is bottom-left.
        # (Image.Transpose enum: the module-level FLIP_TOP_BOTTOM constant was
        # removed in Pillow 10.)
        img = Image.open(png).convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        w, h = img.size
        data = img.tobytes()
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA8,
            w,
            h,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            data,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return (tex, w, h)

    # -- per-frame drawing -----------------------------------------------
    def begin(self, view_M, proj_M, viewport) -> None:
        if not self.available:
            return
        self._view = np.ascontiguousarray(view_M, dtype=np.float32)
        self._proj = np.ascontiguousarray(proj_M, dtype=np.float32)
        # CPU copy (math convention, M @ column) for the behind-camera test.
        self._vp = np.asarray(proj_M, dtype=float) @ np.asarray(view_M, dtype=float)
        self._vw, self._vh = float(viewport[0]), float(viewport[1])
        GL.glUseProgram(self._program)
        GL.glUniformMatrix4fv(self._u_v, 1, GL.GL_TRUE, self._view)
        GL.glUniformMatrix4fv(self._u_p, 1, GL.GL_TRUE, self._proj)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glUniform1i(self._u_tex, 0)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDisable(GL.GL_DEPTH_TEST)  # labels read on top, always legible
        # Save the caller's bindings: renderer.py's compile_shader binds ONE
        # VAO for the program's lifetime and every draw call relies on it
        # staying bound, so end() must restore it (binding 0 instead leaves
        # the next frame's first glVertexAttribPointer with no VAO ->
        # GL_INVALID_OPERATION).
        self._prev_vao = int(GL.glGetIntegerv(GL.GL_VERTEX_ARRAY_BINDING))
        self._prev_array_buffer = int(GL.glGetIntegerv(GL.GL_ARRAY_BUFFER_BINDING))
        GL.glBindVertexArray(self._vao)

    def draw(self, latex: str, center_world, *, height_px: float = 44.0) -> None:
        """Draw ``latex`` centered at world point ``center_world`` (xyz), sized to
        ``height_px`` screen pixels tall (width follows the glyph aspect)."""
        if not self.available:
            return
        tex = self._texture_for(latex)
        if tex is None:
            return
        tex_id, iw, ih = tex
        c = np.asarray(center_world, dtype=float).reshape(-1)[:3]
        clip = self._vp @ np.array([c[0], c[1], c[2], 1.0])
        if clip[3] <= 1e-6:
            return  # behind the camera
        hh = height_px / self._vh  # half-height in NDC
        hw = (height_px * (iw / ih)) / self._vw  # half-width (pixel aspect)
        corners = (
            (-hw, -hh, 0.0, 0.0),
            (hw, -hh, 1.0, 0.0),
            (hw, hh, 1.0, 1.0),
            (-hw, -hh, 0.0, 0.0),
            (hw, hh, 1.0, 1.0),
            (-hw, hh, 0.0, 1.0),
        )
        buf = np.empty((_VERTS, _FLOATS_PER_VERT), dtype=np.float32)
        for i, (ox, oy, u, v) in enumerate(corners):
            buf[i] = (c[0], c[1], c[2], ox, oy, u, v)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, buf.nbytes, buf)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, _VERTS)

    def end(self) -> None:
        if not self.available:
            return
        # Restore the caller's bindings saved in begin() (see comment there).
        GL.glBindVertexArray(self._prev_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._prev_array_buffer)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_DEPTH_TEST)

    # -- teardown ---------------------------------------------------------
    def cleanup(self) -> None:
        if not self.available:
            return
        try:
            for entry in self._tex_cache.values():
                if entry is not None:
                    GL.glDeleteTextures([entry[0]])
            GL.glDeleteBuffers(1, [self._vbo])
            GL.glDeleteVertexArrays(1, [self._vao])
            GL.glDeleteProgram(self._program)
        except Exception:
            pass  # context may already be gone at shutdown
        if self._cachedir:
            shutil.rmtree(self._cachedir, ignore_errors=True)
