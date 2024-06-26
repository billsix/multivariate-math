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


import os
import numpy as np
import math
from OpenGL.GL import (
    glEnable,
    glDisable,
    GL_DEPTH_TEST,
    GL_TRUE,
    glGenVertexArrays,
    glBindVertexArray,
    GL_VERTEX_SHADER,
    GL_FRAGMENT_SHADER,
    GL_GEOMETRY_SHADER,
    glGenBuffers,
    glBindBuffer,
    GL_ARRAY_BUFFER,
    glGetAttribLocation,
    glEnableVertexAttribArray,
    glVertexAttribPointer,
    GL_FLOAT,
    glBufferData,
    GL_STATIC_DRAW,
    glUseProgram,
    glGetUniformLocation,
    glUniformMatrix4fv,
    glDrawArrays,
    GL_LINES,
    glDeleteBuffers,
    glDeleteProgram,
    glUniform1f,
    glUniform2f,
    glUniform3f,
    glDeleteVertexArrays,
)


import OpenGL.GL.shaders as shaders
import pyMatrixStack as ms

from dataclasses import dataclass

from contextlib import contextmanager

import ctypes
from OpenGL.GL.shaders import ShaderProgram
from numpy import ndarray
from typing import Iterator


# NEW - for shader location
pwd = os.path.dirname(os.path.abspath(__file__))


# NEW - for shaders
glfloat_size = 4
floatsPerVertex = 3
floatsPerColor = 3


line_thickness = 2.0


@contextmanager
def compile_shader(vert: str, frag: str, geom: str) -> Iterator[ShaderProgram]:
    with open(os.path.join(pwd, vert), "r") as f:
        vs = shaders.compileShader(f.read(), GL_VERTEX_SHADER)

    with open(os.path.join(pwd, frag), "r") as f:
        fs = shaders.compileShader(f.read(), GL_FRAGMENT_SHADER)

    with open(os.path.join(pwd, geom), "r") as f:
        gs = shaders.compileShader(f.read(), GL_GEOMETRY_SHADER)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    shader = shaders.compileProgram(vs, gs, fs)
    try:
        yield shader
    finally:
        glDeleteProgram(shader)
        glDeleteVertexArrays(1, [vao])


def do_draw_lines(
    shader: ShaderProgram,
    vertices: ndarray,
    time: float,
    width: int,
    height: int,
    xy: bool = True,
    yz: bool = False,
    zx: bool = False,
) -> None:
    numberOfVertices = np.size(vertices) // floatsPerVertex

    glUseProgram(shader)

    mvpMatrixLoc = glGetUniformLocation(shader, "mvpMatrix")
    colorLoc = glGetUniformLocation(shader, "color")
    thicknessLoc = glGetUniformLocation(shader, "u_thickness")
    viewportLoc = glGetUniformLocation(shader, "u_viewport_size")

    # send the modelspace data to the GPU
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)

    position = glGetAttribLocation(shader, "position")
    glEnableVertexAttribArray(position)

    glVertexAttribPointer(position, floatsPerVertex, GL_FLOAT, False, 0, ctypes.c_void_p(0))

    glBufferData(
        GL_ARRAY_BUFFER,
        glfloat_size * np.size(vertices),
        vertices,
        GL_STATIC_DRAW,
    )

    # xy
    if xy:
        with ms.push_matrix(ms.MatrixStack.model):
            # ascontiguousarray puts the array in column major order
            glUniformMatrix4fv(
                mvpMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.get_current_matrix(ms.MatrixStack.modelviewprojection),
                    dtype=np.float32,
                ),
            )

            glUniform3f(colorLoc, 0.3, 0.3, 0.3)
            glUniform1f(thicknessLoc, line_thickness)
            glUniform2f(viewportLoc, width, height)

            glDrawArrays(GL_LINES, 0, numberOfVertices)

    # yz
    if yz:
        with ms.push_matrix(ms.MatrixStack.model):
            ms.rotate_y(ms.MatrixStack.model, math.radians(90.0))

            # ascontiguousarray puts the array in column major order
            glUniformMatrix4fv(
                mvpMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.get_current_matrix(ms.MatrixStack.modelviewprojection),
                    dtype=np.float32,
                ),
            )

            glUniform3f(colorLoc, 0.3, 0.3, 0.3)
            glUniform1f(thicknessLoc, line_thickness)
            glUniform2f(viewportLoc, width, height)

            glDrawArrays(GL_LINES, 0, numberOfVertices)
    # zx
    if zx:
        with ms.push_matrix(ms.MatrixStack.model):
            ms.rotate_x(ms.MatrixStack.model, math.radians(90.0))

            # ascontiguousarray puts the array in column major order
            glUniformMatrix4fv(
                mvpMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.get_current_matrix(ms.MatrixStack.modelviewprojection),
                    dtype=np.float32,
                ),
            )

            glUniform3f(colorLoc, 0.3, 0.3, 0.3)
            glUniform1f(thicknessLoc, line_thickness)
            glUniform2f(viewportLoc, width, height)

            glDrawArrays(GL_LINES, 0, numberOfVertices)

    glDeleteBuffers(1, [vbo])
    # reset VAO/VBO to default
    glBindBuffer(GL_ARRAY_BUFFER, 0)


@dataclass
class Vector:
    x: float
    y: float
    z: float
    r: float
    g: float
    b: float
    highlight: bool = False

    @property
    def angle_y(self) -> float:
        return -math.atan2(self.z, math.sqrt(self.x**2 + self.y**2))

    @property
    def angle_z(self) -> float:
        return math.atan2(self.y, self.x)


def do_draw_vector(shader: ShaderProgram, v: Vector, width: int, height: int) -> None:
    glUseProgram(shader)

    magnitude = math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def vertices_of_arrow():
        # glColor3f(0.1,0.1,0.1)
        verts = []

        for y in range(-10, 11, 1):
            verts.append(float(0.0))
            verts.append(float(0.0))
            verts.append(float(0.0))

            verts.append(float(0.0))
            verts.append(float(magnitude))
            verts.append(float(0.0))

            # arrow
            verts.append(float(0.0))
            verts.append(float(magnitude))
            verts.append(float(0.0))

            verts.append(float(0.25))
            verts.append(float(magnitude - 0.25))
            verts.append(float(0.0))

            verts.append(float(0.0))
            verts.append(float(magnitude))
            verts.append(float(0.0))

            verts.append(float(-0.25))
            verts.append(float(magnitude - 0.25))
            verts.append(float(0.0))

        return np.array(verts, dtype=np.float32)

    # GL_QUADS aren't available anymore, only triangles
    # need 6 vertices instead of 4
    vertices = vertices_of_arrow()
    numberOfVertices = np.size(vertices) // floatsPerVertex

    mvpMatrixLoc = glGetUniformLocation(shader, "mvpMatrix")
    colorLoc = glGetUniformLocation(shader, "color")
    thicknessLoc = glGetUniformLocation(shader, "u_thickness")
    viewportLoc = glGetUniformLocation(shader, "u_viewport_size")

    # send the modelspace data to the GPU
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)

    position = glGetAttribLocation(shader, "position")
    glEnableVertexAttribArray(position)

    glVertexAttribPointer(position, floatsPerVertex, GL_FLOAT, False, 0, ctypes.c_void_p(0))

    glBufferData(
        GL_ARRAY_BUFFER,
        glfloat_size * np.size(vertices),
        vertices,
        GL_STATIC_DRAW,
    )

    # send the modelspace data to the GPU
    # TODO, send color to the shader

    # do rendering

    with ms.push_matrix(ms.MatrixStack.model):
        ms.rotate_z(ms.MatrixStack.model, v.angle_z)
        ms.rotate_y(ms.MatrixStack.model, v.angle_y)

        # x axis
        with ms.push_matrix(ms.MatrixStack.model):
            ms.rotate_z(ms.MatrixStack.model, math.radians(-90.0))

            if v.highlight:
                glUniform3f(colorLoc, 1.0, 1.0, 1.0)
            else:
                glUniform3f(colorLoc, v.r, v.g, v.b)
            # ascontiguousarray puts the array in column major order
            glUniformMatrix4fv(
                mvpMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.get_current_matrix(ms.MatrixStack.modelviewprojection),
                    dtype=np.float32,
                ),
            )

            glUniform1f(thicknessLoc, line_thickness)
            glUniform2f(viewportLoc, width, height)

            glDrawArrays(GL_LINES, 0, numberOfVertices)
    glDeleteBuffers(1, [vbo])


def do_draw_axis(
    shader: ShaderProgram,
    width: int,
    height: int,
    highlight_x: bool = False,
    highlight_y: bool = False,
    highlight_z: bool = False,
) -> None:
    glUseProgram(shader)

    def vertices_of_axis():
        # glColor3f(0.1,0.1,0.1)
        verts = []

        for y in range(-10, 11, 1):
            verts.append(float(0.0))
            verts.append(float(0.0))
            verts.append(float(0.0))

            verts.append(float(0.0))
            verts.append(float(1.0))
            verts.append(float(0.0))

            # arrow
            verts.append(float(0.0))
            verts.append(float(1.0))
            verts.append(float(0.0))

            verts.append(float(0.25))
            verts.append(float(0.75))
            verts.append(float(0.0))

            verts.append(float(0.0))
            verts.append(float(1.0))
            verts.append(float(0.0))

            verts.append(float(-0.25))
            verts.append(float(0.75))
            verts.append(float(0.0))

            return np.array(verts, dtype=np.float32)

    # GL_QUADS aren't available anymore, only triangles
    # need 6 vertices instead of 4
    vertices = vertices_of_axis()
    numberOfVertices = np.size(vertices) // floatsPerVertex

    mvpMatrixLoc = glGetUniformLocation(shader, "mvpMatrix")
    colorLoc = glGetUniformLocation(shader, "color")
    thicknessLoc = glGetUniformLocation(shader, "u_thickness")
    viewportLoc = glGetUniformLocation(shader, "u_viewport_size")

    # send the modelspace data to the GPU
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)

    position = glGetAttribLocation(shader, "position")
    glEnableVertexAttribArray(position)

    glVertexAttribPointer(position, floatsPerVertex, GL_FLOAT, False, 0, ctypes.c_void_p(0))

    glBufferData(
        GL_ARRAY_BUFFER,
        glfloat_size * np.size(vertices),
        vertices,
        GL_STATIC_DRAW,
    )

    # send the modelspace data to the GPU
    # TODO, send color to the shader

    # do rendering

    glDisable(GL_DEPTH_TEST)

    with ms.push_matrix(ms.MatrixStack.model):
        # x axis
        with ms.push_matrix(ms.MatrixStack.model):
            ms.rotate_z(ms.MatrixStack.model, math.radians(-90.0))

            if highlight_x:
                glUniform3f(colorLoc, 1.0, 1.0, 1.0)
            else:
                glUniform3f(colorLoc, 1.0, 0.0, 0.0)

            # ascontiguousarray puts the array in column major order
            glUniformMatrix4fv(
                mvpMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.get_current_matrix(ms.MatrixStack.modelviewprojection),
                    dtype=np.float32,
                ),
            )
            glUniform1f(thicknessLoc, line_thickness)
            glUniform2f(viewportLoc, width, height)
            glDrawArrays(GL_LINES, 0, numberOfVertices)

        # z
        # glColor3f(0.0,0.0,1.0) # blue z
        with ms.push_matrix(ms.MatrixStack.model):
            ms.rotate_y(ms.MatrixStack.model, math.radians(90.0))
            ms.rotate_z(ms.MatrixStack.model, math.radians(90.0))

            if highlight_z:
                glUniform3f(colorLoc, 1.0, 1.0, 1.0)
            else:
                glUniform3f(colorLoc, 0.0, 0.0, 1.0)
            # ascontiguousarray puts the array in column major order
            glUniformMatrix4fv(
                mvpMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.get_current_matrix(ms.MatrixStack.modelviewprojection),
                    dtype=np.float32,
                ),
            )
            glUniform1f(thicknessLoc, line_thickness)
            glUniform2f(viewportLoc, width, height)
            glDrawArrays(GL_LINES, 0, numberOfVertices)

        # y
        if highlight_y:
            glUniform3f(colorLoc, 1.0, 1.0, 1.0)
        else:
            glUniform3f(colorLoc, 0.0, 1.0, 0.0)
        # glColor3f(0.0,1.0,0.0) # green y
        # ascontiguousarray puts the array in column major order
        glUniformMatrix4fv(
            mvpMatrixLoc,
            1,
            GL_TRUE,
            np.ascontiguousarray(
                ms.get_current_matrix(ms.MatrixStack.modelviewprojection),
                dtype=np.float32,
            ),
        )
        glUniform1f(thicknessLoc, line_thickness)
        glUniform2f(viewportLoc, width, height)
        glDrawArrays(GL_LINES, 0, numberOfVertices)
    glEnable(GL_DEPTH_TEST)

    # clean up

    glDeleteBuffers(1, [vbo])


@dataclass
class Camera:
    r: float = 0.0
    rot_y: float = 0.0
    rot_x: float = 0.0
