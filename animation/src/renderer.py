# Copyright (c) 2022 William Emerison Six
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
    glUniform1f,
    glUniformMatrix4fv,
    glDrawArrays,
    GL_LINES,
    glDeleteVertexArrays,
    glDeleteBuffers,
    glDeleteProgram,
    glUniform3f,
)


import OpenGL.GL.shaders as shaders
import pyMatrixStack as ms

from dataclasses import dataclass

from contextlib import contextmanager

import ctypes


# NEW - for shader location
pwd = os.path.dirname(os.path.abspath(__file__))


# NEW - for shaders
glfloat_size = 4
floatsPerVertex = 3
floatsPerColor = 3


@contextmanager
def compile_shader(vert, frag):
    with open(os.path.join(pwd, vert), "r") as f:
        vs = shaders.compileShader(f.read(), GL_VERTEX_SHADER)

    with open(os.path.join(pwd, frag), "r") as f:
        fs = shaders.compileShader(f.read(), GL_FRAGMENT_SHADER)

    shader = shaders.compileProgram(vs, fs)

    try:
        yield shader
    finally:
        glDeleteProgram(shader)


def ground_vertices():
    verts = []
    for x in range(-10, 11, 1):
        for y in range(-10, 11, 1):
            verts.append(float(-x))
            verts.append(float(y))
            verts.append(float(0.0))
            verts.append(float(x))
            verts.append(float(y))
            verts.append(float(0.0))
            verts.append(float(x))
            verts.append(float(-y))
            verts.append(float(0.0))
            verts.append(float(x))
            verts.append(float(y))
            verts.append(float(0.0))
    return np.array(verts, dtype=np.float32)


def unit_circle_vertices():
    verts = []
    the_range = 100
    the_list = np.linspace(0.0, 2 * np.pi, the_range)

    for x in range(the_range - 1):
        verts.append(math.cos(the_list[x]))
        verts.append(math.sin(the_list[x]))
        verts.append(float(0.0))
        verts.append(math.cos(the_list[x + 1]))
        verts.append(math.sin(the_list[x + 1]))
        verts.append(float(0.0))
    return np.array(verts, dtype=np.float32)


def do_draw_lines(shader, vertices, time, xy=True, yz=False, zx=False):

    numberOfVertices = np.size(vertices) // floatsPerVertex

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    mMatrixLoc = glGetUniformLocation(shader, "mMatrix")
    vMatrixLoc = glGetUniformLocation(shader, "vMatrix")
    pMatrixLoc = glGetUniformLocation(shader, "pMatrix")

    # send the modelspace data to the GPU
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)

    position = glGetAttribLocation(shader, "position")
    glEnableVertexAttribArray(position)

    glVertexAttribPointer(
        position, floatsPerVertex, GL_FLOAT, False, 0, ctypes.c_void_p(0)
    )

    glBufferData(
        GL_ARRAY_BUFFER,
        glfloat_size * np.size(vertices),
        vertices,
        GL_STATIC_DRAW,
    )

    with ms.push_matrix(ms.MatrixStack.model):
        if xy:
            pass
        elif yz:
            ms.rotate_y(ms.MatrixStack.model, math.radians(90.0))
        elif zx:
            ms.rotate_x(ms.MatrixStack.model, math.radians(90.0))
        glUseProgram(shader)
        glBindVertexArray(vao)

        # pass projection parameters to the shader
        fov_loc = glGetUniformLocation(shader, "fov")
        glUniform1f(fov_loc, 45.0)
        aspect_loc = glGetUniformLocation(shader, "aspectRatio")
        glUniform1f(aspect_loc, 1.0)
        nearZ_loc = glGetUniformLocation(shader, "nearZ")
        glUniform1f(nearZ_loc, -5.0)
        farZ_loc = glGetUniformLocation(shader, "farZ")
        glUniform1f(farZ_loc, -150.00)

        # ascontiguousarray puts the array in column major order
        glUniformMatrix4fv(
            mMatrixLoc,
            1,
            GL_TRUE,
            np.ascontiguousarray(
                ms.getCurrentMatrix(ms.MatrixStack.model), dtype=np.float32
            ),
        )
        glUniformMatrix4fv(
            vMatrixLoc,
            1,
            GL_TRUE,
            np.ascontiguousarray(
                ms.getCurrentMatrix(ms.MatrixStack.view), dtype=np.float32
            ),
        )
        glUniformMatrix4fv(
            pMatrixLoc,
            1,
            GL_TRUE,
            np.ascontiguousarray(
                ms.getCurrentMatrix(ms.MatrixStack.projection),
                dtype=np.float32,
            ),
        )
        glDrawArrays(GL_LINES, 0, numberOfVertices)

    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])
    # reset VAO/VBO to default
    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)


@dataclass
class Vector:
    x: float
    y: float
    z: float
    r: float
    g: float
    b: float

    @property
    def angle_y(self):
        return -math.atan2(self.z, math.sqrt(self.x**2 + self.y**2))

    @property
    def angle_z(self):
        return math.atan2(self.y, self.x)


def do_draw_vector(shader, v, time):

    magnitude = math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def vertices_of_arrow():
        # glColor3f(0.1,0.1,0.1)
        verts = []
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

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    mMatrixLoc = glGetUniformLocation(shader, "mMatrix")
    vMatrixLoc = glGetUniformLocation(shader, "vMatrix")
    pMatrixLoc = glGetUniformLocation(shader, "pMatrix")
    colorLoc = glGetUniformLocation(shader, "color")

    # send the modelspace data to the GPU
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)

    position = glGetAttribLocation(shader, "position")
    glEnableVertexAttribArray(position)

    glVertexAttribPointer(
        position, floatsPerVertex, GL_FLOAT, False, 0, ctypes.c_void_p(0)
    )

    glBufferData(
        GL_ARRAY_BUFFER,
        glfloat_size * np.size(vertices),
        vertices,
        GL_STATIC_DRAW,
    )

    # send the modelspace data to the GPU
    # TODO, send color to the shader

    # reset VAO/VBO to default
    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # do rendering

    glUseProgram(shader)
    glBindVertexArray(vao)

    # pass projection parameters to the shader
    fov_loc = glGetUniformLocation(shader, "fov")
    glUniform1f(fov_loc, 45.0)
    aspect_loc = glGetUniformLocation(shader, "aspectRatio")
    glUniform1f(aspect_loc, 1.0)
    nearZ_loc = glGetUniformLocation(shader, "nearZ")
    glUniform1f(nearZ_loc, -5.0)
    farZ_loc = glGetUniformLocation(shader, "farZ")
    glUniform1f(farZ_loc, -150.00)
    # TODO, set the color

    with ms.push_matrix(ms.MatrixStack.model):
        ms.rotate_z(ms.MatrixStack.model, v.angle_z)
        ms.rotate_y(ms.MatrixStack.model, v.angle_y)

        # x axis
        with ms.push_matrix(ms.MatrixStack.model):
            ms.rotate_z(ms.MatrixStack.model, math.radians(-90.0))

            glUniform3f(colorLoc, v.r, v.g, v.b)

            # ascontiguousarray puts the array in column major order
            glUniformMatrix4fv(
                mMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.getCurrentMatrix(ms.MatrixStack.model),
                    dtype=np.float32,
                ),
            )
            glUniformMatrix4fv(
                vMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.getCurrentMatrix(ms.MatrixStack.view),
                    dtype=np.float32,
                ),
            )
            glUniformMatrix4fv(
                pMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.getCurrentMatrix(ms.MatrixStack.projection),
                    dtype=np.float32,
                ),
            )
            glDrawArrays(GL_LINES, 0, numberOfVertices)
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])


def do_draw_axis(shader):
    def vertices_of_axis():

        # glColor3f(0.1,0.1,0.1)
        verts = []
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

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    mMatrixLoc = glGetUniformLocation(shader, "mMatrix")
    vMatrixLoc = glGetUniformLocation(shader, "vMatrix")
    pMatrixLoc = glGetUniformLocation(shader, "pMatrix")
    colorLoc = glGetUniformLocation(shader, "color")

    # send the modelspace data to the GPU
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)

    position = glGetAttribLocation(shader, "position")
    glEnableVertexAttribArray(position)

    glVertexAttribPointer(
        position, floatsPerVertex, GL_FLOAT, False, 0, ctypes.c_void_p(0)
    )

    glBufferData(
        GL_ARRAY_BUFFER,
        glfloat_size * np.size(vertices),
        vertices,
        GL_STATIC_DRAW,
    )

    # send the modelspace data to the GPU
    # TODO, send color to the shader

    # reset VAO/VBO to default
    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # do rendering

    glDisable(GL_DEPTH_TEST)
    glUseProgram(shader)
    glBindVertexArray(vao)

    # pass projection parameters to the shader
    fov_loc = glGetUniformLocation(shader, "fov")
    glUniform1f(fov_loc, 45.0)
    aspect_loc = glGetUniformLocation(shader, "aspectRatio")
    glUniform1f(aspect_loc, 1.0)
    nearZ_loc = glGetUniformLocation(shader, "nearZ")
    glUniform1f(nearZ_loc, -5.0)
    farZ_loc = glGetUniformLocation(shader, "farZ")
    glUniform1f(farZ_loc, -150.00)
    # TODO, set the color

    with ms.push_matrix(ms.MatrixStack.model):

        # x axis
        with ms.push_matrix(ms.MatrixStack.model):
            ms.rotate_z(ms.MatrixStack.model, math.radians(-90.0))

            glUniform3f(colorLoc, 1.0, 0.0, 0.0)

            # ascontiguousarray puts the array in column major order
            glUniformMatrix4fv(
                mMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.getCurrentMatrix(ms.MatrixStack.model),
                    dtype=np.float32,
                ),
            )
            glUniformMatrix4fv(
                vMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.getCurrentMatrix(ms.MatrixStack.view),
                    dtype=np.float32,
                ),
            )
            glUniformMatrix4fv(
                pMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.getCurrentMatrix(ms.MatrixStack.projection),
                    dtype=np.float32,
                ),
            )
            glDrawArrays(GL_LINES, 0, numberOfVertices)

        # z
        # glColor3f(0.0,0.0,1.0) # blue z
        with ms.push_matrix(ms.MatrixStack.model):
            ms.rotate_y(ms.MatrixStack.model, math.radians(90.0))
            ms.rotate_z(ms.MatrixStack.model, math.radians(90.0))

            glUniform3f(colorLoc, 0.0, 0.0, 1.0)
            # ascontiguousarray puts the array in column major order
            glUniformMatrix4fv(
                mMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.getCurrentMatrix(ms.MatrixStack.model),
                    dtype=np.float32,
                ),
            )
            glUniformMatrix4fv(
                vMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.getCurrentMatrix(ms.MatrixStack.view),
                    dtype=np.float32,
                ),
            )
            glUniformMatrix4fv(
                pMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.getCurrentMatrix(ms.MatrixStack.projection),
                    dtype=np.float32,
                ),
            )
            glDrawArrays(GL_LINES, 0, numberOfVertices)

        # y
        glUniform3f(colorLoc, 0.0, 1.0, 0.0)
        # glColor3f(0.0,1.0,0.0) # green y
        # ascontiguousarray puts the array in column major order
        glUniformMatrix4fv(
            mMatrixLoc,
            1,
            GL_TRUE,
            np.ascontiguousarray(
                ms.getCurrentMatrix(ms.MatrixStack.model), dtype=np.float32
            ),
        )
        glUniformMatrix4fv(
            vMatrixLoc,
            1,
            GL_TRUE,
            np.ascontiguousarray(
                ms.getCurrentMatrix(ms.MatrixStack.view), dtype=np.float32
            ),
        )
        glUniformMatrix4fv(
            pMatrixLoc,
            1,
            GL_TRUE,
            np.ascontiguousarray(
                ms.getCurrentMatrix(ms.MatrixStack.projection),
                dtype=np.float32,
            ),
        )
        glDrawArrays(GL_LINES, 0, numberOfVertices)
        glBindVertexArray(0)
    glEnable(GL_DEPTH_TEST)

    # clean up

    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])


@dataclass
class Camera:
    r: float = 0.0
    rot_y: float = 0.0
    rot_x: float = 0.0
