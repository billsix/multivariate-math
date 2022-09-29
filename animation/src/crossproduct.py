# Copyright (c) 2018-2022 William Emerison Six
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

import sys
import os
import numpy as np
import math
from OpenGL.GL import (
    glClear,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    glViewport,
    glClearColor,
    glEnable,
    GL_SCISSOR_TEST,
    glScissor,
    glDisable,
    glClearDepth,
    glDepthFunc,
    GL_GREATER,
    GL_DEPTH_TEST,
    GL_LEQUAL,
    GL_TRUE,
    GL_BLEND,
    glBlendFunc,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
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
    GL_TRIANGLES,
    GL_LESS,
    glDeleteVertexArrays,
    glDeleteBuffers,
    glDeleteProgram,
    glUniform3f,
)


import OpenGL.GL.shaders as shaders
import glfw
import pyMatrixStack as ms
import atexit
import colorsys
import imgui
from imgui.integrations.glfw import GlfwRenderer
import staticlocal

from dataclasses import dataclass


import ctypes


# NEW - for shader location
pwd = os.path.dirname(os.path.abspath(__file__))

# NEW - for shaders
glfloat_size = 4
floatsPerVertex = 3
floatsPerColor = 3

if not glfw.init():
    sys.exit()

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
# for osx
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)


window = glfw.create_window(
    800, 800, "Cross Product Visualization", None, None
)
if not window:
    glfw.terminate()
    sys.exit()


# Make the window's context current
glfw.make_context_current(window)
imgui.create_context()
impl = GlfwRenderer(window)

# Install a key handler


def on_key(window, key, scancode, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, 1)


glfw.set_key_callback(window, on_key)

glClearColor(0.0, 0.0, 0.0, 1.0)

# NEW - TODO - talk about opengl matricies and z pos/neg
glClearDepth(1.0)
glDepthFunc(GL_LESS)
glEnable(GL_DEPTH_TEST)


class Ground:
    def __init__(self):
        pass

    def vertices(self):

        # glColor3f(0.1,0.1,0.1)
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

    def prepare_to_render(self):
        # GL_QUADS aren't available anymore, only triangles
        # need 6 vertices instead of 4
        vertices = self.vertices()
        self.numberOfVertices = np.size(vertices) // floatsPerVertex

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # initialize shaders

        with open(os.path.join(pwd, "ground.vert"), "r") as f:
            vs = shaders.compileShader(f.read(), GL_VERTEX_SHADER)

        with open(os.path.join(pwd, "ground.frag"), "r") as f:
            fs = shaders.compileShader(f.read(), GL_FRAGMENT_SHADER)

        self.shader = shaders.compileProgram(vs, fs)

        self.mMatrixLoc = glGetUniformLocation(self.shader, "mMatrix")
        self.vMatrixLoc = glGetUniformLocation(self.shader, "vMatrix")
        self.pMatrixLoc = glGetUniformLocation(self.shader, "pMatrix")

        # send the modelspace data to the GPU
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        position = glGetAttribLocation(self.shader, "position")
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

    # destructor
    def __del__(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteProgram(self.shader)

    def render(self, time, vertical=False):

        rotation_amount = 90.0 if vertical else 0.0
        with ms.push_matrix(ms.MatrixStack.model):
            ms.rotate_x(ms.MatrixStack.model, math.radians(rotation_amount))
            glUseProgram(self.shader)
            glBindVertexArray(self.vao)

            # pass projection parameters to the shader
            fov_loc = glGetUniformLocation(self.shader, "fov")
            glUniform1f(fov_loc, 45.0)
            aspect_loc = glGetUniformLocation(self.shader, "aspectRatio")
            glUniform1f(aspect_loc, 1.0)
            nearZ_loc = glGetUniformLocation(self.shader, "nearZ")
            glUniform1f(nearZ_loc, -5.0)
            farZ_loc = glGetUniformLocation(self.shader, "farZ")
            glUniform1f(farZ_loc, -150.00)

            # ascontiguousarray puts the array in column major order
            glUniformMatrix4fv(
                self.mMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.getCurrentMatrix(ms.MatrixStack.model), dtype=np.float32
                ),
            )
            glUniformMatrix4fv(
                self.vMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.getCurrentMatrix(ms.MatrixStack.view), dtype=np.float32
                ),
            )
            glUniformMatrix4fv(
                self.pMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.getCurrentMatrix(ms.MatrixStack.projection),
                    dtype=np.float32,
                ),
            )
            glDrawArrays(GL_LINES, 0, self.numberOfVertices)
            glBindVertexArray(0)


ground = Ground()
ground.prepare_to_render()


# terrible hack of a class, only because I somehow can't seem to figure
# out how to use two vbos with a vao
class UnitCircle(Ground):
    def vertices(self):
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


unit_circle = UnitCircle()
unit_circle.prepare_to_render()


class Vector:
    def __init__(self, x, y, z, r, g, b):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.g = g
        self.b = b

        self.angle_z = math.atan2(self.y, self.x)
        self.angle_y = -math.atan2(self.z, math.sqrt(self.x**2 + self.y**2))

    def vertices(self):

        magnitude = math.sqrt(self.x**2 + self.y**2 + self.z**2)

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

    def prepare_to_render(self):
        # GL_QUADS aren't available anymore, only triangles
        # need 6 vertices instead of 4
        vertices = self.vertices()
        self.numberOfVertices = np.size(vertices) // floatsPerVertex

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # initialize shaders

        with open(os.path.join(pwd, "axis.vert"), "r") as f:
            vs = shaders.compileShader(f.read(), GL_VERTEX_SHADER)

        with open(os.path.join(pwd, "axis.frag"), "r") as f:
            fs = shaders.compileShader(f.read(), GL_FRAGMENT_SHADER)

        self.shader = shaders.compileProgram(vs, fs)

        self.mMatrixLoc = glGetUniformLocation(self.shader, "mMatrix")
        self.vMatrixLoc = glGetUniformLocation(self.shader, "vMatrix")
        self.pMatrixLoc = glGetUniformLocation(self.shader, "pMatrix")
        self.colorLoc = glGetUniformLocation(self.shader, "color")

        # send the modelspace data to the GPU
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        position = glGetAttribLocation(self.shader, "position")
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

    # destructor
    def __del__(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteProgram(self.shader)

    def render(self, time, grayed_out=False):
        glUseProgram(self.shader)
        glBindVertexArray(self.vao)

        # pass projection parameters to the shader
        fov_loc = glGetUniformLocation(self.shader, "fov")
        glUniform1f(fov_loc, 45.0)
        aspect_loc = glGetUniformLocation(self.shader, "aspectRatio")
        glUniform1f(aspect_loc, 1.0)
        nearZ_loc = glGetUniformLocation(self.shader, "nearZ")
        glUniform1f(nearZ_loc, -5.0)
        farZ_loc = glGetUniformLocation(self.shader, "farZ")
        glUniform1f(farZ_loc, -150.00)
        # TODO, set the color

        with ms.push_matrix(ms.MatrixStack.model):
            ms.rotate_z(ms.MatrixStack.model, self.angle_z)
            ms.rotate_y(ms.MatrixStack.model, self.angle_y)

            # x axis
            with ms.push_matrix(ms.MatrixStack.model):
                ms.rotate_z(ms.MatrixStack.model, math.radians(-90.0))

                glUniform3f(self.colorLoc, self.r, self.g, self.b)

                # ascontiguousarray puts the array in column major order
                glUniformMatrix4fv(
                    self.mMatrixLoc,
                    1,
                    GL_TRUE,
                    np.ascontiguousarray(
                        ms.getCurrentMatrix(ms.MatrixStack.model),
                        dtype=np.float32,
                    ),
                )
                glUniformMatrix4fv(
                    self.vMatrixLoc,
                    1,
                    GL_TRUE,
                    np.ascontiguousarray(
                        ms.getCurrentMatrix(ms.MatrixStack.view),
                        dtype=np.float32,
                    ),
                )
                glUniformMatrix4fv(
                    self.pMatrixLoc,
                    1,
                    GL_TRUE,
                    np.ascontiguousarray(
                        ms.getCurrentMatrix(ms.MatrixStack.projection),
                        dtype=np.float32,
                    ),
                )
                glDrawArrays(GL_LINES, 0, self.numberOfVertices)


# vec1 = Vector(x=0.0, y=0.0, z=5.0, r=1.0, g=1.0, b=1.0)
# vec1.prepare_to_render()

# vec2 = Vector(x=0.0, y=5.0, z=0.0, r=1.0, g=0.0, b=1.0)
# vec2.prepare_to_render()

# vec1 = Vector(x=3.0, y=4.0, z=5.0, r=1.0, g=1.0, b=1.0)
# vec1.prepare_to_render()

# vec2 = Vector(x=0.0, y=3.0, z=5.5, r=1.0, g=0.0, b=1.0)
# vec2.prepare_to_render()

vec1 = Vector(x=3.0, y=4.0, z=5.0, r=1.0, g=1.0, b=1.0)
vec1.prepare_to_render()

vec2 = Vector(x=0.0, y=3.0, z=5.5, r=1.0, g=0.0, b=1.0)
vec2.prepare_to_render()

vec3 = None


class Axis:
    def __init__(self):
        pass

    def vertices(self):

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

    def prepare_to_render(self):
        # GL_QUADS aren't available anymore, only triangles
        # need 6 vertices instead of 4
        vertices = self.vertices()
        self.numberOfVertices = np.size(vertices) // floatsPerVertex

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # initialize shaders

        with open(os.path.join(pwd, "axis.vert"), "r") as f:
            vs = shaders.compileShader(f.read(), GL_VERTEX_SHADER)

        with open(os.path.join(pwd, "axis.frag"), "r") as f:
            fs = shaders.compileShader(f.read(), GL_FRAGMENT_SHADER)

        self.shader = shaders.compileProgram(vs, fs)

        self.mMatrixLoc = glGetUniformLocation(self.shader, "mMatrix")
        self.vMatrixLoc = glGetUniformLocation(self.shader, "vMatrix")
        self.pMatrixLoc = glGetUniformLocation(self.shader, "pMatrix")
        self.colorLoc = glGetUniformLocation(self.shader, "color")

        # send the modelspace data to the GPU
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        position = glGetAttribLocation(self.shader, "position")
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

    # destructor
    def __del__(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteProgram(self.shader)

    def render(self, time, grayed_out=False):
        glDisable(GL_DEPTH_TEST)
        glUseProgram(self.shader)
        glBindVertexArray(self.vao)

        # pass projection parameters to the shader
        fov_loc = glGetUniformLocation(self.shader, "fov")
        glUniform1f(fov_loc, 45.0)
        aspect_loc = glGetUniformLocation(self.shader, "aspectRatio")
        glUniform1f(aspect_loc, 1.0)
        nearZ_loc = glGetUniformLocation(self.shader, "nearZ")
        glUniform1f(nearZ_loc, -5.0)
        farZ_loc = glGetUniformLocation(self.shader, "farZ")
        glUniform1f(farZ_loc, -150.00)
        # TODO, set the color

        with ms.push_matrix(ms.MatrixStack.model):

            # x axis
            with ms.push_matrix(ms.MatrixStack.model):
                ms.rotate_z(ms.MatrixStack.model, math.radians(-90.0))

                glUniform3f(self.colorLoc, 1.0, 0.0, 0.0)

                # ascontiguousarray puts the array in column major order
                glUniformMatrix4fv(
                    self.mMatrixLoc,
                    1,
                    GL_TRUE,
                    np.ascontiguousarray(
                        ms.getCurrentMatrix(ms.MatrixStack.model),
                        dtype=np.float32,
                    ),
                )
                glUniformMatrix4fv(
                    self.vMatrixLoc,
                    1,
                    GL_TRUE,
                    np.ascontiguousarray(
                        ms.getCurrentMatrix(ms.MatrixStack.view),
                        dtype=np.float32,
                    ),
                )
                glUniformMatrix4fv(
                    self.pMatrixLoc,
                    1,
                    GL_TRUE,
                    np.ascontiguousarray(
                        ms.getCurrentMatrix(ms.MatrixStack.projection),
                        dtype=np.float32,
                    ),
                )
                glDrawArrays(GL_LINES, 0, self.numberOfVertices)

            # z
            # glColor3f(0.0,0.0,1.0) # blue z
            with ms.push_matrix(ms.MatrixStack.model):
                ms.rotate_y(ms.MatrixStack.model, math.radians(90.0))
                ms.rotate_z(ms.MatrixStack.model, math.radians(90.0))

                glUniform3f(self.colorLoc, 0.0, 0.0, 1.0)
                # ascontiguousarray puts the array in column major order
                glUniformMatrix4fv(
                    self.mMatrixLoc,
                    1,
                    GL_TRUE,
                    np.ascontiguousarray(
                        ms.getCurrentMatrix(ms.MatrixStack.model),
                        dtype=np.float32,
                    ),
                )
                glUniformMatrix4fv(
                    self.vMatrixLoc,
                    1,
                    GL_TRUE,
                    np.ascontiguousarray(
                        ms.getCurrentMatrix(ms.MatrixStack.view),
                        dtype=np.float32,
                    ),
                )
                glUniformMatrix4fv(
                    self.pMatrixLoc,
                    1,
                    GL_TRUE,
                    np.ascontiguousarray(
                        ms.getCurrentMatrix(ms.MatrixStack.projection),
                        dtype=np.float32,
                    ),
                )
                glDrawArrays(GL_LINES, 0, self.numberOfVertices)

            # y
            glUniform3f(self.colorLoc, 0.0, 1.0, 0.0)
            # glColor3f(0.0,1.0,0.0) # green y
            # ascontiguousarray puts the array in column major order
            glUniformMatrix4fv(
                self.mMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.getCurrentMatrix(ms.MatrixStack.model), dtype=np.float32
                ),
            )
            glUniformMatrix4fv(
                self.vMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.getCurrentMatrix(ms.MatrixStack.view), dtype=np.float32
                ),
            )
            glUniformMatrix4fv(
                self.pMatrixLoc,
                1,
                GL_TRUE,
                np.ascontiguousarray(
                    ms.getCurrentMatrix(ms.MatrixStack.projection),
                    dtype=np.float32,
                ),
            )
            glDrawArrays(GL_LINES, 0, self.numberOfVertices)
            glBindVertexArray(0)
        glEnable(GL_DEPTH_TEST)


axis = Axis()
axis.prepare_to_render()


@dataclass
class Camera:
    r: float = 0.0
    rot_y: float = 0.0
    rot_x: float = 0.0


camera = Camera(r=22.0, rot_y=math.radians(45.0), rot_x=math.radians(35.264))


def handle_inputs():
    global camera

    move_multiple = 15.0
    if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
        camera.rot_y -= math.radians(1.0)
    if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
        camera.rot_y += math.radians(1.0)
    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        camera.rot_x -= math.radians(1.0)
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        camera.rot_x += math.radians(1.0)

    if camera.rot_x > math.pi / 2.0:
        camera.rot_x = math.pi / 2.0
    if camera.rot_x < -math.pi / 2.0:
        camera.rot_x = -math.pi / 2.0


TARGET_FRAMERATE = 60  # fps

# to try to standardize on 60 fps, compare times between frames
time_at_beginning_of_previous_frame = glfw.get_time()

animation_time = 0.0
animation_time_multiplier = 1.0
animation_paused = False


draw_first_relative_coordinates = False
do_first_rotate = False
draw_second_relative_coordinates = False
do_second_rotate = False
project_onto_yz_plane = False
rotate_yz_90 = False
undo_rotate_z = False
undo_rotate_y = False
do_scale = False

# Loop until the user closes the window
while not glfw.window_should_close(window):
    # poll the time to try to get a constant framerate
    while (
        glfw.get_time() < time_at_beginning_of_previous_frame + 1.0 / TARGET_FRAMERATE
    ):
        pass
    # set for comparison on the next frame
    time_at_beginning_of_previous_frame = glfw.get_time()

    if not animation_paused:
        animation_time += 1.0 / 60.0 * animation_time_multiplier

    # Poll for and process events
    glfw.poll_events()
    impl.process_inputs()

    width, height = glfw.get_framebuffer_size(window)
    glViewport(0, 0, width, height)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # render scene
    handle_inputs()

    ms.setToIdentityMatrix(ms.MatrixStack.model)
    ms.setToIdentityMatrix(ms.MatrixStack.view)
    ms.setToIdentityMatrix(ms.MatrixStack.projection)

    # set the projection matrix to be perspective
    ms.perspective(
        fov=45.0,
        aspectRatio=float(width) / float(height),
        nearZ=0.1,
        farZ=10000.0,
    )

    # note - opengl matricies use degrees
    ms.translate(ms.MatrixStack.view, 0.0, 0.0, -camera.r)
    ms.rotate_x(ms.MatrixStack.view, camera.rot_x)
    ms.rotate_y(ms.MatrixStack.view, -camera.rot_y)

    # do everything in math coordinate system
    ms.rotate_x(ms.MatrixStack.model, math.radians(-90.0))

    ground.render(animation_time)
    ground.render(animation_time, vertical=True)
    unit_circle.render(animation_time)
    unit_circle.render(animation_time, vertical=True)

    axis.render(animation_time)

    imgui.new_frame()

    if imgui.begin_main_menu_bar():
        if imgui.begin_menu("File", True):
            clicked_quit, selected_quit = imgui.menu_item("Quit", "Cmd+Q", False, True)

            if clicked_quit:
                exit(0)

            imgui.end_menu()
        imgui.end_main_menu_bar()

    imgui.set_next_window_bg_alpha(0.05)
    imgui.begin("Time", True)

    clicked_animation_paused, animation_paused = imgui.checkbox(
        "Pause", animation_paused
    )
    clicked_camera, camera.r = imgui.slider_float("Camera Radius", camera.r, 3, 100.0)
    (
        clicked_animation_time_multiplier,
        animation_time_multiplier,
    ) = imgui.slider_float("Sim Speed", animation_time_multiplier, 0.1, 10.0)
    if imgui.button("Restart"):
        animation_time = 0.0

    if imgui.button("View Down X Axis"):
        camera.rot_x = 0.0
        camera.rot_y = math.pi / 2.0
    imgui.same_line()
    if imgui.button("View Down Y Axis"):
        camera.rot_x = 0.0
        camera.rot_y = 0.0
    imgui.same_line()
    if imgui.button("View Down Z Axis"):
        camera.rot_x = math.pi / 2.0
        camera.rot_y = 0.0

    changed, draw_first_relative_coordinates = imgui.checkbox(
        label="Draw Relative Coordinates", state=draw_first_relative_coordinates
    )
    imgui.same_line()
    changed, do_first_rotate = imgui.checkbox(label="Rotate Z", state=do_first_rotate)

    changed, draw_second_relative_coordinates = imgui.checkbox(
        label="Draw Second Relative Coordinates",
        state=draw_second_relative_coordinates,
    )
    imgui.same_line()
    changed, do_second_rotate = imgui.checkbox(label="Rotate Y", state=do_second_rotate)

    if draw_second_relative_coordinates:

        with ms.push_matrix(ms.MatrixStack.model):

            ms.rotate_y(ms.MatrixStack.model, vec1.angle_y)
            ground.render(animation_time, vertical=True)
            axis.render(animation_time)

    if do_second_rotate:
        draw_second_relative_coordinates = False
        if not undo_rotate_y:
            ms.rotate_y(ms.MatrixStack.model, -vec1.angle_y)

    if do_first_rotate:
        draw_first_relative_coordinates = False
        if not undo_rotate_z:
            ms.rotate_z(ms.MatrixStack.model, -vec1.angle_z)

    if draw_first_relative_coordinates:
        with ms.push_matrix(ms.MatrixStack.model):
            ms.rotate_z(ms.MatrixStack.model, vec1.angle_z)
            #  ms.rotate_y(ms.MatrixStack.model, self.angle_y)
            ground.render(animation_time)
            axis.render(animation_time)

    if imgui.button("Show Triangle"):
        vec3_after_rotate = np.ascontiguousarray(
            ms.getCurrentMatrix(ms.MatrixStack.model), dtype=np.float32
        ) @ np.array([vec2.x, vec2.y, vec2.z, 1.0], dtype=np.float32)

        vec3 = Vector(
            x=0.0,
            y=-vec3_after_rotate[2],
            z=vec3_after_rotate[1],
            r=0.0,
            g=1.0,
            b=1.0,
        )
        vec3.translate_amount = vec3_after_rotate[0]
        vec3.prepare_to_render()
    imgui.same_line()
    if imgui.button("Project onto e_2 e_3 plane"):
        project_onto_yz_plane = True

    if imgui.button("Rotate Y to Z, Z to -Y"):
        rotate_yz_90 = True

    if imgui.button("Undo Rotate Y"):
        undo_rotate_y = True
    imgui.same_line()

    if imgui.button("Undo Rotate Z"):
        undo_rotate_z = True

    if imgui.button("Scale By Magnitude of first vector"):
        do_scale = True

    if imgui.tree_node(
        "From World Space, Against Arrows, Read Bottom Up",
        imgui.TREE_NODE_DEFAULT_OPEN,
    ):
        imgui.tree_pop()

    imgui.end()

    glDisable(GL_DEPTH_TEST)

    vec1.render(animation_time)
    vec2.render(animation_time)

    if vec3:
        with ms.push_matrix(ms.MatrixStack.model):
            ms.setToIdentityMatrix(ms.MatrixStack.model)
            ms.rotate_x(ms.MatrixStack.model, math.radians(-90.0))

            if do_scale:
                magnitude = math.sqrt(vec1.x**2 + vec1.y**2 + vec1.z**2)

                ms.scale(ms.MatrixStack.model, magnitude, magnitude, magnitude)

            if undo_rotate_z:
                ms.rotate_z(ms.MatrixStack.model, vec1.angle_z)
            if undo_rotate_y:
                ms.rotate_y(ms.MatrixStack.model, vec1.angle_y)

            if project_onto_yz_plane:
                if rotate_yz_90:
                    with ms.push_matrix(ms.MatrixStack.model):
                        ms.rotate_x(ms.MatrixStack.model, math.radians(90.0))
                        vec3.render(animation_time)
                else:
                    vec3.render(animation_time)
            else:
                ms.translate(ms.MatrixStack.model, vec3.translate_amount, 0.0, 0.0)
                vec3.render(animation_time)

    imgui.render()
    impl.render(imgui.get_draw_data())

    # done with frame, flush and swap buffers
    # Swap front and back buffers
    glfw.swap_buffers(window)


glfw.terminate()
