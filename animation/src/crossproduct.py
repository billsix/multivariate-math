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
    glDisable,
    glClearDepth,
    glDepthFunc,
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
    GL_LESS,
    glDeleteVertexArrays,
    glDeleteBuffers,
    glDeleteProgram,
    glUniform3f,
)


import OpenGL.GL.shaders as shaders
import glfw
import pyMatrixStack as ms
import imgui
from imgui.integrations.glfw import GlfwRenderer

from dataclasses import dataclass

from contextlib import contextmanager

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


window = glfw.create_window(800, 800, "Cross Product Visualization", None, None)
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


def draw_lines(shader, vertices, time, xy=True, yz=False, zx=False):

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
            pass
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


vec1 = Vector(x=3.0, y=4.0, z=5.0, r=1.0, g=1.0, b=1.0)

vec2 = Vector(x=0.0, y=3.0, z=5.5, r=1.0, g=0.0, b=1.0)

vec3 = None


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


camera = Camera(r=22.0, rot_y=math.radians(45.0), rot_x=math.radians(35.264))


def handle_inputs():
    global camera

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
draw_third_relative_coordinates = False
do_third_rotate = False
project_onto_yz_plane = False
rotate_yz_90 = False
undo_rotate_z = False
undo_rotate_y = False
undo_rotate_x = False
do_scale = False
use_ortho = False

new_b = None
angle_x = None


with compile_shader(
    "ground.vert", "ground.frag"
) as lines_shader, compile_shader(
    "vector.vert", "vector.frag"
) as vector_shader:

    def draw_ground(time, xy=True, yz=False, zx=False):
        draw_lines(lines_shader, ground_vertices(), time, xy, yz, zx)

    def draw_unit_circle(time, xy=True, yz=False, zx=False):
        draw_lines(lines_shader, unit_circle_vertices(), time, xy, yz, zx)

    def draw_vector(v, time):
        do_draw_vector(vector_shader, v, time)

    def draw_axis():
        do_draw_axis(vector_shader)

    # Loop until the user closes the window
    while not glfw.window_should_close(window):

        # poll the time to try to get a constant framerate
        while (
            glfw.get_time()
            < time_at_beginning_of_previous_frame + 1.0 / TARGET_FRAMERATE
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

        if use_ortho:
            ms.ortho(
                left=-10.0 * float(width) / float(height),
                right=10.0 * float(width) / float(height),
                back=-10.00,
                top=10.00,
                near=10.0,
                far=-10.0,
            )

        else:
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

        draw_ground(animation_time)
        draw_ground(animation_time, xy=False, zx=True)
        draw_unit_circle(animation_time)
        draw_unit_circle(animation_time, xy=False, zx=True)

        draw_axis()

        imgui.new_frame()

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", "Cmd+Q", False, True
                )

                if clicked_quit:
                    exit(0)

                imgui.end_menu()
            imgui.end_main_menu_bar()

        imgui.set_next_window_bg_alpha(0.05)
        imgui.begin("Time", True)

        clicked_use_ortho, use_ortho = imgui.checkbox(
            "Orthogonal View", use_ortho
        )
        clicked_camera, camera.r = imgui.slider_float(
            "Camera Radius", camera.r, 3, 100.0
        )
        (
            clicked_animation_time_multiplier,
            animation_time_multiplier,
        ) = imgui.slider_float(
            "Sim Speed", animation_time_multiplier, 0.1, 10.0
        )
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
            label="Draw Relative Coordinates",
            state=draw_first_relative_coordinates,
        )
        imgui.same_line()
        changed, do_first_rotate = imgui.checkbox(
            label="Rotate Z", state=do_first_rotate
        )

        changed, draw_second_relative_coordinates = imgui.checkbox(
            label="Draw Second Relative Coordinates",
            state=draw_second_relative_coordinates,
        )
        imgui.same_line()
        changed, do_second_rotate = imgui.checkbox(
            label="Rotate Y", state=do_second_rotate
        )

        changed, draw_third_relative_coordinates = imgui.checkbox(
            label="Draw Third Relative Coordinates",
            state=draw_third_relative_coordinates,
        )
        imgui.same_line()
        changed, do_third_rotate = imgui.checkbox(
            label="Rotate X", state=do_third_rotate
        )

        if new_b is not None:
            angle_x = math.atan2(new_b[2], new_b[1])

        if draw_third_relative_coordinates:
            with ms.push_matrix(ms.MatrixStack.model):
                ms.rotate_x(ms.MatrixStack.model, angle_x)
                draw_ground(animation_time, xy=False, yz=True)
                draw_axis()

        if do_third_rotate:
            draw_third_relative_coordinates = False
            if not undo_rotate_x:
                ms.rotate_x(ms.MatrixStack.model, -angle_x)

        if draw_second_relative_coordinates:

            with ms.push_matrix(ms.MatrixStack.model):

                ms.rotate_y(ms.MatrixStack.model, vec1.angle_y)
                draw_ground(animation_time, xy=False, zx=True)
                draw_axis()

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
                draw_ground(animation_time)
                draw_axis()

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

        imgui.same_line()
        if imgui.button("Project onto e_2 e_3 plane"):
            project_onto_yz_plane = True

        if imgui.button("Rotate Y to Z, Z to -Y"):
            rotate_yz_90 = True

        if imgui.button("Undo Rotate X"):
            undo_rotate_x = True
        imgui.same_line()

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

        if do_second_rotate:
            if new_b is None:
                new_b = np.ascontiguousarray(
                    ms.getCurrentMatrix(ms.MatrixStack.model), dtype=np.float32
                ) @ np.array([vec2.x, vec2.y, vec2.z, 1.0], dtype=np.float32)

                # because we need to use math coordinate system
                y = -new_b[2]
                z = new_b[1]

                new_b[1] = y
                new_b[2] = z

        if vec3:
            with ms.push_matrix(ms.MatrixStack.model):
                ms.setToIdentityMatrix(ms.MatrixStack.model)
                ms.rotate_x(ms.MatrixStack.model, math.radians(-90.0))

                if undo_rotate_z:
                    ms.rotate_z(ms.MatrixStack.model, vec1.angle_z)
                if undo_rotate_y:
                    ms.rotate_y(ms.MatrixStack.model, vec1.angle_y)
                if undo_rotate_x:
                    ms.rotate_x(ms.MatrixStack.model, angle_x)
                    draw_ground(animation_time)

                if do_scale:
                    magnitude = math.sqrt(
                        vec1.x**2 + vec1.y**2 + vec1.z**2
                    )

                    ms.scale(
                        ms.MatrixStack.model, magnitude, magnitude, magnitude
                    )

                if project_onto_yz_plane:
                    if rotate_yz_90:
                        with ms.push_matrix(ms.MatrixStack.model):
                            ms.rotate_x(
                                ms.MatrixStack.model, math.radians(90.0)
                            )
                            draw_vector(vec3, animation_time)
                    else:
                        draw_vector(vec3, animation_time)
                else:
                    ms.translate(
                        ms.MatrixStack.model, vec3.translate_amount, 0.0, 0.0
                    )
                    draw_vector(vec3, animation_time)

        glDisable(GL_DEPTH_TEST)

        draw_vector(vec1, animation_time)
        draw_vector(vec2, animation_time)

        imgui.render()
        impl.render(imgui.get_draw_data())

        # done with frame, flush and swap buffers
        # Swap front and back buffers
        glfw.swap_buffers(window)


glfw.terminate()
