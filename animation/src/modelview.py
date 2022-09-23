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
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import pyMatrixStack as ms

from dataclasses import dataclass

if not glfw.init():
    sys.exit()

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)

window = glfw.create_window(
    800, 800, "Derivation of the Cross Product", None, None
)
if not window:
    glfw.terminate()
    sys.exit()

# Make the window's context current
glfw.make_context_current(window)

# Install a key handler


def on_key(window, key, scancode, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, 1)


glfw.set_key_callback(window, on_key)

glClearColor(0.0, 0.0, 0.0, 1.0)

glEnable(GL_DEPTH_TEST)
glClearDepth(1.0)
glDepthFunc(GL_LEQUAL)


@dataclass
class Camera:
    r: float = 0.0
    rot_y: float = 0.0
    rot_x: float = 0.0


camera = Camera(r=20.0, rot_y=math.radians(45.0), rot_x=math.radians(35.264))


def handle_inputs():
    global camera

    move_multiple = 15.0
    if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
        camera.rot_y -= math.radians(1.0) % 360.0
    if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
        camera.rot_y += math.radians(1.0) % 360.0
    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        camera.rot_x -= math.radians(1.0) % 360.0
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        camera.rot_x += math.radians(1.0) % 360.0


square_vertices = np.array(
    [[-5.0, -5.0, 0.0], [5.0, -5.0, 0.0], [5.0, 5.0, 0.0], [-5.0, 5.0, 0.0]],
    dtype=np.float32,
)
virtual_camera_position = np.array([-40.0, 0.0, 80.0], dtype=np.float32)
virtual_camera_rot_y = math.radians(-30.0)
virtual_camera_rot_x = math.radians(15.0)


def draw_ground():
    # ascontiguousarray puts the array in column major order
    glLoadMatrixf(
        np.ascontiguousarray(ms.getCurrentMatrix(ms.MatrixStack.modelview).T)
    )
    glColor3f(0.1, 0.1, 0.1)
    glBegin(GL_LINES)
    for x in range(-10, 10, 1):
        for z in range(-10, 10, 1):
            glVertex3f(float(-x), float(0.0), float(z))
            glVertex3f(float(x), float(0.0), float(z))
            glVertex3f(float(x), float(0.0), float(-z))
            glVertex3f(float(x), float(0.0), float(z))

    glEnd()


def draw_y_axis():

    # ascontiguousarray puts the array in column major order
    glLoadMatrixf(
        np.ascontiguousarray(ms.getCurrentMatrix(ms.MatrixStack.modelview).T)
    )

    glLineWidth(3.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 1.0, 0.0)

    # arrow
    glVertex3f(0.0, 1.0, 0.0)
    glVertex3f(0.25, 0.75, 0.0)

    glVertex3f(0.0, 1.0, 0.0)
    glVertex3f(-0.25, 0.75, 0.0)
    glEnd()


def draw_axises():

    with ms.push_matrix(ms.MatrixStack.model):
        ms.rotate_x(ms.MatrixStack.model, math.radians(-90.0))

        # x axis
        with ms.push_matrix(ms.MatrixStack.model):
            ms.rotate_z(ms.MatrixStack.model, math.radians(-90.0))

            glColor3f(1.0, 0.0, 0.0)
            draw_y_axis()

        # z
        glColor3f(0.0, 0.0, 1.0)  # blue z
        with ms.push_matrix(ms.MatrixStack.model):
            ms.rotate_x(ms.MatrixStack.model, math.radians(90.0))
            # ms.rotate_z(ms.MatrixStack.model, math.radians(90.0))

            glColor3f(0.0, 0.0, 1.0)
            draw_y_axis()

        # y
        glColor3f(0.0, 1.0, 0.0)  # green y
        draw_y_axis()


TARGET_FRAMERATE = 60  # fps

# to try to standardize on 60 fps, compare times between frames
time_at_beginning_of_previous_frame = glfw.get_time()

animation_time = 0.0
animation_time_multiplier = 1.0
animation_paused = False

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
    # ms.ortho(left=-150.0,
    #          right=150.0,
    #          back=-150.0,
    #          top=150.0,
    #          near=1.0,
    #          far=10000.0)

    glMatrixMode(GL_PROJECTION)
    # ascontiguousarray puts the array in column major order
    glLoadMatrixf(
        np.ascontiguousarray(ms.getCurrentMatrix(ms.MatrixStack.projection).T)
    )

    # note - opengl matricies use degrees
    ms.translate(ms.MatrixStack.view, 0.0, 0.0, -camera.r)
    ms.rotate_x(ms.MatrixStack.view, camera.rot_x)
    ms.rotate_y(ms.MatrixStack.view, -camera.rot_y)

    glMatrixMode(GL_MODELVIEW)

    draw_ground()
    with ms.push_matrix(ms.MatrixStack.model):
        ms.rotate_x(ms.MatrixStack.model, math.radians(90.0))
        draw_ground()

    glClear(GL_DEPTH_BUFFER_BIT)

    draw_axises()

    # done with frame, flush and swap buffers
    # Swap front and back buffers
    glfw.swap_buffers(window)


glfw.terminate()
