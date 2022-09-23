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

vec1 = np.array([5.0, 2.0, 1.5], dtype=np.double)
vec2 = np.array([3.0, 4.0, 2.5], dtype=np.double)
vec2_globalspace = None

draw_rotate_z_ground = False
draw_rotate_vec1_to_natural_basis = False
draw_rotate_vec1_to_natural_time_start = 0.0

percent_comlete = 0.0

draw_second_rotate_ground = False
draw_second_rotate_to_natural_basis = False
draw_second_rotate_to_natural_basis_time_start = 0.0

percent_comlete2 = 0.0

draw_3 = False
draw_3_animate = False
draw_3_time_start = 0.0
draw_3_percent_complete = 0.0


draw_4 = False
draw_4_animate = False
draw_4_time_start = 0.0
draw_4_percent_complete = 0.0


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

    global draw_rotate_z_ground

    if glfw.get_key(window, glfw.KEY_1) == glfw.PRESS:
        draw_rotate_z_ground = not draw_rotate_z_ground

    global draw_rotate_vec1_to_natural_basis
    global draw_rotate_vec1_to_natural_time_start
    if glfw.get_key(window, glfw.KEY_2) == glfw.PRESS:
        draw_rotate_vec1_to_natural_basis = True
        draw_rotate_vec1_to_natural_time_start = animation_time

    global draw_second_rotate_ground
    global draw_second_rotate_to_natural_basis
    global draw_second_rotate_to_natural_basis_time_start

    if glfw.get_key(window, glfw.KEY_3) == glfw.PRESS:
        draw_second_rotate_ground = not draw_second_rotate_ground
    if glfw.get_key(window, glfw.KEY_4) == glfw.PRESS:
        draw_second_rotate_to_natural_basis = True
        draw_second_rotate_to_natural_basis_time_start = animation_time

    global draw_3
    global draw_3_animate
    global draw_3_time_start
    global draw_3_percent_complete

    if glfw.get_key(window, glfw.KEY_5) == glfw.PRESS:
        draw_3 = not draw_3
    if glfw.get_key(window, glfw.KEY_6) == glfw.PRESS:
        draw_3_animate = True
        draw_3_time_start = animation_time

    global draw_4
    global draw_4_animate
    global draw_4_time_start
    global draw_4_percent_complete

    if glfw.get_key(window, glfw.KEY_7) == glfw.PRESS:
        draw_4 = not draw_4
        draw_4_animate = True
        draw_4_time_start = animation_time


virtual_camera_position = np.array([-40.0, 0.0, 80.0], dtype=np.float32)
virtual_camera_rot_y = math.radians(-30.0)
virtual_camera_rot_x = math.radians(15.0)


def draw_ground(emphasize=False, unit_circle=False):
    # ascontiguousarray puts the array in column major order
    glLoadMatrixf(
        np.ascontiguousarray(ms.getCurrentMatrix(ms.MatrixStack.modelview).T)
    )

    if emphasize:
        glColor3f(0.3, 0.3, 0.3)
    else:
        glColor3f(0.2, 0.2, 0.2)

    glBegin(GL_LINES)
    for x in range(-10, 10, 1):
        for z in range(-10, 10, 1):
            glVertex3f(float(-x), float(0.0), float(z))
            glVertex3f(float(x), float(0.0), float(z))
            glVertex3f(float(x), float(0.0), float(-z))
            glVertex3f(float(x), float(0.0), float(z))

    glEnd()

    glDisable(GL_DEPTH_TEST)

    if unit_circle:
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_LINES)
        for x in np.linspace(0.0, 2 * np.pi, 500):
            glVertex3f(math.cos(x), float(0.0), math.sin(x))
        glEnd()
    glEnable(GL_DEPTH_TEST)


def draw_natural_basis():
    def draw_y_axis():

        # ascontiguousarray puts the array in column major order
        glLoadMatrixf(
            np.ascontiguousarray(
                ms.getCurrentMatrix(ms.MatrixStack.modelview).T
            )
        )

        glLineWidth(3.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        glEnd()

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


def draw_vector(v):
    def draw_y_axis():

        # ascontiguousarray puts the array in column major order
        glLoadMatrixf(
            np.ascontiguousarray(
                ms.getCurrentMatrix(ms.MatrixStack.modelview).T
            )
        )
        magnitude = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

        glLineWidth(3.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, magnitude, 0.0)

        glEnd()

        global vec2_globalspace
        vec2_globalspace = ms.getCurrentMatrix(ms.MatrixStack.model) @ np.array(
            [0.0, magnitude, 0.0, 1.0]
        )

        vec2_globalspace[1], vec2_globalspace[2] = (
            -vec2_globalspace[2],
            vec2_globalspace[1],
        )

    with ms.push_matrix(ms.MatrixStack.model):
        ms.rotate_y(ms.MatrixStack.model, math.atan2(v[1], v[0]))
        ms.rotate_z(
            ms.MatrixStack.model,
            math.atan2(v[2], math.sqrt(v[0] ** 2 + v[1] ** 2)),
        )

        # x axis
        with ms.push_matrix(ms.MatrixStack.model):
            ms.rotate_z(ms.MatrixStack.model, math.radians(-90.0))

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

    # draw using math coordinate system, not opengl

    draw_ground(unit_circle=True)
    with ms.push_matrix(ms.MatrixStack.model):
        ms.rotate_x(ms.MatrixStack.model, math.radians(90.0))
        draw_ground(unit_circle=True)

    glClear(GL_DEPTH_BUFFER_BIT)

    draw_natural_basis()

    if draw_3:
        with ms.push_matrix(ms.MatrixStack.model):

            if draw_4:
                draw_4_percent_complete = min(
                    1.0, (animation_time - draw_4_time_start) / 5.0
                )
                ms.rotate_x(
                    ms.MatrixStack.model,
                    draw_4_percent_complete * math.radians(90.0),
                )

            draw_3_percent_complete = 0.0
            if draw_3_animate:
                draw_3_percent_complete = min(
                    1.0, (animation_time - draw_3_time_start) / 5.0
                )

            glRotate(90.0 * draw_4_percent_complete, 1.0, 0.0, 0.0)
            glColor3f(1.0, 1.0, 0.0)
            glBegin(GL_LINES)
            glVertex3f(
                (1.0 - draw_3_percent_complete) * vec2_globalspace[0], 0.0, 0.0
            )
            glVertex3f(
                (1.0 - draw_3_percent_complete) * vec2_globalspace[0],
                vec2_globalspace[1],
                vec2_globalspace[2],
            )
            glEnd()

    emphasize2 = draw_rotate_z_ground
    if draw_second_rotate_to_natural_basis:
        percent_comlete2 = min(
            1.0,
            (animation_time - draw_second_rotate_to_natural_basis_time_start)
            / 5.0,
        )

        emphasize2 = emphasize2 and percent_comlete2 < 1.0
        ms.rotate_z(
            ms.MatrixStack.model,
            -percent_comlete2
            * math.atan2(vec1[2], math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)),
        )

    emphasize1 = draw_rotate_z_ground
    if draw_rotate_vec1_to_natural_basis:
        percent_comlete = min(
            1.0, (animation_time - draw_rotate_vec1_to_natural_time_start) / 5.0
        )

        emphasize1 = emphasize1 and percent_comlete < 1.0
        ms.rotate_y(
            ms.MatrixStack.model,
            -percent_comlete * math.atan2(vec1[1], vec1[0]),
        )

    if draw_rotate_z_ground or draw_second_rotate_ground:
        with ms.push_matrix(ms.MatrixStack.model):
            ms.rotate_y(ms.MatrixStack.model, math.atan2(vec1[1], vec1[0]))

            if draw_second_rotate_ground:
                ms.setToIdentityMatrix(ms.MatrixStack.model)

                ms.rotate_z(
                    ms.MatrixStack.model,
                    math.atan2(vec1[2], math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)),
                )

                if draw_second_rotate_to_natural_basis:
                    ms.rotate_z(
                        ms.MatrixStack.model,
                        -percent_comlete2
                        * math.atan2(
                            vec1[2], math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
                        ),
                    )

                ms.rotate_x(ms.MatrixStack.model, math.radians(90.0))

            draw_ground(emphasize1)
            glDisable(GL_DEPTH_TEST)
            # such crap code Bill
            if draw_second_rotate_ground:
                ms.rotate_x(ms.MatrixStack.model, math.radians(-90.0))

            draw_natural_basis()
            glEnable(GL_DEPTH_TEST)

    glClear(GL_DEPTH_BUFFER_BIT)
    glColor3f(0.0, 1.0, 1.0)
    draw_vector(vec1)
    glColor3f(1.0, 0.0, 1.0)
    draw_vector(vec2)

    # done with frame, flush and swap buffers
    # Swap front and back buffers
    glfw.swap_buffers(window)


glfw.terminate()
