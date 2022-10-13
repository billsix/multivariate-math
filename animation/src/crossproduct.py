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
    GL_LESS,
)


import glfw
import pyMatrixStack as ms
import imgui
from imgui.integrations.glfw import GlfwRenderer

from renderer import (
    do_draw_axis,
    do_draw_lines,
    do_draw_vector,
    Vector,
    Camera,
    compile_shader,
    ground_vertices,
    unit_circle_vertices,
)


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


vec1 = Vector(x=3.0, y=4.0, z=5.0, r=1.0, g=1.0, b=1.0)

vec2 = Vector(x=0.0, y=3.0, z=5.5, r=1.0, g=0.0, b=1.0)

vec3 = None


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

step_number = 0


with compile_shader("lines.vert", "lines.frag") as lines_shader:

    def draw_ground(time, xy=True, yz=False, zx=False):
        do_draw_lines(lines_shader, ground_vertices(), time, xy, yz, zx)

    def draw_unit_circle(time, xy=True, yz=False, zx=False):
        do_draw_lines(lines_shader, unit_circle_vertices(), time, xy, yz, zx)

    def draw_vector(v, time):
        do_draw_vector(lines_shader, v, time)

    def draw_axis():
        do_draw_axis(lines_shader)

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

        if not undo_rotate_x:
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
        imgui.begin("Camera", True)

        clicked_use_ortho, use_ortho = imgui.checkbox(
            "Orthogonal View", use_ortho
        )

        imgui.same_line()

        clicked_camera, camera.r = imgui.slider_float(
            "Camera Radius", camera.r, 3, 100.0
        )

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

        imgui.end()

        imgui.set_next_window_bg_alpha(0.05)
        imgui.begin("Time", True)

        (
            clicked_animation_time_multiplier,
            animation_time_multiplier,
        ) = imgui.slider_float(
            "Sim Speed", animation_time_multiplier, 0.1, 10.0
        )
        if imgui.button("Restart"):
            animation_time = 0.0
            step_number = 0

        if step_number == 0:
            changed, draw_first_relative_coordinates = imgui.checkbox(
                label="Draw Relative Coordinates",
                state=draw_first_relative_coordinates,
            )
            imgui.same_line()
            changed, do_first_rotate = imgui.checkbox(
                label="Rotate Z", state=do_first_rotate
            )
            if changed:
                step_number = 1

        if step_number == 1:
            changed, draw_second_relative_coordinates = imgui.checkbox(
                label="Draw Second Relative Coordinates",
                state=draw_second_relative_coordinates,
            )
            imgui.same_line()
            changed, do_second_rotate = imgui.checkbox(
                label="Rotate Y", state=do_second_rotate
            )
            if changed:
                step_number = 2

        if step_number == 2:
            changed, draw_third_relative_coordinates = imgui.checkbox(
                label="Draw Third Relative Coordinates",
                state=draw_third_relative_coordinates,
            )
            imgui.same_line()
            changed, do_third_rotate = imgui.checkbox(
                label="Rotate X", state=do_third_rotate
            )
            if changed:
                step_number = 3

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

        if step_number == 3:
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
                step_number = 4

        if step_number == 4:

            if imgui.button("Rotate Y to Z, Z to -Y"):
                rotate_yz_90 = True
                step_number = 5

        if step_number == 5:
            if imgui.button("Undo Rotate X"):
                undo_rotate_x = True
                step_number = 6

        if step_number == 6:
            if imgui.button("Undo Rotate Y"):
                undo_rotate_y = True
                step_number = 7

        if step_number == 7:
            if imgui.button("Undo Rotate Z"):
                undo_rotate_z = True
                step_number = 8

        if step_number == 8:
            if imgui.button("Scale By Magnitude of first vector"):
                do_scale = True

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
