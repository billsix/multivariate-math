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

import functools


from renderer import (
    do_draw_axis,
    do_draw_lines,
    do_draw_vector,
    Vector,
    Camera,
    compile_shader,
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


@functools.cache
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


@functools.cache
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


vec1 = Vector(x=3.0, y=4.0, z=5.0, r=1.0, g=1.0, b=1.0)

vec2 = Vector(x=0.3, y=3.0, z=5.5, r=1.0, g=0.0, b=1.0)

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


animation_time = None
current_animation_start_time = None
animation_time_multiplier = None
animation_paused = None
draw_first_relative_coordinates = None
do_first_rotate = None
draw_second_relative_coordinates = None
do_second_rotate = None
draw_third_relative_coordinates = None
do_third_rotate = None
project_onto_yz_plane = None
rotate_yz_90 = None
undo_rotate_z = None
draw_undo_rotate_z_relative_coordinates = None
undo_rotate_y = None
draw_undo_rotate_y_relative_coordinates = None
undo_rotate_x = None
draw_undo_rotate_x_relative_coordinates = None
do_scale = None
use_ortho = None
new_b = None
angle_x = None
draw_coordinate_system_of_natural_basis = None
step_number = None


def restart():
    global animation_time
    animation_time = 0.0
    global current_animation_start_time
    current_animation_start_time = 0.0
    global animation_time_multiplier
    animation_time_multiplier = 1.0
    global animation_paused
    animation_paused = False
    global draw_first_relative_coordinates
    draw_first_relative_coordinates = False
    global do_first_rotate
    do_first_rotate = False
    global draw_second_relative_coordinates
    draw_second_relative_coordinates = False
    global do_second_rotate
    do_second_rotate = False
    global draw_third_relative_coordinates
    draw_third_relative_coordinates = False
    global do_third_rotate
    do_third_rotate = False
    global project_onto_yz_plane
    project_onto_yz_plane = False
    global rotate_yz_90
    rotate_yz_90 = False
    global undo_rotate_z
    undo_rotate_z = False
    global undo_rotate_y
    undo_rotate_y = False
    global undo_rotate_x
    undo_rotate_x = False
    global do_scale
    do_scale = False
    global use_ortho
    use_ortho = False
    global new_b
    new_b = None
    global angle_x
    angle_x = None
    global draw_coordinate_system_of_natural_basis
    draw_coordinate_system_of_natural_basis = True
    global do_remove_ground
    do_remove_ground = False
    global step_number
    step_number = 0

    global draw_undo_rotate_x_relative_coordinates
    draw_undo_rotate_x_relative_coordinates = False
    global draw_undo_rotate_y_relative_coordinates
    draw_undo_rotate_y_relative_coordinates = False
    global draw_undo_rotate_z_relative_coordinates
    draw_undo_rotate_z_relative_coordinates = False


# initiliaze
restart()


def current_animation_ratio():
    if step_number == 0:
        return 0.0
    return min(1.0, (animation_time - current_animation_start_time) / 2.0)


with compile_shader("lines.vert", "lines.frag") as lines_shader:

    def draw_ground(time, xy=True, yz=False, zx=False):
        do_draw_lines(lines_shader, ground_vertices(), time, xy, yz, zx)

    def draw_unit_circle(time, xy=True, yz=False, zx=False):
        do_draw_lines(lines_shader, unit_circle_vertices(), time, xy, yz, zx)

    def draw_vector(v):
        do_draw_vector(lines_shader, v)

    def draw_axis():
        do_draw_axis(lines_shader)

    # Loop until the user closes the window
    while not glfw.window_should_close(window):

        # poll the time to try to get a constant framerate
        while glfw.get_time() < time_at_beginning_of_previous_frame + 1.0 / TARGET_FRAMERATE:
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

        glEnable(GL_DEPTH_TEST)

        # render scene
        handle_inputs()

        ms.set_to_identity_matrix(ms.MatrixStack.model)
        ms.set_to_identity_matrix(ms.MatrixStack.view)
        ms.set_to_identity_matrix(ms.MatrixStack.projection)

        if use_ortho:
            ms.ortho(
                left=-10.0 * float(width) / float(height),
                right=10.0 * float(width) / float(height),
                bottom=-10.00,
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

        if draw_coordinate_system_of_natural_basis:
            if not do_remove_ground:
                draw_ground(animation_time)
                draw_ground(animation_time, xy=False, zx=True)
                draw_unit_circle(animation_time)
                draw_unit_circle(animation_time, xy=False, zx=True)

        draw_axis()

        imgui.new_frame()

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                clicked_quit, selected_quit = imgui.menu_item("Quit", "Cmd+Q", False, True)

                if clicked_quit:
                    exit(0)

                imgui.end_menu()
            imgui.end_main_menu_bar()

        imgui.set_next_window_bg_alpha(0.05)
        imgui.set_next_window_size(400, 100)
        imgui.set_next_window_position(0, 0, imgui.FIRST_USE_EVER)
        imgui.begin("Input Vectors", True)

        changed, (vec1.x, vec1.y, vec1.z,) = imgui.input_float3(
            label="vec A",
            value0=vec1.x,
            value1=vec1.y,
            value2=vec1.z,
        )

        if changed:
            animation_time = 0.0
            step_number = 0

        changed, (vec2.x, vec2.y, vec2.z,) = imgui.input_float3(
            label="vec B",
            value0=vec2.x,
            value1=vec2.y,
            value2=vec2.z,
        )

        if changed:
            animation_time = 0.0
            step_number = 0

        imgui.end()

        imgui.set_next_window_bg_alpha(0.05)
        imgui.set_next_window_size(400, 100)
        imgui.set_next_window_position(0, 100, imgui.FIRST_USE_EVER)
        imgui.begin("Camera", True)

        clicked = imgui.button("Perspective View" if use_ortho else "Orthogonal View")
        if clicked:
            use_ortho = not use_ortho

        imgui.same_line()

        clicked_camera, camera.r = imgui.slider_float("Camera Radius", camera.r, 3, 100.0)

        if imgui.button("View Down X Axis"):
            camera.rot_x = 0.0
            camera.rot_y = math.pi / 2.0
        imgui.same_line()
        if imgui.button("View Down Negative Y Axis"):
            camera.rot_x = 0.0
            camera.rot_y = 0.0
        imgui.same_line()
        if imgui.button("View Down Z Axis"):
            camera.rot_x = math.pi / 2.0
            camera.rot_y = 0.0

        if imgui.button(
            "Draw Coordinate System Of Natural Basis"
            if not draw_coordinate_system_of_natural_basis
            else "Don't Draw Coordinate System Of Natural Basis"
        ):
            draw_coordinate_system_of_natural_basis = not draw_coordinate_system_of_natural_basis

        imgui.end()

        imgui.set_next_window_position(0, 200, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(400, 100)
        imgui.set_next_window_bg_alpha(0.05)
        imgui.begin("Time", True)

        (
            clicked_animation_time_multiplier,
            animation_time_multiplier,
        ) = imgui.slider_float("Sim Speed", animation_time_multiplier, 0.1, 10.0)
        if imgui.button("Restart"):
            restart()
        if step_number == 0:
            changed, draw_first_relative_coordinates = imgui.checkbox(
                label="Draw Relative Coordinates",
                state=draw_first_relative_coordinates,
            )
            imgui.same_line()
            changed, do_first_rotate = imgui.checkbox(label="Rotate Z", state=do_first_rotate)
            if changed:
                current_animation_start_time = animation_time
                step_number = 1

                def calc_angle_x():
                    a1 = vec1.x
                    a2 = vec1.y
                    a3 = vec1.z
                    mag_a = np.sqrt(a1**2 + a2**2 + a3**2)
                    b1 = vec2.x
                    b2 = vec2.y
                    b3 = vec2.z
                    k1 = np.sqrt(a1**2 + a2**2)

                    b_doubleprime_2 = (-a2 * b1) / k1 + (a1 * b2) / k1
                    b_doubleprime_3 = (
                        (-a1 * a3 * b1) / (k1 * mag_a) + (-a2 * a3 * b2) / (k1 * mag_a) + (k1 * b3) / mag_a
                    )

                    angle = math.atan2(b_doubleprime_3, b_doubleprime_2)
                    return angle if angle > 0.0 else angle + 2 * np.pi

                angle_x = calc_angle_x()
                print(angle_x)

        if step_number == 1:
            changed, draw_second_relative_coordinates = imgui.checkbox(
                label="Draw Second Relative Coordinates",
                state=draw_second_relative_coordinates,
            )
            imgui.same_line()
            changed, do_second_rotate = imgui.checkbox(label="Rotate Y", state=do_second_rotate)
            if changed:
                step_number = 2
                current_animation_start_time = animation_time

        if step_number == 2:
            changed, draw_third_relative_coordinates = imgui.checkbox(
                label="Draw Third Relative Coordinates",
                state=draw_third_relative_coordinates,
            )
            imgui.same_line()
            changed, do_third_rotate = imgui.checkbox(label="Rotate X", state=do_third_rotate)
            if changed:
                step_number = 3
                current_animation_start_time = animation_time

        if do_third_rotate:
            ratio = current_animation_ratio() if step_number == 3 else 1.0
            ms.rotate_x(ms.MatrixStack.model, -angle_x * ratio)
            if ratio > 0.9999:
                draw_third_relative_coordinates = False
            ratio = current_animation_ratio() if step_number == 7 else 0.0 if step_number < 7 else 1.0
            ms.rotate_x(ms.MatrixStack.model, angle_x * ratio)
            if draw_undo_rotate_x_relative_coordinates and not do_remove_ground:
                draw_ground(animation_time, xy=False, yz=True)
                draw_axis()

        if draw_third_relative_coordinates:
            with ms.push_matrix(ms.MatrixStack.model):
                ratio = current_animation_ratio() if step_number == 4 else 1.0
                ms.rotate_x(ms.MatrixStack.model, angle_x * ratio)

                draw_ground(animation_time, xy=False, yz=True)
                draw_axis()

        if do_second_rotate:
            ratio = current_animation_ratio() if step_number == 2 else 1.0
            ms.rotate_y(ms.MatrixStack.model, -vec1.angle_y * ratio)
            if ratio > 0.99:
                draw_second_relative_coordinates = False
            ratio = current_animation_ratio() if step_number == 8 else 0.0 if step_number < 8 else 1.0
            ms.rotate_y(ms.MatrixStack.model, vec1.angle_y * ratio)
            if draw_undo_rotate_y_relative_coordinates and not do_remove_ground:
                draw_ground(animation_time, xy=False, zx=True)
                draw_axis()

        if draw_second_relative_coordinates:

            with ms.push_matrix(ms.MatrixStack.model):

                ratio = current_animation_ratio() if step_number == 3 else 1.0
                ms.rotate_y(ms.MatrixStack.model, vec1.angle_y * ratio)
                draw_ground(animation_time, xy=False, zx=True)
                draw_axis()

        if do_first_rotate:
            ratio = current_animation_ratio() if step_number == 1 else 1.0
            ms.rotate_z(ms.MatrixStack.model, -vec1.angle_z * ratio)
            if ratio > 0.99:
                draw_first_relative_coordinates = False
            ratio = current_animation_ratio() if step_number == 9 else 0.0 if step_number < 9 else 1.0
            ms.rotate_z(ms.MatrixStack.model, vec1.angle_z * ratio)
            if draw_undo_rotate_z_relative_coordinates and not do_remove_ground:
                draw_ground(animation_time)
                draw_axis()

        if draw_first_relative_coordinates:
            with ms.push_matrix(ms.MatrixStack.model):
                ratio = current_animation_ratio() if step_number == 2 else 1.0
                ms.rotate_z(ms.MatrixStack.model, vec1.angle_z * ratio)
                draw_ground(animation_time)
                draw_axis()

        if step_number == 3 or step_number == 4:
            if imgui.button("Show Triangle"):
                vec3_after_rotate = np.ascontiguousarray(
                    ms.get_current_matrix(ms.MatrixStack.model),
                    dtype=np.float32,
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
                step_number = 4

            imgui.same_line()
            if imgui.button("Project onto e_2 e_3 plane"):
                project_onto_yz_plane = True
                step_number = 5
                current_animation_start_time = animation_time

        if step_number == 5:

            if imgui.button("Rotate Y to Z, Z to -Y"):
                rotate_yz_90 = True
                step_number = 6
                current_animation_start_time = animation_time

        if step_number == 6:
            changed, draw_undo_rotate_x_relative_coordinates = imgui.checkbox(
                label="Draw Relative Coordinates",
                state=draw_undo_rotate_x_relative_coordinates,
            )
            imgui.same_line()
            if imgui.button("Undo Rotate X"):
                undo_rotate_x = True
                step_number = 7
                current_animation_start_time = animation_time

        if step_number == 7:
            changed, draw_undo_rotate_y_relative_coordinates = imgui.checkbox(
                label="Draw Relative Coordinates",
                state=draw_undo_rotate_y_relative_coordinates,
            )
            imgui.same_line()
            if imgui.button("Undo Rotate Y"):
                undo_rotate_y = True
                step_number = 8
                current_animation_start_time = animation_time

        if step_number == 8:
            changed, draw_undo_rotate_z_relative_coordinates = imgui.checkbox(
                label="Draw Relative Coordinates",
                state=draw_undo_rotate_z_relative_coordinates,
            )
            imgui.same_line()
            if imgui.button("Undo Rotate Z"):
                undo_rotate_z = True
                step_number = 9
                current_animation_start_time = animation_time

        if step_number == 9:
            if imgui.button("Scale By Magnitude of first vector"):
                do_scale = True
                step_number = 10

        if step_number == 10:
            if imgui.button("Show Plane spanned by vec a and vec b"):
                do_remove_ground = True

        imgui.end()

        if vec3 and (step_number >= 4):
            with ms.push_matrix(ms.MatrixStack.model):
                ms.set_to_identity_matrix(ms.MatrixStack.model)
                ms.rotate_x(ms.MatrixStack.model, math.radians(-90.0))

                if undo_rotate_z:
                    ratio = current_animation_ratio() if step_number == 9 else 0.0 if step_number < 9 else 1.0
                    ms.rotate_z(ms.MatrixStack.model, vec1.angle_z * ratio)
                if undo_rotate_y:
                    ratio = current_animation_ratio() if step_number == 8 else 0.0 if step_number < 8 else 1.0
                    ms.rotate_y(ms.MatrixStack.model, vec1.angle_y * ratio)
                if undo_rotate_x:
                    ratio = current_animation_ratio() if step_number == 7 else 0.0 if step_number < 7 else 1.0
                    ms.rotate_x(ms.MatrixStack.model, angle_x * ratio)

                if do_scale:
                    if do_remove_ground:
                        draw_ground(animation_time)

                    magnitude = math.sqrt(vec1.x**2 + vec1.y**2 + vec1.z**2)

                    ms.scale(
                        ms.MatrixStack.model,
                        magnitude,
                        magnitude,
                        magnitude,
                    )

                glDisable(GL_DEPTH_TEST)
                ratio = current_animation_ratio() if step_number == 5 else 0.0 if step_number <= 5 else 1.0
                ms.translate(
                    ms.MatrixStack.model,
                    vec3.translate_amount * (1.0 - ratio),
                    0.0,
                    0.0,
                )
                if rotate_yz_90:
                    with ms.push_matrix(ms.MatrixStack.model):
                        ratio = current_animation_ratio() if step_number == 6 else 1.0 if step_number > 6 else 0.0
                        ms.rotate_x(ms.MatrixStack.model, math.radians(90.0 * ratio))
                        draw_vector(vec3)
                else:
                    draw_vector(vec3)
                glEnable(GL_DEPTH_TEST)

        glDisable(GL_DEPTH_TEST)

        draw_vector(vec1)
        draw_vector(vec2)

        imgui.render()
        impl.render(imgui.get_draw_data())

        # done with frame, flush and swap buffers
        # Swap front and back buffers
        glfw.swap_buffers(window)


glfw.terminate()
