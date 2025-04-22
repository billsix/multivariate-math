# Copyright (c) 2018-2023 William Emerison Six
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


import functools
import math
import sys
from dataclasses import dataclass
from enum import Enum, auto

import glfw
import imgui
import numpy as np
import pyMatrixStack as ms
from imgui.integrations.glfw import GlfwRenderer
from numpy import ndarray
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LESS,
    GL_LINEAR,
    GL_RED,
    GL_REPEAT,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRUE,
    GL_UNSIGNED_BYTE,
    glBindTexture,
    glClear,
    glClearColor,
    glClearDepth,
    glDepthFunc,
    glDisable,
    glEnable,
    glGenerateMipmap,
    glGenTextures,
    glTexImage2D,
    glTexParameteri,
    glViewport,
)
from PIL import Image, ImageOps

from renderer import Camera, Vector, compile_shader, do_draw_axis, do_draw_lines, do_draw_vector

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


def scroll_callback(window, xoffset, yoffset):
    global g
    g.camera.r = g.camera.r + -1 * (yoffset * math.log(g.camera.r))
    if g.camera.r < 3.0:
        g.camera.r = 3.0


glfw.set_scroll_callback(window, scroll_callback)


glClearColor(0.0, 0.0, 0.0, 1.0)

# NEW - TODO - talk about opengl matricies and z pos/neg
glClearDepth(1.0)
glDepthFunc(GL_LESS)
glEnable(GL_DEPTH_TEST)


@functools.cache
def ground_vertices() -> ndarray:
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
def unit_circle_vertices() -> ndarray:
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


swap = False


class StepNumber(Enum):
    beginning = auto()  # 0
    rotate_z = auto()  # 1
    rotate_y = auto()  # 2
    rotate_x = auto()  # 3
    show_triangle = auto()  # 4
    project_onto_y = auto()  # 5
    rotate_to_z = auto()  # 6
    undo_rotate_x = auto()  # 7
    undo_rotate_y = auto()  # 8
    undo_rotate_z = auto()  # 9
    scale_by_mag_a = auto()  # 10
    show_plane = auto()  # 11


@dataclass
class Globals:
    vec1: any
    vec2: any
    vec3: any
    swap: any
    camera: any
    use_ortho: any
    animation_time: any
    current_animation_start_time: any
    animation_time_multiplier: any
    animation_paused: any
    draw_first_relative_coordinates: any
    do_first_rotate: any
    draw_second_relative_coordinates: any
    do_second_rotate: any
    draw_third_relative_coordinates: any
    do_third_rotate: any
    project_onto_yz_plane: any
    rotate_yz_90: any
    undo_rotate_z: any
    undo_rotate_y: any
    undo_rotate_x: any
    do_scale: any
    use_ortho: any
    new_b: any
    angle_x: any
    draw_coordinate_system_of_natural_basis: any
    do_remove_ground: any
    step_number: any
    draw_undo_rotate_x_relative_coordinates: any
    draw_undo_rotate_y_relative_coordinates: any
    draw_undo_rotate_z_relative_coordinates: any
    auto_rotate_camera: any
    seconds_per_operation: any
    auto_play: any
    time_at_beginning_of_previous_frame: any
    highlight_x: bool
    highlight_y: bool
    highlight_z: bool
    highlight_relative_x: bool
    highlight_relative_y: bool
    highlight_relative_z: bool


g = Globals(
    animation_time=None,
    current_animation_start_time=None,
    animation_time_multiplier=None,
    animation_paused=None,
    draw_first_relative_coordinates=None,
    do_first_rotate=None,
    draw_second_relative_coordinates=None,
    do_second_rotate=None,
    draw_third_relative_coordinates=None,
    do_third_rotate=None,
    project_onto_yz_plane=None,
    rotate_yz_90=None,
    undo_rotate_z=None,
    draw_undo_rotate_z_relative_coordinates=None,
    undo_rotate_y=None,
    draw_undo_rotate_y_relative_coordinates=None,
    undo_rotate_x=None,
    draw_undo_rotate_x_relative_coordinates=None,
    do_scale=None,
    use_ortho=None,
    new_b=None,
    angle_x=None,
    draw_coordinate_system_of_natural_basis=None,
    step_number=StepNumber.beginning,
    auto_rotate_camera=False,
    seconds_per_operation=2.0,
    auto_play=False,
    vec1=None,
    vec2=None,
    vec3=None,
    swap=None,
    camera=None,
    do_remove_ground=None,
    time_at_beginning_of_previous_frame=glfw.get_time(),
    highlight_x=False,
    highlight_y=False,
    highlight_z=False,
    highlight_relative_x=False,
    highlight_relative_y=False,
    highlight_relative_z=False,
)


def initiliaze_vecs() -> None:
    global g
    if not g.swap:
        g.vec1 = Vector(x=3.0, y=4.0, z=5.0, r=1.0, g=0.5, b=0.0, highlight=False)
        g.vec2 = Vector(x=-1.0, y=2.0, z=2.0, r=0.5, g=0.0, b=1.0, highlight=False)
    else:
        g.vec1 = Vector(x=-1.0, y=2.0, z=2.0, r=1.0, g=0.5, b=0.0, highlight=False)
        g.vec2 = Vector(x=3.0, y=4.0, z=5.0, r=0.5, g=0.0, b=1.0, highlight=False)

    g.vec3 = None


initiliaze_vecs()

g.camera = Camera(r=22.0, rot_y=math.radians(45.0), rot_x=math.radians(35.264))


def handle_inputs(previous_mouse_position) -> None:
    if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
        g.camera.rot_y -= math.radians(1.0)
        g.use_ortho = False
    if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
        g.camera.rot_y += math.radians(1.0)
        g.use_ortho = False
    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        g.camera.rot_x -= math.radians(1.0)
        g.use_ortho = False
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        g.camera.rot_x += math.radians(1.0)
        g.use_ortho = False

    new_mouse_position = glfw.get_cursor_pos(window)
    return_none = False
    if glfw.PRESS == glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT):
        if previous_mouse_position:
            g.camera.rot_y -= 0.2 * math.radians(new_mouse_position[0] - previous_mouse_position[0])
            g.camera.rot_x += 0.2 * math.radians(new_mouse_position[1] - previous_mouse_position[1])
            g.use_ortho = False
    else:
        return_none = True

    if g.camera.rot_x > math.pi / 2.0:
        g.camera.rot_x = math.pi / 2.0
    if g.camera.rot_x < -math.pi / 2.0:
        g.camera.rot_x = -math.pi / 2.0

    return None if return_none else new_mouse_position


TARGET_FRAMERATE = 60  # fps

# to try to standardize on 60 fps, compare times between frames
g.time_at_beginning_of_previous_frame = glfw.get_time()


def restart() -> None:
    global g
    g = Globals(
        animation_time=0.0,
        current_animation_start_time=0.0,
        animation_time_multiplier=1.0,
        animation_paused=False,
        draw_first_relative_coordinates=False,
        do_first_rotate=False,
        draw_second_relative_coordinates=False,
        do_second_rotate=False,
        draw_third_relative_coordinates=False,
        do_third_rotate=False,
        project_onto_yz_plane=False,
        rotate_yz_90=False,
        undo_rotate_z=False,
        undo_rotate_y=False,
        undo_rotate_x=False,
        do_scale=False,
        use_ortho=False,
        new_b=None,
        angle_x=None,
        draw_coordinate_system_of_natural_basis=True,
        do_remove_ground=False,
        step_number=StepNumber.beginning,
        draw_undo_rotate_x_relative_coordinates=False,
        draw_undo_rotate_y_relative_coordinates=False,
        draw_undo_rotate_z_relative_coordinates=False,
        auto_rotate_camera=False,
        auto_play=False,
        vec1=g.vec1,
        vec2=g.vec2,
        vec3=g.vec3,
        swap=False,
        camera=g.camera,
        seconds_per_operation=2.0,
        time_at_beginning_of_previous_frame=glfw.get_time(),
        highlight_x=False,
        highlight_y=False,
        highlight_z=False,
        highlight_relative_x=False,
        highlight_relative_y=False,
        highlight_relative_z=False,
    )


# initiliaze
restart()


# Load a math image, and make the black white, and white black
def load_and_flip_image(image_path):
    image = Image.open(image_path).convert("L")
    inverted_image = ImageOps.invert(image)
    return inverted_image


# Function to bind the image to an OpenGL texture
def generate_texture(image):
    image_data = np.array(list(image.getdata()), np.uint8)
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, image.width, image.height, 0, GL_RED, GL_UNSIGNED_BYTE, image_data)
    glGenerateMipmap(GL_TEXTURE_2D)
    return texture_id


x_image = load_and_flip_image("./images/x.png")
y_image = load_and_flip_image("./images/y.png")
z_image = load_and_flip_image("./images/z.png")

a_image = load_and_flip_image("./images/a.png")
aprime_image = load_and_flip_image("./images/aprime.png")
aprimeprime_image = load_and_flip_image("./images/aprimeprime.png")
b_image = load_and_flip_image("./images/b.png")
bprimeprime_image = load_and_flip_image("./images/bprimeprime.png")
bprimeprimeprime_image = load_and_flip_image("./images/bprimeprimeprime.png")


def current_animation_ratio() -> float:
    if g.step_number == StepNumber.beginning:
        return 0.0
    return min(
        1.0,
        (g.animation_time - g.current_animation_start_time) / g.seconds_per_operation,
    )


with (
    compile_shader("lines.vert", "lines.frag", "lines.geom") as lines_shader,
    compile_shader("image.vert", "image.frag", None) as image_shader,
):

    def draw_ground(
        time: float,
        width: int,
        height: int,
        xy: bool = True,
        yz: bool = False,
        zx: bool = False,
    ) -> None:
        do_draw_lines(lines_shader, ground_vertices(), time, width, height, xy, yz, zx)

    def draw_unit_circle(
        time: float,
        width: int,
        height: int,
        xy: bool = True,
        yz: bool = False,
        zx: bool = False,
    ) -> None:
        do_draw_lines(lines_shader, unit_circle_vertices(), time, width, height, xy, yz, zx)

    def draw_vector(v: Vector, width: int, height: int) -> None:
        do_draw_vector(lines_shader, v, width, height)

    def draw_axis(
        width: int,
        height: int,
        highlight_x: bool = False,
        highlight_y: bool = False,
        highlight_z: bool = False,
    ) -> None:
        do_draw_axis(lines_shader, width, height, highlight_x, highlight_y, highlight_z)

    # local variable for event loop
    previous_mouse_position = None
    # Loop until the user closes the window
    while not glfw.window_should_close(window):
        # poll the time to try to get a constant framerate
        while glfw.get_time() < g.time_at_beginning_of_previous_frame + 1.0 / TARGET_FRAMERATE:
            pass
        # set for comparison on the next frame
        g.time_at_beginning_of_previous_frame = glfw.get_time()

        if not g.animation_paused:
            g.animation_time += 1.0 / 60.0 * g.animation_time_multiplier

        # Poll for and process events
        glfw.poll_events()
        impl.process_inputs()

        width, height = glfw.get_framebuffer_size(window)
        glViewport(0, 0, width, height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glEnable(GL_DEPTH_TEST)

        previous_mouse_position = handle_inputs(previous_mouse_position)

        # render scene
        ms.set_to_identity_matrix(ms.MatrixStack.model)
        ms.set_to_identity_matrix(ms.MatrixStack.view)
        ms.set_to_identity_matrix(ms.MatrixStack.projection)

        if g.use_ortho:
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
                field_of_view=45.0,
                aspect_ratio=float(width) / float(height),
                near_z=0.1,
                far_z=10000.0,
            )

            # note - opengl matricies use degrees
            ms.translate(ms.MatrixStack.view, 0.0, 0.0, -g.camera.r)

        ms.rotate_x(ms.MatrixStack.view, g.camera.rot_x)
        ms.rotate_y(ms.MatrixStack.view, -g.camera.rot_y)

        # do everything in math coordinate system
        ms.rotate_x(ms.MatrixStack.model, math.radians(-90.0))

        if g.draw_coordinate_system_of_natural_basis:
            if not g.do_remove_ground:
                draw_ground(g.animation_time, width, height)
                draw_ground(g.animation_time, width, height, xy=False, zx=True)
                draw_unit_circle(g.animation_time, width, height, xy=True, yz=True, zx=True)

        draw_axis(
            width,
            height,
            highlight_x=g.highlight_x,
            highlight_y=g.highlight_y,
            highlight_z=g.highlight_z,
        )

        imgui.new_frame()

        imgui.set_next_window_bg_alpha(0.05)
        imgui.set_next_window_size(300, 175, imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(0, 0, imgui.FIRST_USE_EVER)
        imgui.begin("Cross Product", True)

        def a_extra_text():
            match g.step_number:
                case StepNumber.beginning:
                    return ""
                case StepNumber.rotate_z:
                    return "'"
                case StepNumber.rotate_y:
                    return "''"
                case StepNumber.rotate_x:
                    return "'''"
                case StepNumber.show_triangle:
                    return "'''"
                case StepNumber.project_onto_y:
                    return "'''"
                case StepNumber.rotate_to_z:
                    return "'''"
                case StepNumber.undo_rotate_x:
                    return "''"
                case StepNumber.undo_rotate_y:
                    return "'"
                case StepNumber.undo_rotate_z:
                    return ""
                case StepNumber.scale_by_mag_a:
                    return ""
                case StepNumber.show_plane:
                    return ""

        show, _ = imgui.collapsing_header("Input Vectors", flags=imgui.TREE_NODE_DEFAULT_OPEN)
        if show:
            imgui.text("a x b")
            imgui.label_text("label", "Value")
            if g.step_number == StepNumber.beginning:
                (
                    changed,
                    (
                        g.vec1.x,
                        g.vec1.y,
                        g.vec1.z,
                    ),
                ) = imgui.input_float3(
                    label="vec a" + a_extra_text(),
                    value0=g.vec1.x,
                    value1=g.vec1.y,
                    value2=g.vec1.z,
                )

                if changed:
                    g.animation_time = 0.0
                    g.step_number = StepNumber.beginning

                (
                    changed,
                    (
                        g.vec2.x,
                        g.vec2.y,
                        g.vec2.z,
                    ),
                ) = imgui.input_float3(
                    label="vec b" + a_extra_text(),
                    value0=g.vec2.x,
                    value1=g.vec2.y,
                    value2=g.vec2.z,
                )

                clicked = imgui.button("G.Swap vectors")
                if clicked:
                    g.swap = not g.swap
                    initiliaze_vecs()

                if changed:
                    g.animation_time = 0.0
                    g.step_number = StepNumber.beginning

        show, _ = imgui.collapsing_header("Camera", flags=imgui.TREE_NODE_DEFAULT_OPEN)
        if show:
            changed, g.auto_rotate_camera = imgui.checkbox(label="Auto Rotate G.Camera", state=g.auto_rotate_camera)

            if g.auto_rotate_camera:
                g.camera.rot_y += math.radians(0.1)

            if not g.use_ortho:
                clicked, g.camera.r = imgui.slider_float("Camera Radius", g.camera.r, 3, 130.0)

            if imgui.button("View Down x Axis"):
                g.camera.rot_x = 0.0
                g.camera.rot_y = math.pi / 2.0
                g.use_ortho = True

            if imgui.button("View Down Negative y Axis"):
                g.camera.rot_x = 0.0
                g.camera.rot_y = 0.0
                g.use_ortho = True

            if imgui.button("View Down z Axis"):
                g.camera.rot_x = math.pi / 2.0
                g.camera.rot_y = 0.0
                g.use_ortho = True

            changed, g.draw_coordinate_system_of_natural_basis = imgui.checkbox(
                label="Draw Coordinate System of Natural Basis",
                state=g.draw_coordinate_system_of_natural_basis,
            )

        show, _ = imgui.collapsing_header("Time", flags=imgui.TREE_NODE_DEFAULT_OPEN)
        if show:
            changed, g.auto_play = imgui.checkbox(label="AutoPlay", state=g.auto_play)

            if imgui.button("Restart"):
                restart()

            imgui.label_text("label", "Value")
            changed, (g.seconds_per_operation) = imgui.input_float("Seconds Per Operation", g.seconds_per_operation)

            if g.step_number == StepNumber.beginning:
                if imgui.button("Rotate Z") or g.auto_play:
                    g.do_first_rotate = True
                    g.current_animation_start_time = g.animation_time
                    g.step_number = StepNumber.rotate_z

                    def calc_angle_x() -> float:
                        a1, a2, a3 = g.vec1.x, g.vec1.y, g.vec1.z
                        mag_a = np.sqrt(a1**2 + a2**2 + a3**2)
                        b1, b2, b3 = g.vec2.x, g.vec2.y, g.vec2.z
                        k1 = np.sqrt(a1**2 + a2**2)

                        if k1 < 0.0001:
                            angle = 0.0
                        else:
                            b_doubleprime_2 = (-a2 * b1) / k1 + (a1 * b2) / k1
                            b_doubleprime_3 = (
                                (-a1 * a3 * b1) / (k1 * mag_a) + (-a2 * a3 * b2) / (k1 * mag_a) + (k1 * b3) / mag_a
                            )

                            angle = math.atan2(b_doubleprime_3, b_doubleprime_2)
                        return angle if abs(angle) <= np.pi / 2.0 else (angle - 2 * np.pi)

                    g.angle_x = calc_angle_x()
                imgui.same_line()
                changed, g.draw_first_relative_coordinates = imgui.checkbox(
                    label="Draw Relative Coordinates",
                    state=g.draw_first_relative_coordinates,
                )
                if g.draw_first_relative_coordinates:
                    imgui.text("Highlight:")
                    imgui.same_line()
                    if imgui.button("x'"):
                        g.highlight_relative_x = not g.highlight_relative_x
                    imgui.same_line()
                    if imgui.button("y'"):
                        g.highlight_relative_y = not g.highlight_relative_y
                    imgui.same_line()
                    if imgui.button("z'"):
                        g.highlight_relative_z = not g.highlight_relative_z

            if g.step_number == StepNumber.rotate_z:
                if current_animation_ratio() >= 0.999999:
                    if imgui.button("Rotate Y") or g.auto_play:
                        g.do_second_rotate = True
                        g.step_number = StepNumber.rotate_y
                        g.current_animation_start_time = g.animation_time
                    imgui.same_line()
                    changed, g.draw_second_relative_coordinates = imgui.checkbox(
                        label="Draw Second Relative Coordinates",
                        state=g.draw_second_relative_coordinates,
                    )
                if g.draw_second_relative_coordinates:
                    imgui.text("Highlight:")
                    imgui.same_line()
                    if imgui.button("x'"):
                        g.highlight_relative_x = not g.highlight_relative_x
                    imgui.same_line()
                    if imgui.button("y'"):
                        g.highlight_relative_y = not g.highlight_relative_y
                    imgui.same_line()
                    if imgui.button("z'"):
                        g.highlight_relative_z = not g.highlight_relative_z

            if g.step_number == StepNumber.rotate_y:
                if current_animation_ratio() >= 0.999999:
                    if imgui.button("Rotate X") or g.auto_play:
                        g.do_third_rotate = True
                        g.step_number = StepNumber.rotate_x
                        g.current_animation_start_time = g.animation_time
                    imgui.same_line()
                    changed, g.draw_third_relative_coordinates = imgui.checkbox(
                        label="Draw Third Relative Coordinates",
                        state=g.draw_third_relative_coordinates,
                    )
                if g.draw_third_relative_coordinates:
                    imgui.text("Highlight:")
                    imgui.same_line()
                    if imgui.button("x'"):
                        g.highlight_relative_x = not g.highlight_relative_x
                    imgui.same_line()
                    if imgui.button("y'"):
                        g.highlight_relative_y = not g.highlight_relative_y
                    imgui.same_line()
                    if imgui.button("z'"):
                        g.highlight_relative_z = not g.highlight_relative_z

            if g.do_third_rotate:
                ratio = current_animation_ratio() if g.step_number == StepNumber.rotate_x else 1.0
                ms.rotate_x(ms.MatrixStack.model, -g.angle_x * ratio)
                if ratio > 0.9999:
                    g.draw_third_relative_coordinates = False
                ratio = (
                    current_animation_ratio()
                    if g.step_number == StepNumber.undo_rotate_x
                    else 0.0
                    if g.step_number.value < StepNumber.undo_rotate_x.value
                    else 1.0
                )
                ms.rotate_x(ms.MatrixStack.model, g.angle_x * ratio)
                if g.draw_undo_rotate_x_relative_coordinates and not g.do_remove_ground:
                    draw_ground(g.animation_time, width, height, xy=False, yz=True)
                    draw_axis(
                        width,
                        height,
                        highlight_x=g.highlight_relative_x,
                        highlight_y=g.highlight_relative_y,
                        highlight_z=g.highlight_relative_z,
                    )

            if g.draw_third_relative_coordinates:
                with ms.push_matrix(ms.MatrixStack.model):
                    ratio = current_animation_ratio() if g.step_number == StepNumber.show_triangle.value else 1.0
                    ms.rotate_x(ms.MatrixStack.model, g.angle_x * ratio)

                    draw_ground(g.animation_time, width, height, xy=False, yz=True)
                    draw_axis(
                        width,
                        height,
                        highlight_x=g.highlight_relative_x,
                        highlight_y=g.highlight_relative_y,
                        highlight_z=g.highlight_relative_z,
                    )

            if g.do_second_rotate:
                ratio = current_animation_ratio() if g.step_number == StepNumber.rotate_y else 1.0
                ms.rotate_y(ms.MatrixStack.model, -g.vec1.angle_y * ratio)
                if ratio > 0.99:
                    g.draw_second_relative_coordinates = False
                ratio = (
                    current_animation_ratio()
                    if g.step_number == StepNumber.undo_rotate_y
                    else 0.0
                    if g.step_number.value < StepNumber.undo_rotate_y.value
                    else 1.0
                )
                ms.rotate_y(ms.MatrixStack.model, g.vec1.angle_y * ratio)
                if g.draw_undo_rotate_y_relative_coordinates and not g.do_remove_ground:
                    draw_ground(g.animation_time, width, height, xy=False, zx=True)
                    draw_axis(
                        width,
                        height,
                        highlight_x=g.highlight_relative_x,
                        highlight_y=g.highlight_relative_y,
                        highlight_z=g.highlight_relative_z,
                    )

            if g.draw_second_relative_coordinates:
                with ms.push_matrix(ms.MatrixStack.model):
                    ratio = current_animation_ratio() if g.step_number == StepNumber.rotate_x else 1.0
                    ms.rotate_y(ms.MatrixStack.model, g.vec1.angle_y * ratio)
                    draw_ground(g.animation_time, width, height, xy=False, zx=True)
                    draw_axis(
                        width,
                        height,
                        highlight_x=g.highlight_relative_x,
                        highlight_y=g.highlight_relative_y,
                        highlight_z=g.highlight_relative_z,
                    )

            if g.do_first_rotate:
                ratio = current_animation_ratio() if g.step_number == StepNumber.rotate_z else 1.0
                ms.rotate_z(ms.MatrixStack.model, -g.vec1.angle_z * ratio)
                if ratio > 0.99:
                    g.draw_first_relative_coordinates = False
                ratio = (
                    current_animation_ratio()
                    if g.step_number == StepNumber.undo_rotate_z
                    else 0.0
                    if g.step_number.value < StepNumber.undo_rotate_z.value
                    else 1.0
                )
                ms.rotate_z(ms.MatrixStack.model, g.vec1.angle_z * ratio)
                if g.draw_undo_rotate_z_relative_coordinates and not g.do_remove_ground:
                    draw_ground(g.animation_time, width, height)
                    draw_axis(
                        width,
                        height,
                        highlight_x=g.highlight_relative_x,
                        highlight_y=g.highlight_relative_y,
                        highlight_z=g.highlight_relative_z,
                    )

            if g.draw_first_relative_coordinates:
                with ms.push_matrix(ms.MatrixStack.model):
                    ratio = current_animation_ratio() if g.step_number == StepNumber.rotate_y else 1.0
                    ms.rotate_z(ms.MatrixStack.model, g.vec1.angle_z * ratio)
                    draw_ground(g.animation_time, width, height)
                    draw_axis(
                        width,
                        height,
                        highlight_x=g.highlight_relative_x,
                        highlight_y=g.highlight_relative_y,
                        highlight_z=g.highlight_relative_z,
                    )

            if g.step_number == StepNumber.rotate_x:
                if current_animation_ratio() >= 0.999999:
                    if imgui.button("Show Triangle") or g.auto_play:
                        g.vec3_after_rotate = np.ascontiguousarray(
                            ms.get_current_matrix(ms.MatrixStack.model),
                            dtype=np.float32,
                        ) @ np.array([g.vec2.x, g.vec2.y, g.vec2.z, 1.0], dtype=np.float32)

                        g.vec3 = Vector(
                            x=0.0,
                            y=-g.vec3_after_rotate[2],
                            z=g.vec3_after_rotate[1],
                            r=0.0,
                            g=1.0,
                            b=0.0,
                        )
                        g.vec3.translate_amount = g.vec3_after_rotate[0]
                        g.step_number = StepNumber.show_triangle
            if g.step_number == StepNumber.show_triangle:
                if current_animation_ratio() >= 0.999999:
                    if imgui.button("Project onto yz plane") or g.auto_play:
                        g.project_onto_yz_plane = True
                        g.step_number = StepNumber.project_onto_y
                        g.current_animation_start_time = g.animation_time

            if g.step_number == StepNumber.project_onto_y:
                if current_animation_ratio() >= 0.999999:
                    if imgui.button("Rotate Y to Z, Z to -Y") or g.auto_play:
                        g.rotate_yz_90 = True
                        g.step_number = StepNumber.rotate_to_z
                        g.current_animation_start_time = g.animation_time

            if g.step_number == StepNumber.rotate_to_z:
                if current_animation_ratio() >= 0.999999:
                    if imgui.button("Undo Rotate X") or g.auto_play:
                        g.undo_rotate_x = True
                        g.step_number = StepNumber.undo_rotate_x
                        g.current_animation_start_time = g.animation_time
                    imgui.same_line()
                    changed, g.draw_undo_rotate_x_relative_coordinates = imgui.checkbox(
                        label="Draw Relative Coordinates",
                        state=g.draw_undo_rotate_x_relative_coordinates,
                    )

            if g.step_number == StepNumber.undo_rotate_x:
                if current_animation_ratio() >= 0.999999:
                    if imgui.button("Undo Rotate Y") or g.auto_play:
                        g.undo_rotate_y = True
                        g.step_number = StepNumber.undo_rotate_y
                        g.current_animation_start_time = g.animation_time
                    imgui.same_line()
                    changed, g.draw_undo_rotate_y_relative_coordinates = imgui.checkbox(
                        label="Draw Relative Coordinates",
                        state=g.draw_undo_rotate_y_relative_coordinates,
                    )

            if g.step_number == StepNumber.undo_rotate_y:
                if current_animation_ratio() >= 0.999999:
                    if imgui.button("Undo Rotate Z") or g.auto_play:
                        g.undo_rotate_z = True
                        g.step_number = StepNumber.undo_rotate_z
                        g.current_animation_start_time = g.animation_time
                    imgui.same_line()
                    changed, g.draw_undo_rotate_z_relative_coordinates = imgui.checkbox(
                        label="Draw Relative Coordinates",
                        state=g.draw_undo_rotate_z_relative_coordinates,
                    )

            if g.step_number == StepNumber.undo_rotate_z:
                if current_animation_ratio() >= 0.999999:
                    if imgui.button("Scale By Magnitude of first vector") or g.auto_play:
                        g.do_scale = True
                        g.step_number = StepNumber.scale_by_mag_a

            if g.step_number == StepNumber.scale_by_mag_a:
                if current_animation_ratio() >= 0.999999:
                    if imgui.button("Show Plane spanned by vec a and vec b") or g.auto_play:
                        g.do_remove_ground = True

            imgui.text("Highlight:")
            imgui.same_line()
            if imgui.button("x"):
                g.highlight_x = not g.highlight_x
            imgui.same_line()
            if imgui.button("y"):
                g.highlight_y = not g.highlight_y
            imgui.same_line()
            if imgui.button("z"):
                g.highlight_z = not g.highlight_z

            imgui.text("Highlight:")
            imgui.same_line()
            if imgui.button("a" + a_extra_text()):
                g.vec1.highlight = not g.vec1.highlight
                g.vec2.highlight = False
            imgui.same_line()
            if imgui.button("b" + a_extra_text()):
                g.vec2.highlight = not g.vec2.highlight
                g.vec1.highlight = False

        imgui.end()

        if g.vec3 and (g.step_number.value >= StepNumber.show_triangle.value):
            with ms.push_matrix(ms.MatrixStack.model):
                ms.set_to_identity_matrix(ms.MatrixStack.model)
                ms.rotate_x(ms.MatrixStack.model, math.radians(-90.0))

                if g.undo_rotate_z:
                    ratio = (
                        current_animation_ratio()
                        if g.step_number == StepNumber.undo_rotate_z
                        else 0.0
                        if g.step_number.value < StepNumber.undo_rotate_z.value
                        else 1.0
                    )
                    ms.rotate_z(ms.MatrixStack.model, g.vec1.angle_z * ratio)
                if g.undo_rotate_y:
                    ratio = (
                        current_animation_ratio()
                        if g.step_number == StepNumber.undo_rotate_y
                        else 0.0
                        if g.step_number.value < StepNumber.undo_rotate_y.value
                        else 1.0
                    )
                    ms.rotate_y(ms.MatrixStack.model, g.vec1.angle_y * ratio)
                if g.undo_rotate_x:
                    ratio = (
                        current_animation_ratio()
                        if g.step_number == StepNumber.undo_rotate_x
                        else 0.0
                        if g.step_number.value < StepNumber.undo_rotate_x.value
                        else 1.0
                    )
                    ms.rotate_x(ms.MatrixStack.model, g.angle_x * ratio)

                if g.do_scale:
                    if g.do_remove_ground:
                        draw_ground(g.animation_time, width, height)

                    magnitude = math.sqrt(g.vec1.x**2 + g.vec1.y**2 + g.vec1.z**2)

                    ms.scale(
                        ms.MatrixStack.model,
                        1.0,
                        1.0,
                        magnitude,
                    )

                glDisable(GL_DEPTH_TEST)
                ratio = (
                    current_animation_ratio()
                    if g.step_number == StepNumber.project_onto_y
                    else 0.0
                    if g.step_number.value <= StepNumber.project_onto_y.value
                    else 1.0
                )
                ms.translate(
                    ms.MatrixStack.model,
                    g.vec3.translate_amount * (1.0 - ratio),
                    0.0,
                    0.0,
                )
                if g.rotate_yz_90:
                    with ms.push_matrix(ms.MatrixStack.model):
                        ratio = (
                            current_animation_ratio()
                            if g.step_number == StepNumber.rotate_to_z
                            else 1.0
                            if g.step_number.value > StepNumber.rotate_to_z.value
                            else 0.0
                        )

                        g.vec3.r, g.vec3.g, g.vec3.b = (
                            0.0 * (1.0 - ratio),
                            1.0 * (1.0 - ratio),
                            1.0 * ratio,
                        )
                        ms.rotate_x(ms.MatrixStack.model, math.radians(90.0 * ratio))
                        draw_vector(g.vec3, width, height)
                else:
                    draw_vector(g.vec3, width, height)
                glEnable(GL_DEPTH_TEST)

        glDisable(GL_DEPTH_TEST)

        draw_vector(g.vec1, width, height)
        draw_vector(g.vec2, width, height)

        imgui.render()
        impl.render(imgui.get_draw_data())

        # done with frame, flush and g.swap buffers
        # G.Swap front and back buffers
        glfw.swap_buffers(window)


glfw.terminate()
