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
import os
import sys
from dataclasses import dataclass
from enum import Enum, auto

# When using a pure python backend, prefer to import glfw before
# imgui_bundle (so that you end up using the standard glfw, not the
# one provided by imgui_bundle)
# (see https://github.com/pthom/imgui_bundle/issues/321)
import glfw
import numpy as np
import pyMatrixStack as ms
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
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

from _labels import LabelRenderer
from renderer import (
    Camera,
    Vector,
    compile_shader,
    do_draw_axis,
    do_draw_lines,
    do_draw_vector,
)

if not glfw.init():
    sys.exit()

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)


window = glfw.create_window(800, 800, "Cross Product Visualization", None, None)
if not window:
    glfw.terminate()
    sys.exit()


# Make the window's context current
glfw.make_context_current(window)
imgui.create_context()
impl = GlfwRenderer(window)

# TeX billboard labels rendered at runtime by texExpToPng (camera-facing
# quads).  No-ops gracefully when texExpToPng is not on PATH (i.e. when
# running outside the podman image).
labels = LabelRenderer(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class WindowState:
    """Backs the F11 / View->Fullscreen toggle: saves the windowed geometry so
    leaving fullscreen restores it (same shape as mvp's demos)."""

    fullscreen: bool = False
    saved_x: int = 0
    saved_y: int = 0
    saved_w: int = 800
    saved_h: int = 800


win_state = WindowState()


def toggle_fullscreen(window, state: WindowState) -> None:
    if state.fullscreen:
        glfw.set_window_monitor(
            window,
            None,
            state.saved_x,
            state.saved_y,
            state.saved_w,
            state.saved_h,
            0,
        )
        state.fullscreen = False
    else:
        state.saved_x, state.saved_y = glfw.get_window_pos(window)
        state.saved_w, state.saved_h = glfw.get_window_size(window)
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        glfw.set_window_monitor(
            window,
            monitor,
            0,
            0,
            mode.size.width,
            mode.size.height,
            mode.refresh_rate,
        )
        state.fullscreen = True


# Install a key handler


def on_key(window, key, scancode, action, mods):
    global advance_requested
    if action != glfw.PRESS:
        return
    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, 1)
        return
    # Don't let the shortcuts fire while an imgui field (e.g. the vector
    # inputs in the Vectors menu) is capturing keyboard input.
    if imgui.get_io().want_capture_keyboard:
        return
    if key == glfw.KEY_SPACE:
        advance_requested = True
    elif key == glfw.KEY_R:
        restart()
    elif key == glfw.KEY_P:
        g.auto_play = not g.auto_play
    elif key == glfw.KEY_F11:
        toggle_fullscreen(window, win_state)


glfw.set_key_callback(window, on_key)


def scroll_callback(window, xoffset, yoffset):
    global g
    # Don't zoom the camera while the cursor is over the menubar / a menu.
    if imgui.get_io().want_capture_mouse:
        return
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

# Set by the Animation menu's "Next: ..." item and the Space key; consumed by
# the step-transition processors below (and dropped at end of frame if the
# current step's animation hasn't finished -- matching the old UI, where the
# next button simply wasn't shown yet).
advance_requested = False


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
    # Don't orbit while the cursor is interacting with the menubar / a menu.
    if (
        glfw.PRESS == glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT)
        and not imgui.get_io().want_capture_mouse
    ):
        if previous_mouse_position:
            g.camera.rot_y -= 0.2 * math.radians(
                new_mouse_position[0] - previous_mouse_position[0]
            )
            g.camera.rot_x += 0.2 * math.radians(
                new_mouse_position[1] - previous_mouse_position[1]
            )
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
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RED,
        image.width,
        image.height,
        0,
        GL_RED,
        GL_UNSIGNED_BYTE,
        image_data,
    )
    glGenerateMipmap(GL_TEXTURE_2D)
    return texture_id


def current_animation_ratio() -> float:
    if g.step_number == StepNumber.beginning:
        return 0.0
    return min(
        1.0,
        (g.animation_time - g.current_animation_start_time) / g.seconds_per_operation,
    )


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


# ---------------------------------------------------------------------------
# Billboard label text, derived from proofs/crossproduct.tex.  The proof's
# notation: a' = f_a^{zx}(a) (after the z-rotation), a'' = (f_{a'}^x .
# f_a^{zx})(a) = [|a|, 0, 0] (after the y-rotation); b'' and b''' follow the
# same chain plus f_{b''}^{xy}; c = sqrt(b''_y^2 + b''_z^2) = |b|sin(theta);
# f(b) = (1/|a|)(a x b) after the inverse rotations; scaling by |a| gives
# a x b.  texExpToPng renders \documentclass{standalone} + amsmath ONLY, so
# use \lVert...\rVert (not commath's \norm).
# ---------------------------------------------------------------------------
def a_label() -> str:
    match g.step_number:
        case StepNumber.beginning:
            return r"\vec{a}"
        case StepNumber.rotate_z:
            return r"\vec{a}\,'"
        case StepNumber.undo_rotate_y:
            return r"\vec{a}\,'"
        case (
            StepNumber.rotate_y
            | StepNumber.rotate_x
            | StepNumber.show_triangle
            | StepNumber.project_onto_y
            | StepNumber.rotate_to_z
            | StepNumber.undo_rotate_x
        ):
            return r"\vec{a}\,''"
        case _:
            return r"\vec{a}"


def b_label() -> str:
    match g.step_number:
        case StepNumber.beginning:
            return r"\vec{b}"
        case StepNumber.rotate_z:
            return r"\vec{f}_a^{zx}(\vec{b})"
        case StepNumber.undo_rotate_y:
            return r"\vec{f}_a^{zx}(\vec{b})"
        case StepNumber.rotate_y | StepNumber.undo_rotate_x:
            return r"\vec{b}\,''"
        case (
            StepNumber.rotate_x
            | StepNumber.show_triangle
            | StepNumber.project_onto_y
            | StepNumber.rotate_to_z
        ):
            return r"\vec{b}\,'''"
        case _:
            return r"\vec{b}"


def c_label() -> str:
    match g.step_number:
        case StepNumber.show_triangle | StepNumber.project_onto_y:
            return r"c = \lVert\vec{b}\rVert\sin(\theta)"
        case StepNumber.rotate_to_z:
            return r"c"
        case (
            StepNumber.undo_rotate_x
            | StepNumber.undo_rotate_y
            | StepNumber.undo_rotate_z
        ):
            return r"\vec{f}(\vec{b})"
        case _:
            return r"\vec{a}\times\vec{b}"


# ---------------------------------------------------------------------------
# The step machine, menubar-driven (mvp's mathdemos pattern, sans the Cayley
# machinery).  Each StepNumber maps to the label of the action that advances
# past it; the transition side effects live in the two processors below, which
# run at the same points in the frame where the old in-window buttons ran
# (the Show Triangle transition reads the model matrix AFTER this frame's
# rotation blocks, so position matters).
# ---------------------------------------------------------------------------
STEP_NEXT_LABEL: dict = {
    StepNumber.beginning: "Rotate Z",
    StepNumber.rotate_z: "Rotate Y",
    StepNumber.rotate_y: "Rotate X",
    StepNumber.rotate_x: "Show Triangle",
    StepNumber.show_triangle: "Project onto yz plane",
    StepNumber.project_onto_y: "Rotate Y to Z, Z to -Y",
    StepNumber.rotate_to_z: "Undo Rotate X",
    StepNumber.undo_rotate_x: "Undo Rotate Y",
    StepNumber.undo_rotate_y: "Undo Rotate Z",
    StepNumber.undo_rotate_z: "Scale By Magnitude of first vector",
    StepNumber.scale_by_mag_a: "Show Plane spanned by vec a and vec b",
    StepNumber.show_plane: None,
}

# Which draw-relative-coordinates flag the current step exposes (the old UI
# showed the matching checkbox next to each rotation's button).
REL_FLAG: dict = {
    StepNumber.beginning: "draw_first_relative_coordinates",
    StepNumber.rotate_z: "draw_second_relative_coordinates",
    StepNumber.rotate_y: "draw_third_relative_coordinates",
    StepNumber.rotate_to_z: "draw_undo_rotate_x_relative_coordinates",
    StepNumber.undo_rotate_x: "draw_undo_rotate_y_relative_coordinates",
    StepNumber.undo_rotate_y: "draw_undo_rotate_z_relative_coordinates",
}


def can_advance() -> bool:
    if STEP_NEXT_LABEL[g.step_number] is None:
        return False
    if g.step_number == StepNumber.beginning:
        return True
    return current_animation_ratio() >= 0.999999


def process_pre_step_transitions() -> None:
    """Transitions out of beginning/rotate_z/rotate_y.  These ran BEFORE the
    frame's rotation blocks in the old in-window UI; keep that order."""
    global g, advance_requested
    if not (advance_requested or g.auto_play):
        return
    if g.step_number == StepNumber.beginning:
        advance_requested = False
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
                    (-a1 * a3 * b1) / (k1 * mag_a)
                    + (-a2 * a3 * b2) / (k1 * mag_a)
                    + (k1 * b3) / mag_a
                )

                angle = math.atan2(b_doubleprime_3, b_doubleprime_2)
            return angle if abs(angle) <= np.pi / 2.0 else (angle - 2 * np.pi)

        g.angle_x = calc_angle_x()
    elif g.step_number == StepNumber.rotate_z and current_animation_ratio() >= 0.999999:
        advance_requested = False
        g.do_second_rotate = True
        g.step_number = StepNumber.rotate_y
        g.current_animation_start_time = g.animation_time
    elif g.step_number == StepNumber.rotate_y and current_animation_ratio() >= 0.999999:
        advance_requested = False
        g.do_third_rotate = True
        g.step_number = StepNumber.rotate_x
        g.current_animation_start_time = g.animation_time


def process_post_step_transitions() -> None:
    """Transitions from rotate_x onward.  These ran AFTER the frame's rotation
    blocks in the old in-window UI -- the Show Triangle transition reads the
    model matrix with this frame's rotations applied."""
    global g, advance_requested
    if not (advance_requested or g.auto_play):
        return
    if g.step_number != StepNumber.beginning and current_animation_ratio() < 0.999999:
        return
    if g.step_number == StepNumber.rotate_x:
        advance_requested = False
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
    elif g.step_number == StepNumber.show_triangle:
        advance_requested = False
        g.project_onto_yz_plane = True
        g.step_number = StepNumber.project_onto_y
        g.current_animation_start_time = g.animation_time
    elif g.step_number == StepNumber.project_onto_y:
        advance_requested = False
        g.rotate_yz_90 = True
        g.step_number = StepNumber.rotate_to_z
        g.current_animation_start_time = g.animation_time
    elif g.step_number == StepNumber.rotate_to_z:
        advance_requested = False
        g.undo_rotate_x = True
        g.step_number = StepNumber.undo_rotate_x
        g.current_animation_start_time = g.animation_time
    elif g.step_number == StepNumber.undo_rotate_x:
        advance_requested = False
        g.undo_rotate_y = True
        g.step_number = StepNumber.undo_rotate_y
        g.current_animation_start_time = g.animation_time
    elif g.step_number == StepNumber.undo_rotate_y:
        advance_requested = False
        g.undo_rotate_z = True
        g.step_number = StepNumber.undo_rotate_z
        g.current_animation_start_time = g.animation_time
    elif g.step_number == StepNumber.undo_rotate_z:
        advance_requested = False
        g.do_scale = True
        g.step_number = StepNumber.scale_by_mag_a
    elif g.step_number == StepNumber.scale_by_mag_a:
        advance_requested = False
        g.do_remove_ground = True
        g.step_number = StepNumber.show_plane


# ---------------------------------------------------------------------------
# Menubar (mvp's mathdemos pattern): all controls live in the main menu bar;
# there is no floating window.
# ---------------------------------------------------------------------------
def menu_action(label, key, action, *, selected=False) -> None:
    """A menubar item that also shows its keyboard shortcut (``key``, in the
    right-hand column) and an optional check mark (``selected``).  Runs
    ``action()`` once on click.  Call inside a ``begin_menu`` block."""
    clicked, _ = imgui.menu_item(label, key, selected, True)
    if clicked:
        action()


def restart_derivation() -> None:
    # restart() rebuilds Globals (which resets swap); the vectors and camera
    # are preserved by restart itself, so only swap needs saving.
    sw = g.swap
    restart()
    g.swap = sw


def swap_vectors() -> None:
    g.swap = not g.swap
    initiliaze_vecs()
    restart_derivation()


def view_down(rot_x, rot_y) -> None:
    g.camera.rot_x = rot_x
    g.camera.rot_y = rot_y
    g.use_ortho = True


def toggle(attr) -> None:
    setattr(g, attr, not getattr(g, attr))


def menubar() -> None:
    global advance_requested
    if not imgui.begin_main_menu_bar():
        return
    if imgui.begin_menu("File", True):
        menu_action("Quit", "Esc", lambda: glfw.set_window_should_close(window, 1))
        imgui.end_menu()
    if imgui.begin_menu("Animation", True):
        nxt = STEP_NEXT_LABEL[g.step_number]
        if nxt is None:
            imgui.menu_item("(final step)", "", False, False)
        else:
            clicked, _ = imgui.menu_item("Next: " + nxt, "Space", False, can_advance())
            if clicked:
                advance_requested = True
        menu_action("Restart", "R", restart)
        menu_action("AutoPlay", "P", lambda: toggle("auto_play"), selected=g.auto_play)
        _, g.seconds_per_operation = imgui.slider_float(
            "Seconds / step", g.seconds_per_operation, 0.25, 5.0
        )
        flag = REL_FLAG.get(g.step_number)
        if flag is not None:
            menu_action(
                "Draw Relative Coordinates",
                "",
                lambda f=flag: toggle(f),
                selected=getattr(g, flag),
            )
        imgui.end_menu()
    if imgui.begin_menu("Camera", True):
        menu_action(
            "Auto Rotate Camera",
            "",
            lambda: toggle("auto_rotate_camera"),
            selected=g.auto_rotate_camera,
        )
        if not g.use_ortho:
            _, g.camera.r = imgui.slider_float("Camera Radius", g.camera.r, 3.0, 130.0)
        menu_action("View Down x Axis", "", lambda: view_down(0.0, math.pi / 2.0))
        menu_action("View Down Negative y Axis", "", lambda: view_down(0.0, 0.0))
        menu_action("View Down z Axis", "", lambda: view_down(math.pi / 2.0, 0.0))
        imgui.end_menu()
    if imgui.begin_menu("Vectors", True):
        ca, (g.vec1.x, g.vec1.y, g.vec1.z) = imgui.input_float3(
            "vec a" + a_extra_text(),
            [g.vec1.x, g.vec1.y, g.vec1.z],
        )
        cb, (g.vec2.x, g.vec2.y, g.vec2.z) = imgui.input_float3(
            "vec b" + a_extra_text(),
            [g.vec2.x, g.vec2.y, g.vec2.z],
        )
        if ca or cb:  # editing the inputs restarts the derivation
            restart_derivation()
        menu_action("Swap vectors", "", swap_vectors)
        menu_action(
            "Highlight a" + a_extra_text(),
            "",
            lambda: (
                setattr(g.vec1, "highlight", not g.vec1.highlight),
                setattr(g.vec2, "highlight", False),
            ),
            selected=g.vec1.highlight,
        )
        menu_action(
            "Highlight b" + a_extra_text(),
            "",
            lambda: (
                setattr(g.vec2, "highlight", not g.vec2.highlight),
                setattr(g.vec1, "highlight", False),
            ),
            selected=g.vec2.highlight,
        )
        imgui.end_menu()
    if imgui.begin_menu("Highlight", True):
        for name, attr in (
            ("x", "highlight_x"),
            ("y", "highlight_y"),
            ("z", "highlight_z"),
            ("x'", "highlight_relative_x"),
            ("y'", "highlight_relative_y"),
            ("z'", "highlight_relative_z"),
        ):
            menu_action(name, "", lambda a=attr: toggle(a), selected=getattr(g, attr))
        imgui.end_menu()
    if imgui.begin_menu("View", True):
        menu_action(
            "Fullscreen",
            "F11",
            lambda: toggle_fullscreen(window, win_state),
            selected=win_state.fullscreen,
        )
        menu_action(
            "Draw Coordinate System of Natural Basis",
            "",
            lambda: toggle("draw_coordinate_system_of_natural_basis"),
            selected=g.draw_coordinate_system_of_natural_basis,
        )
        imgui.end_menu()
    imgui.end_main_menu_bar()


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
        do_draw_lines(
            lines_shader, unit_circle_vertices(), time, width, height, xy, yz, zx
        )

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
        while (
            glfw.get_time()
            < g.time_at_beginning_of_previous_frame + 1.0 / TARGET_FRAMERATE
        ):
            pass
        # set for comparison on the next frame
        g.time_at_beginning_of_previous_frame = glfw.get_time()

        if not g.animation_paused:
            g.animation_time += 1.0 / 60.0 * g.animation_time_multiplier

        if g.auto_rotate_camera:
            g.camera.rot_y += math.radians(0.1)

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

        # the math-coordinate frame, for placing the axis labels
        coords_M = np.array(ms.get_current_matrix(ms.MatrixStack.model))

        if g.draw_coordinate_system_of_natural_basis:
            if not g.do_remove_ground:
                draw_ground(g.animation_time, width, height)
                draw_ground(g.animation_time, width, height, xy=False, zx=True)
                draw_unit_circle(
                    g.animation_time, width, height, xy=True, yz=True, zx=True
                )

        draw_axis(
            width,
            height,
            highlight_x=g.highlight_x,
            highlight_y=g.highlight_y,
            highlight_z=g.highlight_z,
        )

        imgui.new_frame()
        menubar()

        # Step transitions whose side effects must happen BEFORE this frame's
        # rotation blocks (mirrors where the old in-window buttons ran).
        process_pre_step_transitions()

        if g.do_third_rotate:
            ratio = (
                current_animation_ratio()
                if g.step_number == StepNumber.rotate_x
                else 1.0
            )
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
                ratio = (
                    current_animation_ratio()
                    if g.step_number == StepNumber.show_triangle.value
                    else 1.0
                )
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
            ratio = (
                current_animation_ratio()
                if g.step_number == StepNumber.rotate_y
                else 1.0
            )
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
                ratio = (
                    current_animation_ratio()
                    if g.step_number == StepNumber.rotate_x
                    else 1.0
                )
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
            ratio = (
                current_animation_ratio()
                if g.step_number == StepNumber.rotate_z
                else 1.0
            )
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
                ratio = (
                    current_animation_ratio()
                    if g.step_number == StepNumber.rotate_y
                    else 1.0
                )
                ms.rotate_z(ms.MatrixStack.model, g.vec1.angle_z * ratio)
                draw_ground(g.animation_time, width, height)
                draw_axis(
                    width,
                    height,
                    highlight_x=g.highlight_relative_x,
                    highlight_y=g.highlight_relative_y,
                    highlight_z=g.highlight_relative_z,
                )

        # Step transitions that read the post-rotation model matrix (mirrors
        # where the old in-window buttons ran, AFTER the rotation blocks).
        process_post_step_transitions()
        # A pending advance that couldn't fire this frame is dropped, matching
        # the old UI (where the next button simply wasn't shown yet).
        advance_requested = False

        vec3_label_M = None
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
                        vec3_label_M = np.array(
                            ms.get_current_matrix(ms.MatrixStack.model)
                        )
                        draw_vector(g.vec3, width, height)
                else:
                    vec3_label_M = np.array(ms.get_current_matrix(ms.MatrixStack.model))
                    draw_vector(g.vec3, width, height)
                glEnable(GL_DEPTH_TEST)

        glDisable(GL_DEPTH_TEST)

        # the frame the vectors are drawn in, for placing their labels
        model_M = np.array(ms.get_current_matrix(ms.MatrixStack.model))
        draw_vector(g.vec1, width, height)
        draw_vector(g.vec2, width, height)

        # --- TeX billboard labels (no-op when texExpToPng is unavailable) ---
        # Drawn last, over the scene, at the same frames the vectors use.
        if labels.available:
            labels.begin(
                np.array(ms.get_current_matrix(ms.MatrixStack.view)),
                np.array(ms.get_current_matrix(ms.MatrixStack.projection)),
                (width, height),
            )
            if g.draw_coordinate_system_of_natural_basis and not g.do_remove_ground:
                for axis_v, tex in (
                    (np.array([1.18, 0.0, 0.0, 1.0]), "x"),
                    (np.array([0.0, 1.18, 0.0, 1.0]), "y"),
                    (np.array([0.0, 0.0, 1.18, 1.0]), "z"),
                ):
                    labels.draw(tex, (coords_M @ axis_v)[:3])
            a_tip = np.array([g.vec1.x * 1.08, g.vec1.y * 1.08, g.vec1.z * 1.08, 1.0])
            labels.draw(a_label(), (model_M @ a_tip)[:3])
            b_tip = np.array([g.vec2.x * 1.08, g.vec2.y * 1.08, g.vec2.z * 1.08, 1.0])
            labels.draw(b_label(), (model_M @ b_tip)[:3])
            if g.vec3 and vec3_label_M is not None:
                c_tip = np.array(
                    [g.vec3.x * 1.08, g.vec3.y * 1.08, g.vec3.z * 1.08, 1.0]
                )
                labels.draw(c_label(), (vec3_label_M @ c_tip)[:3])
            labels.end()

        imgui.render()
        impl.render(imgui.get_draw_data())

        # done with frame, flush and g.swap buffers
        # G.Swap front and back buffers
        glfw.swap_buffers(window)


labels.cleanup()
glfw.terminate()
