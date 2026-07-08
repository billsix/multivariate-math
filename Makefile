.DEFAULT_GOAL := help

USE_SPYDER ?= 0
USE_EMACS ?= 0

CONTAINER_CMD = podman
CONTAINER_NAME = multivariate-math

TMUX_FILE := $(HOME)/.tmux.conf
TMUX_REAL_PATH := $(shell readlink -f $(TMUX_FILE))
TMUX_MOUNT := $(shell if [ -f $(TMUX_REAL_PATH) ]; then echo "-v $(TMUX_REAL_PATH):/root/.tmux.conf:Z" ; fi)

GITCONFIG_FILE := $(HOME)/.gitconfig
GITCONFIG_REAL_PATH := $(shell readlink -f $(GITCONFIG_FILE))
GITCONFIG_MOUNT := $(shell if [ -f $(GITCONFIG_REAL_PATH) ]; then echo "-v $(GITCONFIG_REAL_PATH):/root/.gitconfig:Z" ; fi)

GNUPG_FILE := $(HOME)/.gnupg
GNUPG_REAL_PATH := $(shell readlink -f $(GNUPG_FILE))
GNUPG_MOUNT := $(shell if [ -d $(GNUPG_REAL_PATH) ]; then echo "-v $(GNUPG_REAL_PATH):/root/.gnupg:Z" ; fi)


FILES_TO_MOUNT = -v $(shell pwd):/mvm/:Z \
		-v ./entrypoint/entrypoint.sh:/entrypoint.sh:Z \
		-v ./entrypoint/jupyter.sh:/usr/local/bin/jupyter.sh:Z \
		-v ./entrypoint/spyder.sh:/usr/local/bin/spyder.sh:Z \
		-v ./entrypoint/format.sh:/format.sh:Z \
		-v ./output/:/output/:Z \
                $(TMUX_MOUNT) \
                $(GNUPG_MOUNT) \
                $(GITCONFIG_MOUNT)

EXPOSE_PORT = -p 8888:8888

X_FLAGS_FOR_CONTAINER = -e DISPLAY=$(DISPLAY) \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	--security-opt label=type:container_runtime_t

# GPU render node for hardware GL (Wayland EGL / X11); skipped if /dev/dri is absent.
DRI_DEVICE := $(shell [ -d /dev/dri ] && echo "--device /dev/dri")

# Wayland for the GUI demos, WITHOUT breaking imgui_bundle (same setup as mvp).
# The demos' `import glfw` (PyPI) loads first, then imgui_bundle's native lib --
# and BOTH bind the same soname `libglfw.so.3`.  PyPI glfw's *Wayland-only*
# build lacks `glfwGetX11Window` (which imgui_bundle needs), so forcing that
# variant crashes imgui_bundle.  Instead point PyPI glfw at the SYSTEM Fedora
# libglfw (a DUAL X11+Wayland build, from the `glfw` rpm); it loads first, so
# imgui_bundle binds the same dual lib and GLFW picks Wayland at runtime via
# WAYLAND_DISPLAY.
#
# PYOPENGL_PLATFORM=egl: under Wayland, GLFW creates an EGL context, and
# PyOpenGL's GLX default can't see it ("Attempt to retrieve context when no
# valid context").  Fedora's python3-pyopengl (which the image installs) is
# patched to select EGL automatically when XDG_SESSION_TYPE=wayland, so this
# variable is redundant here -- kept because it documents the requirement and
# covers a non-Fedora PyOpenGL sneaking into the venv.
WAYLAND_FLAGS_FOR_CONTAINER = -e "XDG_SESSION_TYPE=wayland" \
                              -e "WAYLAND_DISPLAY=${WAYLAND_DISPLAY}" \
                              -e "XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR}" \
                              -e "PYGLFW_LIBRARY=/usr/lib64/libglfw.so.3" \
                              -e "PYOPENGL_PLATFORM=egl" \
                              -v "${XDG_RUNTIME_DIR}:${XDG_RUNTIME_DIR}" \
                              $(DRI_DEVICE)


.PHONY: all
all: image shell ## Build the image and go into the shell

.PHONY: image
image: ## Build the OCI image
	$(CONTAINER_CMD) build -t $(CONTAINER_NAME) \
                         --build-arg USE_SPYDER=$(USE_SPYDER) \
                         --build-arg USE_EMACS=$(USE_EMACS) \
                         .

.PHONY: clean
clean: ## Delete the output directory, cleaning out the built PDFs
	rm -rf output/*

.PHONY: shell
shell: ## Get Shell into a ephermeral container made from the image
	$(CONTAINER_CMD) run -it --rm \
		--entrypoint /bin/bash \
		$(FILES_TO_MOUNT) \
		-v ./entrypoint/shell.sh:/shell.sh:Z \
		$(X_FLAGS_FOR_CONTAINER) \
		$(WAYLAND_FLAGS_FOR_CONTAINER) \
		$(EXPOSE_PORT) \
		$(CONTAINER_NAME) \
		/shell.sh

.PHONY: jupyter
jupyter: image ## Launch JupyterLab (mvm kernel) on http://127.0.0.1:8888/lab
	$(CONTAINER_CMD) run -it --rm \
		--entrypoint /bin/bash \
		$(FILES_TO_MOUNT) \
		$(X_FLAGS_FOR_CONTAINER) \
		$(WAYLAND_FLAGS_FOR_CONTAINER) \
		$(EXPOSE_PORT) \
		$(CONTAINER_NAME) \
		/usr/local/bin/jupyter.sh

# Run ruff + ty over the source INSIDE the container (the image's pinned
# toolchain).  The editable install overlays the image's baked-in copy of the
# package with the live bind-mounted tree first, so ty checks what's on disk.
.PHONY: format
format: image ## (container) ruff + ty over the source (entrypoint/format.sh)
	$(CONTAINER_CMD) run --rm \
		--entrypoint /bin/bash \
		$(FILES_TO_MOUNT) \
		$(CONTAINER_NAME) \
		-c 'set -e; source /venv/bin/activate; cd /mvm; \
		    uv pip install --python $$(which python) --no-deps --no-index --no-build-isolation -e .; \
		    bash /format.sh'

.PHONY: pdfs
pdfs: image ## (container) build the proofs (proofs/*.tex) into PDFs in ./output
	$(CONTAINER_CMD) run --rm \
		--entrypoint /bin/bash \
		$(FILES_TO_MOUNT) \
		-v ./entrypoint/pdfs.sh:/pdfs.sh:Z \
		$(CONTAINER_NAME) \
		/pdfs.sh

.PHONY: image-export
image-export: ## export the OCI image to a timestamped tar in the repo root
	$(CONTAINER_CMD) save $(CONTAINER_NAME) -o $(CONTAINER_NAME)-$(shell date +%m-%d-%Y_%H-%M-%S).tar

.PHONY: image-import
image-import: ## import an OCI image tar: make image-import FILE=foo.tar
	$(CONTAINER_CMD) load -i $(FILE)

.PHONY: help
help:
	@grep --extended-regexp '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
