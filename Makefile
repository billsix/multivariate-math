.DEFAULT_GOAL := help

CONTAINER_CMD = podman
CONTAINER_NAME = multivariate-math


FILES_TO_MOUNT = -v $(shell pwd):/mvm/:Z \
		-v ./entrypoint/entrypoint.sh:/entrypoint.sh:Z \
		-v ./entrypoint/jupyter.sh:/usr/local/bin/jupyter.sh:Z \
		-v ./entrypoint/spyder.sh:/usr/local/bin/spyder.sh:Z \
		-v ./entrypoint/format.sh:/format.sh:Z \
		-v ./entrypoint/.bashrc:/root/.bashrc:Z \
		-v ./output/:/output/:Z

X_FLAGS_FOR_CONTAINER = -e DISPLAY=$(DISPLAY) \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	--security-opt label=type:container_runtime_t

WAYLAND_FLAGS_FOR_CONTAINER = -e "WAYLAND_DISPLAY=${WAYLAND_DISPLAY}" \
                              -e "XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR}" \
                              -v "${XDG_RUNTIME_DIR}:${XDG_RUNTIME_DIR}"

EXPOSE_PORT = -p 8888:8888


.PHONY: all
all: clean image ## Build the image

.PHONY: image
image: image  ## Build a podman image in which to run the demos
	$(CONTAINER_CMD) build -t $(CONTAINER_NAME) -f Dockerfile

.PHONY: clean
clean: ## Delete the output directory, cleaning out the HTML and the PDF
	rm -rf output/*


.PHONY: shell
shell:  ## Get Shell into a ephermeral container made from the image
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


.PHONY: format
format: image ## Format the Python source with ruff + ty (entrypoint/format.sh)
	$(CONTAINER_CMD) run -it --rm \
		--entrypoint /bin/bash \
		$(FILES_TO_MOUNT) \
		$(CONTAINER_NAME) \
		-c 'export VIRTUAL_ENV_DISABLE_PROMPT=1; \
		    source /venv/bin/activate; \
		    cd /mvm; python3 -m pip install -e .; \
		    bash /format.sh'



.PHONY: pdfs
pdfs: image ## Run Crossproduct
	$(CONTAINER_CMD) run -it --rm \
		--entrypoint /bin/bash \
		$(FILES_TO_MOUNT) \
		-v ./entrypoint/pdfs.sh:/pdfs.sh:Z \
		$(USE_X) \
		$(CONTAINER_NAME) \
		/pdfs.sh



.PHONY: help
help:
	@grep --extended-regexp '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
