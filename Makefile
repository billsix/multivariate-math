.DEFAULT_GOAL := help

PODMAN_CMD = podman
CONTAINER_NAME = multivariate-math
FILES_TO_MOUNT = -v ./entrypoint/entrypoint.sh:/entrypoint.sh:Z \
		-v ./pyproject.toml:/mvm/pyproject.toml:Z \
		-v ./setup.py:/mvm/setup.py:Z \
		-v ./src:/mvm/src/:Z \
		-v ./proofs:/mvm/proofs/:Z \
		-v ./output/:/output/:Z

USE_X = -e DISPLAY=$(DISPLAY) \
	-v /tmp/.X11-unix:/tmp/.X11-unix

.PHONY: all
all: clean image ## Build the image

.PHONY: image
image: image  ## Build a podman image in which to run the demos
	$(PODMAN_CMD) build -t $(CONTAINER_NAME) -f Dockerfile

.PHONY: clean
clean: ## Delete the output directory, cleaning out the HTML and the PDF
	rm -rf output/*

.PHONY: shell
shell: image ## Run shell
	$(PODMAN_CMD) run -it --rm \
		$(FILES_TO_MOUNT) \
		-v ./entrypoint/crossproduct.sh:/crossproduct.sh:Z \
		$(USE_X) \
		$(CONTAINER_NAME) 


.PHONY: crossproduct
crossproduct: image ## Run Crossproduct
	$(PODMAN_CMD) run -it --rm \
		--entrypoint /bin/bash \
		$(FILES_TO_MOUNT) \
		-v ./entrypoint/crossproduct.sh:/crossproduct.sh:Z \
		$(USE_X) \
		$(CONTAINER_NAME) \
		/crossproduct.sh



.PHONY: pdfs
pdfs: image ## Run Crossproduct
	$(PODMAN_CMD) run -it --rm \
		--entrypoint /bin/bash \
		$(FILES_TO_MOUNT) \
		-v ./entrypoint/pdfs.sh:/pdfs.sh:Z \
		$(USE_X) \
		$(CONTAINER_NAME) \
		/pdfs.sh



.PHONY: help
help:
	@grep --extended-regexp '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
