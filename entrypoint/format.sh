#!/bin/env bash

# Activate the venv so ty/ruff resolve deps from it (the venv is
# --system-site-packages, so it also sees the dnf-installed base packages).
export VIRTUAL_ENV_DISABLE_PROMPT=1
source /venv/bin/activate

cd /mvm/

ruff check . --fix
ruff format --line-length=88

ty check
