#!/bin/env bash

# Activate the venv so ty/ruff resolve deps from it (the venv is
# --system-site-packages, so it also sees the dnf-installed base packages).
export VIRTUAL_ENV_DISABLE_PROMPT=1
source /venv/bin/activate

cd /mvm/ || exit 1

# Every step runs (one pass reports all the red), but the script exits
# nonzero if ANY step failed -- otherwise the exit code is the LAST
# command's alone, and a `ruff check` failure is silently masked by a clean
# `ty check` (the flaw that hid ty errors behind a green format gate in
# gacalc, found 2026-07-29).
status=0
ruff check . --fix || status=1
ruff format --line-length=88 || status=1

ty check || status=1
exit $status
