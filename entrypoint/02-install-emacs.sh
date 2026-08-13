#!/usr/bin/env bash
#
# 02-install-emacs.sh -- Emacs. Corresponds to the Dockerfile's USE_EMACS flag; the
# Dockerfile runs this only when USE_EMACS=1. No options -- see 01-install-base.sh.
#
# Single dnf call, so its own exit status is this script's exit status.
set -uo pipefail

dnf install -y emacs
