#!/usr/bin/env bash
#
# 03-install-spyder.sh -- the Spyder IDE plus the Mesa GL bits it wants. Corresponds to
# the Dockerfile's USE_SPYDER flag; the Dockerfile runs this only when USE_SPYDER=1. No
# options -- see 01-install-base.sh for the design.
#
# Two dnf calls (matching the original), so accumulate a non-zero exit if either fails.
set -uo pipefail

status=0
dnf install -y mesa-dri-drivers mesa-libGLU-devel || status=1
dnf install -y python3-spyder || status=1
exit $status
