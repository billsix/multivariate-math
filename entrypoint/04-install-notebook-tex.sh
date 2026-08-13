#!/usr/bin/env bash
#
# 04-install-notebook-tex.sh -- pandoc + the XeLaTeX toolchain that nbconvert's
# "Export to PDF" path renders through, plus this repo's own TeX consumers (the
# pdflatex proofs and texExpToPng). Always installed (the Dockerfile calls it
# unconditionally); kept as its own group because it is a distinct, documented set
# separate from the base toolchain. No options -- see 01-install-base.sh for the design.
#
# Single dnf call, so its own exit status is this script's exit status.
set -uo pipefail

dnf install -y \
    pandoc \
    texlive-xetex \
    texlive-collection-fontsrecommended \
    texlive-collection-latexrecommended \
    texlive-adjustbox \
    texlive-tcolorbox \
    texlive-collectbox \
    texlive-ucs \
    texlive-titling \
    texlive-enumitem \
    texlive-rsfs \
    texlive-jknapltx \
    texlive-upquote \
    texlive-ulem \
    texlive-soul \
    texlive-eurosym \
    texlive-pgf \
    texlive-environ \
    texlive-trimspaces \
    texlive-parskip \
    texlive-anyfontsize \
    texlive-commath \
    texlive-dvipng \
    texlive-standalone
