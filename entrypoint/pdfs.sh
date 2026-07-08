#!/bin/env bash
# Build the proofs (proofs/*.tex) with pdflatex and copy the PDFs to the
# bind-mounted /output so they land in ./output on the host.
set -e
cd /mvm/proofs
make all
mkdir -p /output
cp *.pdf /output/
