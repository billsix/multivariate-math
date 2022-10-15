#!/bin/env bash

for f in `find ./src/ -name "*.py"`; do autoflake8 --in-place $f; done
for f in `find ./src/ -name "*.py"`; do black -l 120 $f; done
