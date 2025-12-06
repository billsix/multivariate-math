#!/bin/env bash

cd /mvm/

ruff check . --fix
ruff format --line-length=88

ty check
