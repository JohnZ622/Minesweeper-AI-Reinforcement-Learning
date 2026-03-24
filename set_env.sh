#! /bin/bash
export LD_LIBRARY_PATH=$(find .venv -type d -path "*/nvidia/*/lib" | tr '\n' ':'):$LD_LIBRARY_PATH
