#!/bin/bash

set -e

TYPE=${1:-DOSY}

export TYPE

python main.py
