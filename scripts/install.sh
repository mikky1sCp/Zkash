#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/Scripts/activate
pip install -r requirements.txt
pip install -e .
echo "done — run: bash activate.sh"