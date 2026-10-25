#!/usr/bin/env bash
# Rebuild everything from week 1: download -> clean -> split -> EDA -> tests
# Needs uv: https://docs.astral.sh/uv/getting-started/installation/
set -e
cd "$(dirname "$0")"
uv sync
uv run python -m src.data.download
uv run python -m src.data.clean
uv run python -m src.data.split
uv run python -m src.eda
uv run pytest -q
