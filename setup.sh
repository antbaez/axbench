#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

uv sync

uv run axbench/data/download-seed-sentences.py

cd axbench/data
bash download-2b.sh
bash download-9b.sh
bash download-alpaca.sh

cd ../..
uv run download_from_hf.py
