#!/bin/bash

# Runs generate.py --mode training, then --mode latent, locally in the calling
# shell -- no sbatch, no Slurm job. Meant to be called by run_preemptable.sh,
# which owns CFG/DUMP/NUM_WORKERS/MAX_CONCURRENT_REQUESTS and passes them in.
#
# Usage:
#   bash run_preemptable_generate.sh <CFG> <DUMP> <NUM_WORKERS> <MAX_CONCURRENT_REQUESTS>
#
# These two modes are OpenAI-API + CPU only -- generate.py only moves a model
# onto CUDA in --mode dpo_training (generate.py:748-756) -- so they need no GPU
# allocation. This blocks until both finish, so run it inside tmux/screen (or a
# `cpu` salloc) if the session might drop.
#
# `set -e` is what orders the two: latent only starts if training exited 0.
# That matters because generate_latent reads the concept metadata under
# $DUMP/generate that training mode writes (generate.py:353). Both modes resume
# from their own state files, so re-running after an interruption skips the
# concepts already done rather than redoing them.

set -e

cd ~/axbench

CFG="$1"; DUMP="$2"; NUM_WORKERS="${3:-50}"; MAX_CONCURRENT_REQUESTS="${4:-1000}"

if [ -z "$CFG" ] || [ -z "$DUMP" ]; then
  echo "Usage: bash run_preemptable_generate.sh <CFG> <DUMP> <NUM_WORKERS> <MAX_CONCURRENT_REQUESTS>" >&2
  exit 1
fi

# Sourced here too, not just in run_preemptable.sh, so this stays runnable on
# its own -- generate.py needs OPENAI_API_KEY for the synthesis calls.
if [ -f ~/axbench/.env ]; then
  set -a
  source ~/axbench/.env
  set +a
fi

# Training data: concept-injected positive examples + contrastive minimal-edit negatives, per concept.
echo "=== generate.py --mode training ==="
uv run axbench/scripts/generate.py \
  --config "$CFG" \
  --mode training \
  --dump_dir "$DUMP" \
  --num_workers "$NUM_WORKERS" \
  --max_concurrent_requests "$MAX_CONCURRENT_REQUESTS"

# Latent eval data: held-out data used for concept-detection scoring in latent mode.
echo "=== generate.py --mode latent ==="
uv run axbench/scripts/generate.py \
  --config "$CFG" \
  --mode latent \
  --dump_dir "$DUMP" \
  --num_workers "$NUM_WORKERS" \
  --max_concurrent_requests "$MAX_CONCURRENT_REQUESTS"
