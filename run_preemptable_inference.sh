#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err

# Self-submitting dispatcher + worker for inference.py --mode latent or
# --mode steering (merged from the former run_preemptable_inference_latent.sh
# / _steering.sh, which were identical apart from which mode they ran). Meant
# to be called by run_preemptable.sh, which owns CFG/DUMP/NPROC/NUM_CHUNKS and
# passes them in -- this script just processes them: one invocation submits
# NUM_CHUNKS separate sbatch jobs, each scoring/steering its own disjoint
# slice of concepts in parallel (mirrors run_preemptable_evaluate_local.sh).
# NPROC is torchrun's --nproc_per_node within each chunk's job, not the chunk
# count. Steering mode requires latent mode's chunks to already be merged
# (max_act must be populated) before it runs.
#
# Usage:
#   bash run_preemptable_inference.sh <latent|steering> <CFG> <DUMP> <NPROC> <NUM_CHUNKS>
# NUM_CHUNKS<=1 skips chunking and submits a single unchunked job instead.
# Once every chunk job's output is on disk, the last one to finish
# auto-merges them into {mode}_data.parquet itself (see
# maybe_auto_merge_chunks in inference.py) -- no separate merge step needed.
#
# --overwrite_inference_data_dir (latent mode only) makes this read
# generate.py's own latent_eval_data.parquet instead of silently regenerating
# eval data live with the old response+polysemantic-hard-negative schema.
#
# Dispatch-vs-worker is decided by whether a 6th (chunk index) arg is present,
# not by SLURM_JOB_ID -- so this is safe to call with plain `bash`, either
# from the login node or nested inside another job's own worker payload
# (e.g. run_preemptable.sh calling this from its own submitted job).

set -e

cd ~/axbench

MODE="$1"; CFG="$2"; DUMP="$3"; NPROC="$4"; NUM_CHUNKS="${5:-1}"

if [ "$MODE" != "latent" ] && [ "$MODE" != "steering" ]; then
  echo "Usage: bash run_preemptable_inference.sh <latent|steering> <CFG> <DUMP> <NPROC> <NUM_CHUNKS>" >&2
  exit 1
fi

if [ -z "${6:-}" ]; then
  # Hardcoded, not derived from ${BASH_SOURCE[0]} -- when this script is
  # invoked nested inside another job's already-running worker payload
  # (rather than typed directly on the login node), BASH_SOURCE/dirname
  # resolves to Slurm's job spool directory, not the real repo path, and the
  # sbatch calls below would try to submit a file that doesn't exist there.
  SELF=~/axbench/run_preemptable_inference.sh
  if [ -f ~/axbench/.env ]; then
    set -a
    source ~/axbench/.env
    set +a
  fi

  if [ "$NUM_CHUNKS" -le 1 ]; then
    sbatch "$SELF" "$MODE" "$CFG" "$DUMP" "$NPROC" 1 0
    exit 0
  fi

  for ((i = 0; i < NUM_CHUNKS; i++)); do
    sbatch "$SELF" "$MODE" "$CFG" "$DUMP" "$NPROC" "$NUM_CHUNKS" "$i"
  done
  exit 0
fi

# --- Worker payload (runs only once submitted by sbatch, above) ---

CHUNK="$6"

CHUNK_ARGS=()
if [ "$NUM_CHUNKS" -gt 1 ]; then
  CHUNK_ARGS=(--chunk "$CHUNK" --num_chunks "$NUM_CHUNKS")
fi

MODE_ARGS=()
if [ "$MODE" = "latent" ]; then
  MODE_ARGS=(--overwrite_inference_data_dir "$DUMP/inference")
fi

# --no-sync: NUM_CHUNKS concurrent chunk jobs share one .venv; without this,
# `uv run`'s per-invocation sync races across processes (stale NFS handles,
# partial installs). Run `uv sync` once, serially, before submitting any chunks.
uv run --no-sync torchrun --nproc_per_node="$NPROC" --master_port=$((29500 + CHUNK)) axbench/scripts/inference.py \
  --config "$CFG" \
  --dump_dir "$DUMP" \
  --mode "$MODE" \
  "${MODE_ARGS[@]}" \
  "${CHUNK_ARGS[@]}"
