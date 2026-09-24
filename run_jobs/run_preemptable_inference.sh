#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err

# Self-submitting dispatcher + worker for inference.py --mode latent and
# --mode steering. Meant to be called by run_preemptable.sh, which owns
# CFG/DUMP/NPROC/NUM_CHUNKS and passes them in -- this script just processes
# them.
#
# Usage:
#   bash run_jobs/run_preemptable_inference.sh <CFG> <DUMP> <NPROC> <NUM_CHUNKS>
#
# One invocation submits both modes: NUM_CHUNKS latent jobs (each scoring its
# own disjoint slice of concepts in parallel), then NUM_CHUNKS steering jobs
# that all carry --dependency=afterok on *every* latent job. So no steering
# chunk starts until every latent chunk has succeeded.
#
# That barrier is load-bearing, not tidiness. Steering reads each concept's
# max_act out of the *merged* latent_data.parquet, and the merge only happens
# once the last latent chunk lands (maybe_auto_merge_chunks in inference.py).
# Start a steering chunk early and pre_compute_mean_activations finds nothing,
# so predict_steer silently falls back to a factor scale of 1.0 -- quietly
# changing what every entry in the steering_factors sweep means, with no error.
# As a second guard, each steering worker first runs the CPU-only
# `--mode latent --verify_chunks` (inference.py:1358-1366, returns before the
# process group or any GPU work), which hard-fails if a concept or a model's
# max_act column is missing.
#
# NPROC is torchrun's --nproc_per_node within each chunk's job, not the chunk
# count. NUM_CHUNKS<=1 skips chunking and submits one unchunked job per mode.
# Once every chunk job's output is on disk, the last one to finish auto-merges
# them into {mode}_data.parquet itself -- no separate merge step needed. The
# merge is verified before it is written, then the chunk files are deleted.
#
# --overwrite_inference_data_dir (latent mode only) makes this read
# generate.py's own latent_eval_data.parquet instead of silently regenerating
# eval data live with the old response+polysemantic-hard-negative schema.
# It points at generate/, where generate.py --mode latent now writes it;
# DatasetFactory falls back to a sibling inference/ for older dumps.
#
# Dispatch-vs-worker is decided by whether a 5th (mode) arg is present, not by
# SLURM_JOB_ID -- so this is safe to call with plain `bash`, either from the
# login node or nested inside another job's own worker payload (e.g.
# run_preemptable.sh calling this from its own submitted job).

set -e

cd ~/axbench

CFG="$1"; DUMP="$2"; NPROC="${3:-1}"; NUM_CHUNKS="${4:-1}"

if [ -z "$CFG" ] || [ -z "$DUMP" ]; then
  echo "Usage: bash run_jobs/run_preemptable_inference.sh <CFG> <DUMP> <NPROC> <NUM_CHUNKS>" >&2
  exit 1
fi

if [ "$NUM_CHUNKS" -lt 1 ]; then
  NUM_CHUNKS=1
fi

if [ -z "${5:-}" ]; then
  # Hardcoded, not derived from ${BASH_SOURCE[0]} -- when this script is
  # invoked nested inside another job's already-running worker payload
  # (rather than typed directly on the login node), BASH_SOURCE/dirname
  # resolves to Slurm's job spool directory, not the real repo path, and the
  # sbatch calls below would try to submit a file that doesn't exist there.
  SELF=~/axbench/run_jobs/run_preemptable_inference.sh
  if [ -f ~/axbench/.env ]; then
    set -a
    source ~/axbench/.env
    set +a
  fi

  LATENT_IDS=()
  for ((i = 0; i < NUM_CHUNKS; i++)); do
    jid=$(sbatch --parsable "$SELF" "$CFG" "$DUMP" "$NPROC" "$NUM_CHUNKS" latent "$i")
    # --parsable returns "jobid" or "jobid;cluster" -- keep only the id.
    LATENT_IDS+=("${jid%%;*}")
  done
  echo "submitted ${#LATENT_IDS[@]} latent chunk job(s): ${LATENT_IDS[*]}"

  # afterok takes a colon-separated id list; steering waits on all of them.
  DEP=$(IFS=:; echo "${LATENT_IDS[*]}")

  STEERING_IDS=()
  for ((i = 0; i < NUM_CHUNKS; i++)); do
    jid=$(sbatch --parsable --dependency=afterok:"$DEP" "$SELF" \
      "$CFG" "$DUMP" "$NPROC" "$NUM_CHUNKS" steering "$i")
    STEERING_IDS+=("${jid%%;*}")
  done
  echo "submitted ${#STEERING_IDS[@]} steering chunk job(s): ${STEERING_IDS[*]} (afterok:$DEP)"
  exit 0
fi

# --- Worker payload (runs only once submitted by sbatch, above) ---

MODE="$5"; CHUNK="$6"

CHUNK_ARGS=()
if [ "$NUM_CHUNKS" -gt 1 ]; then
  CHUNK_ARGS=(--chunk "$CHUNK" --num_chunks "$NUM_CHUNKS")
fi

MODE_ARGS=()
if [ "$MODE" = "latent" ]; then
  MODE_ARGS=(--overwrite_inference_data_dir "$DUMP/generate")
fi

# --no-sync: NUM_CHUNKS concurrent chunk jobs share one .venv; without this,
# `uv run`'s per-invocation sync races across processes (stale NFS handles,
# partial installs). Run `uv sync` once, serially, before submitting any chunks.
if [ "$MODE" = "steering" ]; then
  # CPU-only parquet bookkeeping; refuses to let steering start against an
  # incomplete or max_act-less latent merge. No torchrun, holds no GPU.
  uv run --no-sync axbench/scripts/inference.py \
    --config "$CFG" \
    --dump_dir "$DUMP" \
    --mode latent \
    --verify_chunks
fi

uv run --no-sync torchrun --nproc_per_node="$NPROC" --master_port=$((29500 + CHUNK)) axbench/scripts/inference.py \
  --config "$CFG" \
  --dump_dir "$DUMP" \
  --mode "$MODE" \
  "${MODE_ARGS[@]}" \
  "${CHUNK_ARGS[@]}"
