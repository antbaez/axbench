#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err

# Self-submitting dispatcher + worker for evaluate_local.py --mode steering and
# --mode steering_test (local vLLM Llama-3.1-70B judge). One GPU is enough for
# the 4-bit-quantized 70B judge. Meant to be called by run_preemptable.sh,
# which owns CFG/DUMP/NUM_CHUNKS and passes them in -- nothing is hardcoded
# here, so the dump dir can't drift out of sync with the rest of the pipeline.
#
# Usage:
#   bash run_preemptable_evaluate_local.sh <CFG> <DUMP> <NUM_CHUNKS>
#
# One invocation submits both modes: NUM_CHUNKS steering jobs (each judging its
# own disjoint slice of concepts in parallel), then NUM_CHUNKS steering_test
# jobs that all carry --dependency=afterok on *every* steering job. So no
# steering_test chunk starts until every steering chunk has succeeded.
#
# Unlike the latent -> steering barrier in run_preemptable_inference.sh, this
# one is not a data dependency: the two modes read the same steering_data.parquet
# and their chunks/merges are entirely independent. It is here to keep the two
# stages from competing for the same preemptable GPUs at once and to make the
# queue read in pipeline order. No OPENAI_API_KEY needed -- this never calls the
# OpenAI API.
#
# Once every chunk's output is on disk, the last one to finish auto-merges them
# into {mode}.jsonl/{mode}_data.parquet itself (see maybe_auto_merge_chunks in
# evaluate_local.py) -- no separate merge step needed. The merge is verified
# before it is written, then the chunk files are deleted.
#
# Dispatch-vs-worker is decided by whether a 4th (mode) arg is present, not by
# SLURM_JOB_ID -- so this is safe to call with plain `bash`, either from the
# login node or nested inside another job's own worker payload. Don't `sbatch`
# this file directly.

set -e

cd ~/axbench

CFG="$1"; DUMP="$2"; NUM_CHUNKS="${3:-5}"

if [ -z "$CFG" ] || [ -z "$DUMP" ]; then
  echo "Usage: bash run_preemptable_evaluate_local.sh <CFG> <DUMP> <NUM_CHUNKS>" >&2
  exit 1
fi

if [ "$NUM_CHUNKS" -lt 1 ]; then
  NUM_CHUNKS=1
fi

if [ -z "${4:-}" ]; then
  # Hardcoded, not derived from ${BASH_SOURCE[0]} -- when this script is
  # invoked nested inside another job's already-running worker payload
  # (rather than typed directly on the login node), BASH_SOURCE/dirname
  # resolves to Slurm's job spool directory, not the real repo path, and the
  # sbatch calls below would try to submit a file that doesn't exist there.
  SELF=~/axbench/run_preemptable_evaluate_local.sh

  STEERING_IDS=()
  for ((i = 0; i < NUM_CHUNKS; i++)); do
    jid=$(sbatch --parsable "$SELF" "$CFG" "$DUMP" "$NUM_CHUNKS" steering "$i")
    # --parsable returns "jobid" or "jobid;cluster" -- keep only the id.
    STEERING_IDS+=("${jid%%;*}")
  done
  echo "submitted ${#STEERING_IDS[@]} steering chunk job(s): ${STEERING_IDS[*]}"

  # afterok takes a colon-separated id list; steering_test waits on all of them.
  DEP=$(IFS=:; echo "${STEERING_IDS[*]}")

  TEST_IDS=()
  for ((i = 0; i < NUM_CHUNKS; i++)); do
    jid=$(sbatch --parsable --dependency=afterok:"$DEP" "$SELF" \
      "$CFG" "$DUMP" "$NUM_CHUNKS" steering_test "$i")
    TEST_IDS+=("${jid%%;*}")
  done
  echo "submitted ${#TEST_IDS[@]} steering_test chunk job(s): ${TEST_IDS[*]} (afterok:$DEP)"
  exit 0
fi

# --- Worker payload (runs only once submitted by sbatch, above) ---

MODE="$4"; CHUNK="$5"

# flashinfer JIT-compiles a CUDA kernel via nvcc; needs a real toolkit module
# (not the venv's pip-installed nvcc, which mismatches nvidia-cuda-runtime's
# version and breaks flashinfer's CCCL header check). 12.9.1 matches torch's
# cu128 build for the pinned vllm==0.11.2.
module load cuda/12.9.1

# The .so flashinfer JIT-builds needs GLIBCXX_3.4.26+, newer than the system
# libstdc++ (3.4.25) -- point at the spack gcc's libstdc++ instead.
export LD_LIBRARY_PATH="/orcd/software/core/001/spack/pkg/gcc/12.2.0/yt6vabm/lib64:${LD_LIBRARY_PATH:-}"

# --no-sync: NUM_CHUNKS concurrent chunks share one .venv; without this,
# `uv run`'s per-invocation sync races across processes (stale NFS handles,
# partial installs). Run `uv sync` once, serially, before submitting any chunks.
uv run --no-sync axbench/scripts/evaluate_local.py \
  --config "$CFG" \
  --dump_dir "$DUMP" \
  --mode "$MODE" \
  --chunk "$CHUNK" \
  --num_chunks "$NUM_CHUNKS"
