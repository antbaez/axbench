#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err

# Self-submitting dispatcher + worker for evaluate_local.py (local vLLM
# Llama-3.1-70B judge). One GPU is enough for the 4-bit-quantized 70B judge.
#
# Usage: bash run_preemptable_evaluate_local.sh [mode: steering|steering_test] [num_chunks]
# Defaults: mode=steering, num_chunks=5. One invocation submits all N jobs.
# Run from the repo root, or logs/out and logs/err (relative paths) won't resolve.
#
# Once every chunk's output is on disk, the last one to finish auto-merges
# them into {mode}.jsonl/{mode}_data.parquet itself (see maybe_auto_merge_chunks
# in evaluate_local.py) -- no separate merge step needed.
#
# `bash ...` just submits itself via sbatch and exits; the actual work runs
# under Slurm (SLURM_JOB_ID set) below. No OPENAI_API_KEY needed -- this
# never calls the OpenAI API. Don't `sbatch` this file directly.

set -e

if [ -z "${SLURM_JOB_ID:-}" ]; then
  MODE="${1:-steering}"
  NUM_CHUNKS="${2:-5}"
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  for ((i = 0; i < NUM_CHUNKS; i++)); do
    sbatch "$SCRIPT_DIR/run_preemptable_evaluate_local.sh" "$i" "$NUM_CHUNKS" "$MODE"
  done
  exit 0
fi

CHUNK="${1:?internal: chunk index, passed by the dispatcher branch above}"
NUM_CHUNKS="${2:-5}"
MODE="${3:-steering}"

# --- Worker payload (runs only once submitted by sbatch, above) ---

cd ~/axbench

CFG=axbench/sweep/antbaez/diffmean_variants_l20.yaml
DUMP=axbench/results/prod_9b_l20_concept500_diffmean/pos_steer_data

# flashinfer JIT-compiles a CUDA kernel via nvcc; needs a real toolkit module
# (not the venv's pip-installed nvcc, which mismatches nvidia-cuda-runtime's
# version and breaks flashinfer's CCCL header check). 12.9.1 matches torch's
# cu128 build for the pinned vllm==0.11.2.
module load cuda/12.9.1

# The .so flashinfer JIT-builds needs GLIBCXX_3.4.26+, newer than the system
# libstdc++ (3.4.25) -- point at the spack gcc's libstdc++ instead.
export LD_LIBRARY_PATH="/orcd/software/core/001/spack/pkg/gcc/12.2.0/yt6vabm/lib64:${LD_LIBRARY_PATH:-}"

# --no-sync: 5 concurrent chunks share one .venv; without this, `uv run`'s
# per-invocation sync races across processes (stale NFS handles, partial
# installs). Run `uv sync` once, serially, before submitting any chunks.
uv run --no-sync axbench/scripts/evaluate_local.py \
  --config "$CFG" \
  --dump_dir "$DUMP" \
  --mode "$MODE" \
  --chunk "$CHUNK" \
  --num_chunks "$NUM_CHUNKS"
