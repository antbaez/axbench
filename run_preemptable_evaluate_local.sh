#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err

# Self-submitting dispatcher + worker for evaluate_local.py (local vLLM
# Llama-3.1-70B judge, diffmean_variants_l20, prod_9b_l20_v1 pre-generated
# data), run as a single job on mit_preemptable. Only needs 1 GPU -- the
# 4-bit-quantized 70B judge fits on one (tensor_parallel_size=1).
#
# Usage:
#   bash run_preemptable_evaluate_local.sh
# Run from the axbench repo root (or anywhere -- logs/out and logs/err are
# resolved relative to the current directory at submission time, so `cd`
# into axbench/ first if you invoke this from elsewhere).
#
# When run with `bash`, this script holds no GPU allocation itself -- it just
# submits itself via `sbatch`, which inherits the submitting shell's
# environment. The re-submitted copy runs under Slurm (with SLURM_JOB_ID set)
# and executes the actual worker payload below. No OPENAI_API_KEY / .env
# needed here -- evaluate_local.py never talks to the OpenAI API.
# Do not `sbatch` this directly unless you know what you're doing.

set -e

if [ -z "${SLURM_JOB_ID:-}" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  sbatch "$SCRIPT_DIR/run_preemptable_evaluate_local.sh"
  exit 0
fi

# --- Worker payload (runs only once submitted by sbatch, above) ---

cd ~/axbench

CFG=axbench/sweep/antbaez/diffmean_variants_l20.yaml
DUMP=axbench/results

# Evaluate (eval split -- selects best steering factor per concept)
uv run axbench/scripts/evaluate_local.py \
  --config "$CFG" \
  --dump_dir "$DUMP" \
  --mode steering

# Evaluate (test split -- re-evaluates using the factor selected above)
# uv run axbench/scripts/evaluate_local.py \
#   --config "$CFG" \
#   --dump_dir "$DUMP" \
#   --mode steering_test
