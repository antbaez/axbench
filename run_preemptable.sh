#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h100:2
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err

# Self-submitting dispatcher + worker for the AxBench train/inference/evaluate
# pipeline from INSTRUCTIONS.md (diffmean_variants_l20, prod_9b_l20_v1
# pre-generated data), run as a single job on mit_preemptable.
#
# Usage:
#   bash run_preemptable.sh
# Run from the axbench repo root (or anywhere — logs/out and logs/err are
# resolved relative to the current directory at submission time, so `cd`
# into axbench/ first if you invoke this from elsewhere).
#
# When run with `bash`, this script holds no GPU allocation itself — it just
# sources .env (for OPENAI_API_KEY, used by the LLM judge in inference.py)
# and submits itself via `sbatch`, which inherits the submitting shell's
# environment. The re-submitted copy runs under Slurm (with SLURM_JOB_ID set)
# and executes the actual worker payload below. Do not `sbatch` this directly
# unless you know what you're doing.

set -e

if [ -z "${SLURM_JOB_ID:-}" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
  fi
  sbatch "$SCRIPT_DIR/run_preemptable.sh"
  exit 0
fi

# --- Worker payload (runs only once submitted by sbatch, above) ---

cd ~/axbench

CFG=axbench/sweep/antbaez/diffmean_variants_l20.yaml
DUMP=axbench/results
DATA=axbench/concept500/prod_9b_l20_v1

# Train
# uv run torchrun --nproc_per_node=1 axbench/scripts/train.py \
#   --config "$CFG" \
#   --dump_dir "$DUMP" \
#   --overwrite_data_dir "$DATA/generate"

# Inference (latent) — must run before steering, see INSTRUCTIONS.md
# uv run torchrun --nproc_per_node=1 axbench/scripts/inference.py \
#   --config "$CFG" \
#   --dump_dir "$DUMP" \
#   --overwrite_metadata_dir "$DATA/generate" \
#   --overwrite_inference_data_dir "$DATA/inference" \
#   --mode latent

# Inference (steering)
uv run torchrun --nproc_per_node=1 axbench/scripts/inference.py \
  --config "$CFG" \
  --dump_dir "$DUMP" \
  --overwrite_metadata_dir "$DATA/generate" \
  --overwrite_inference_data_dir "$DATA/inference" \
  --mode steering

# Evaluate (eval split — selects best steering factor per concept)
# uv run axbench/scripts/evaluate.py \
#   --config "$CFG" \
#   --dump_dir "$DUMP" \
#   --mode steering

# Evaluate (test split — re-evaluates using the factor selected above)
# uv run axbench/scripts/evaluate.py \
#   --config "$CFG" \
#   --dump_dir "$DUMP" \
#   --mode steering_test
