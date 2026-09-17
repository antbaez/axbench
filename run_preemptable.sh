#!/bin/bash

# Dispatcher for the AxBench generate/train/inference/evaluate pipeline
# (diffmean_variants_l20, our own generated pos_steer_data dataset).
#
# Usage:
#   bash run_preemptable.sh
# Run from the axbench repo root (or anywhere -- run_preemptable_job.sh cd's
# into ~/axbench itself, and logs/out and logs/err are resolved relative to
# that).
#
# Plain bash -- this file holds no GPU allocation and is not itself an sbatch
# script. Each uncommented stage below submits its own independent job via
# `sbatch run_preemptable_job.sh <command>`, which owns the actual #SBATCH
# resource directives. Uncommenting multiple stages submits multiple
# independent jobs with no ordering guarantee between them -- wait for one to
# finish before running a stage that depends on it.
#
# The inference latent/steering stages below call their own standalone
# dispatcher scripts (run_preemptable_inference_{latent,steering}.sh), which
# submit NUM_CHUNKS separate sbatch jobs of their own.

set -e

# Load OPENAI_API_KEY (and anything else in .env) for stages run directly
# in this shell rather than via sbatch run_preemptable_job.sh.
if [ -f ~/axbench/.env ]; then
  set -a
  source ~/axbench/.env
  set +a
fi

CFG=axbench/sweep/antbaez/diffmean_variants_l20.yaml
DUMP=axbench/results/prod_9b_l20_concept500_diffmean/pos_steer_data
NPROC=1
NUM_CHUNKS=4
# generate.py --mode training: concepts generated concurrently, and a cap on total
# in-flight OpenAI requests across them (set from your gpt-4o-mini rate limits)
NUM_WORKERS=50
MAX_CONCURRENT_REQUESTS=1000

# Generate (training data: synthesizes concept-injected positive examples + contrastive minimal-edit negatives, per concept)
# uv run axbench/scripts/generate.py \
#   --config "$CFG" \
#   --mode training \
#   --dump_dir "$DUMP" \
#   --num_workers "$NUM_WORKERS" \
#   --max_concurrent_requests "$MAX_CONCURRENT_REQUESTS"

# Generate (latent eval data: synthesizes held-out eval data used for concept-detection scoring in latent mode)
# uv run axbench/scripts/generate.py \
#   --config "$CFG" \
#   --mode latent \
#   --dump_dir "$DUMP" \
#   --num_workers "$NUM_WORKERS" \
#   --max_concurrent_requests "$MAX_CONCURRENT_REQUESTS"


# Train (fits every configured method — DiffMean, PCA, LAT, DiffMean variants, etc. — independently per concept, sharded across torchrun ranks)
# sbatch run_preemptable_job.sh uv run torchrun --nproc_per_node="$NPROC" axbench/scripts/train.py \
#   --config "$CFG" \
#   --dump_dir "$DUMP"


# Inference (latent: scores how strongly each concept's direction fires on held-out text; writes max_act, which calibrates what each steering factor means)
# bash run_preemptable_inference.sh latent "$CFG" "$DUMP" "$NPROC" "$NUM_CHUNKS"

# Inference (steering: generates concept-steered text per concept x eval prompt x steering factor, injecting the direction scaled by latent mode's max_act; requires latent mode to have run first)
# bash run_preemptable_inference.sh steering "$CFG" "$DUMP" "$NPROC" "$NUM_CHUNKS"


# Evaluate (eval split, local vLLM Llama-3.1-70B judge: routes the LM judge through a local GPU-resident model instead of the OpenAI API, auto-merging its chunks once they all finish)
# bash run_preemptable_evaluate_local.sh steering "$NUM_CHUNKS"

# Evaluate (test split, local vLLM Llama-3.1-70B judge)
bash run_preemptable_evaluate_local.sh steering_test "$NUM_CHUNKS"






# Evaluate (eval split, OpenAI judge: scores steering generations via PerplexityEvaluator + LMJudgeEvaluator, per concept/factor)
# sbatch run_preemptable_job.sh uv run axbench/scripts/evaluate.py \
#   --config "$CFG" \
#   --dump_dir "$DUMP" \
#   --mode steering

# Evaluate (test split, OpenAI judge: re-scores generations on the test split; picking the best eval-split factor per concept is a manual downstream step, not automated here)
# sbatch run_preemptable_job.sh uv run axbench/scripts/evaluate.py \
#   --config "$CFG" \
#   --dump_dir "$DUMP" \
#   --mode steering_test