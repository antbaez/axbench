#!/bin/bash

# Dispatcher for the AxBench generate/train/inference/evaluate pipeline
# (diffmean_variants_l31, our own generated pos_steer_data dataset).
#

# generate/inference/evaluate_local each dispatch both of their modes in one
# call, chained so the second waits on the first.
#
# Stages themselves are unchained -- wait for one to finish before
# uncommenting the next (train needs generate's output, inference needs train's).

set -e

# Load OPENAI_API_KEY
if [ -f ~/axbench/.env ]; then
  set -a
  source ~/axbench/.env
  set +a
fi

CFG=axbench/sweep/antbaez/diffmean_variants_l31.yaml
DUMP=axbench/results/prod_9b_l31_concept500_diffmean_pos_steer_data
NPROC=1
NUM_CHUNKS=4
NUM_WORKERS=50
MAX_CONCURRENT_REQUESTS=1000

# Generate: synthesizes each concept's training data and held-out latent eval data
bash run_jobs/run_preemptable_generate.sh "$CFG" "$DUMP" "$NUM_WORKERS" "$MAX_CONCURRENT_REQUESTS"


# Train: fits each method's steering direction(s) per concept
# sbatch run_jobs/run_preemptable_job.sh uv run torchrun --nproc_per_node="$NPROC" axbench/scripts/train.py \
#   --config "$CFG" \
#   --dump_dir "$DUMP"


# Inference: latent calibrates each direction's natural firing strength, then steering injects it into generation across concepts x prompts x factors
# bash run_jobs/run_preemptable_inference.sh "$CFG" "$DUMP" "$NPROC" "$NUM_CHUNKS"
# Latent chunks take ~1.5 min each (4 chunks)


# Evaluate: scores steered generations for concept presence, fluency, and instruction-following, using a local judge model
# bash run_jobs/run_preemptable_evaluate_local.sh "$CFG" "$DUMP" "$NUM_CHUNKS"






# Evaluate (subset, OpenAI judge, runs locally: picks each concept/model's best steering factor on the eval split, then scores only that factor on the test split)
# uv run axbench/scripts/evaluate_subset.py \
#   --config "$CFG" \
#   --dump_dir "$DUMP" \
#   --mode select_best