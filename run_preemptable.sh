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
# script. Each uncommented stage below submits its own job(s), which own the
# actual #SBATCH resource directives.
#
# Each of the generate/inference/evaluate_local stages is a single call to its
# own standalone dispatcher script, which submits *both* of that stage's modes
# in one go and chains them with --dependency=afterok: every chunk of the
# second mode waits for every chunk of the first to succeed. So one command
# per stage, and no manual waiting between the two halves of a stage.
#
# Ordering *between* stages is still manual -- these three dispatchers know
# nothing about each other, so wait for one stage's jobs to finish before
# uncommenting the next (train.py in particular needs generate's output, and
# inference needs train's).

set -e

# Load OPENAI_API_KEY (and anything else in .env) for stages run directly
# in this shell rather than via sbatch run_preemptable_job.sh.
if [ -f ~/axbench/.env ]; then
  set -a
  source ~/axbench/.env
  set +a
fi

CFG=axbench/sweep/antbaez/diffmean_variants_l20.yaml
DUMP=axbench/results/prod_9b_l20_concept500_diffmean_pos_steer_data
NPROC=1
NUM_CHUNKS=4
# generate.py --mode training: concepts generated concurrently, and a cap on total
# in-flight OpenAI requests across them (set from your gpt-4o-mini rate limits)
NUM_WORKERS=50
MAX_CONCURRENT_REQUESTS=1000

# Generate (both modes run locally in this shell, not via sbatch: training data — concept-injected positive examples + contrastive minimal-edit negatives per concept — then latent eval data, which only starts if training succeeded since it reads the metadata training writes; blocks until both finish)
# bash run_preemptable_generate.sh "$CFG" "$DUMP" "$NUM_WORKERS" "$MAX_CONCURRENT_REQUESTS"


# Train (fits every configured method — DiffMean, PCA, LAT, DiffMean variants, etc. — independently per concept, sharded across torchrun ranks)
# sbatch run_preemptable_job.sh uv run torchrun --nproc_per_node="$NPROC" axbench/scripts/train.py \
#   --config "$CFG" \
#   --dump_dir "$DUMP"


# Inference (both modes chained in one submission: latent scores how strongly each concept's direction fires on held-out text and writes max_act, then steering generates concept-steered text per concept x eval prompt x steering factor; every steering chunk waits on every latent chunk, since steering reads max_act from the merged latent parquet and silently falls back to a factor scale of 1.0 without it)
bash run_preemptable_inference.sh "$CFG" "$DUMP" "$NPROC" "$NUM_CHUNKS"
# Latent chunks take ~1.5 min each (4 chunks)


# Evaluate (both splits chained in one submission, local vLLM Llama-3.1-70B judge instead of the OpenAI API: eval split first, then test split once every eval-split chunk has finished, each auto-merging its own chunks)
# bash run_preemptable_evaluate_local.sh "$CFG" "$DUMP" "$NUM_CHUNKS"






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