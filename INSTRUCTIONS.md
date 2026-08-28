## Training

Train and save your methods:

```bash
uv run torchrun --nproc_per_node=1 axbench/scripts/train.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results
```

Using pre-generated `prod_9b_l20_v1` data:

```bash
uv run torchrun --nproc_per_node=1 axbench/scripts/train.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results \
  --overwrite_data_dir axbench/concept500/prod_9b_l20_v1/generate
```

## Inference (latent)

Must run before steering inference — steering silently falls back to a factor of `1.0` if
the latent mode hasn't populated `max_activations` first.

```bash
uv run torchrun --nproc_per_node=1 axbench/scripts/inference.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results \
  --overwrite_metadata_dir axbench/concept500/prod_9b_l20_v1/generate \
  --overwrite_inference_data_dir axbench/concept500/prod_9b_l20_v1/inference \
  --mode latent
```

## Inference (steering)

Using pre-generated `prod_9b_l20_v1` data (same `--dump_dir` as training, so `inference.py`
picks up the checkpoints `train.py` wrote):

```bash
uv run torchrun --nproc_per_node=1 axbench/scripts/inference.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results \
  --overwrite_metadata_dir axbench/concept500/prod_9b_l20_v1/generate \
  --overwrite_inference_data_dir axbench/concept500/prod_9b_l20_v1/inference \
  --mode steering
```

## Evaluation

Model steering on the eval set. This selects the best steering factor per concept using
the eval split.

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results \
  --mode steering
```

## Evaluation

Model steering on the test set. Run this after the eval-set command above — it re-evaluates
on the held-out test split using the best factor selected there.

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results \
  --mode steering
```

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results \
  --mode steering_test
```

## Evaluation (local judge)

`evaluate_local.py` runs the same LM-judge steps with a local vLLM
Llama-3.1-70B judge instead of the OpenAI API. The 500 concepts are split into
5 fixed 100-concept chunks so 5 separate jobs on `mit_preemptable`. 

Run all 5 chunks for the eval split:

```bash
cd ~/axbench && for i in 0 1 2 3 4; do bash run_preemptable_evaluate_local.sh "$i" steering; done
```

Once all 5 finish, merge them into the canonical `steering.jsonl` / `steering_data.parquet`:

```bash
uv run axbench/scripts/evaluate_local.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results \
  --mode steering \
  --merge_chunks
```

Then run all 5 chunks for the test split the same way:

```bash
cd ~/axbench && for i in 0 1 2 3 4; do bash run_preemptable_evaluate_local.sh "$i" steering_test; done
```

And merge those:

```bash
uv run axbench/scripts/evaluate_local.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results \
  --mode steering_test \
  --merge_chunks
```
