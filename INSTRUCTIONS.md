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

Model steering on the test set. Run this after the eval-set command above — it re-evaluates
on the held-out test split using the best factor selected there.

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results \
  --mode steering_test
```

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results \
  --mode steering
```