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