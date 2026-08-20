## Training

Train and save your methods:

```bash
uv run torchrun --nproc_per_node=1 axbench/scripts/train.py \
  --config axbench/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results
```