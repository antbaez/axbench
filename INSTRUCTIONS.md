## Training

Train and save your methods:

```bash
uv run torchrun --nproc_per_node=1 axbench/scripts/train.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results/prod_9b_l20_concept500_diffmean
```

Using pre-generated `prod_9b_l20_v1` data:

```bash
uv run torchrun --nproc_per_node=1 axbench/scripts/train.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results/prod_9b_l20_concept500_diffmean \
  --overwrite_data_dir axbench/concept500/prod_9b_l20_v1/generate
```

## Inference (latent)

Must run before steering inference — steering silently falls back to a factor of `1.0` if
the latent mode hasn't populated `max_activations` first.

Sharded across N preemptable single-GPU jobs. One invocation submits them all; run
`uv sync` once first, since the jobs use `--no-sync` on a shared venv.

```bash
cd ~/axbench && uv sync
bash run_preemptable_inference.sh latent        # defaults to 5 chunks
```

Once all chunks finish, merge and verify (no GPU, no torchrun):

```bash
uv run axbench/scripts/inference.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results/prod_9b_l20_concept500_diffmean \
  --overwrite_metadata_dir axbench/concept500/prod_9b_l20_v1/generate \
  --mode latent --merge_chunks --verify_chunks
```

`--verify_chunks` fails loudly if any concept is missing or any `{Model}_max_act` column
is absent or null. Do not start steering until it passes.

## Inference (steering)

Requires the merged, verified `inference/latent_data.parquet` from the step above —
`pre_compute_mean_activations` only reads files matching `latent_*.parquet`, which the
per-chunk files deliberately don't.

```bash
bash run_preemptable_inference.sh steering      # defaults to 5 chunks
```

Then merge and verify:

```bash
uv run axbench/scripts/inference.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results/prod_9b_l20_concept500_diffmean \
  --overwrite_metadata_dir axbench/concept500/prod_9b_l20_v1/generate \
  --mode steering --merge_chunks --verify_chunks
```

Both modes take `<CFG> <DUMP> <NPROC>` plus a chunk count as their last argument, e.g.
`bash run_preemptable_inference.sh latent axbench/sweep/antbaez/diffmean_variants_l20.yaml axbench/results/prod_9b_l20_concept500_diffmean 1 8`.

## Evaluation

Model steering on the eval set. This selects the best steering factor per concept using
the eval split.

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results/prod_9b_l20_concept500_diffmean \
  --mode steering
```

## Evaluation

Model steering on the test set. Run this after the eval-set command above — it re-evaluates
on the held-out test split using the best factor selected there.

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results/prod_9b_l20_concept500_diffmean \
  --mode steering
```

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results/prod_9b_l20_concept500_diffmean \
  --mode steering_test
```

## Evaluation (local judge)

`evaluate_local.py` runs the same LM-judge steps with a local vLLM
Llama-3.1-70B judge instead of the OpenAI API. The 500 concepts are split into
5 fixed 100-concept chunks so 5 separate jobs on `mit_preemptable`. 

Run the eval split (one invocation submits all chunks):

```bash
cd ~/axbench && bash run_preemptable_evaluate_local.sh steering
```

Once all 5 finish, merge them into the canonical `steering.jsonl` / `steering_data.parquet`:

```bash
uv run axbench/scripts/evaluate_local.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results/prod_9b_l20_concept500_diffmean \
  --mode steering \
  --merge_chunks
```

Then the test split the same way:

```bash
cd ~/axbench && bash run_preemptable_evaluate_local.sh steering_test
```

And merge those:

```bash
uv run axbench/scripts/evaluate_local.py \
  --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
  --dump_dir axbench/results/prod_9b_l20_concept500_diffmean \
  --mode steering_test \
  --merge_chunks
```
