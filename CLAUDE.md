# CLAUDE.md

Reference notes for working in this repo, accumulated across sessions.

## Benchmarking a custom DiffMean-style steering method

Context: integrating a custom activation-steering method (a DiffMean variant using a
*different steering vector per token position*, rather than one vector per concept)
into AxBench's benchmark pipeline.

### 1. Pipeline overview

Five stages, each its own script under `axbench/scripts/`, all driven by one YAML
with `generate:` / `train:` / `inference:` / `evaluate:` blocks:

1. `generate.py` — synthesize training/eval data per concept (skip if reusing
   pre-generated data, see concept500 below).
2. `train.py` — fit each method per concept.
3. `inference.py --mode latent` and `--mode steering` — run concept detection and
   steering generation.
4. `evaluate.py --mode latent`, `--mode steering`, `--mode steering_test` — score
   results; `steering` picks the best factor on the eval split, `steering_test`
   re-evaluates on held-out test data using that factor.
5. `axbench/scripts/analyses.ipynb` — turns evaluation output into the
   paper/leaderboard-style numbers.

### 2. How to add a new steering method

- Subclass `Model` (`axbench/models/model.py`), or more specifically
  `MeanActivation` (`axbench/models/mean.py`) for a DiffMean-style method — usually
  only `train()` needs overriding, ending with
  `self.ax.proj.weight.data = <direction>` (plus
  `set_decoder_norm_to_unit_norm(self.ax)` to match how AxBench normalizes before
  the steering-factor sweep).
- Register the class by exporting it from `axbench/__init__.py` (directly or via an
  existing wildcard import, e.g. `from .models.mean import *`) — the scripts resolve
  model classes via `getattr(axbench, model_name)` (see
  `axbench/scripts/train.py:255,441`, `inference.py:378` etc.), so the string in the
  YAML `models:` list must exactly match the class's `__str__()`.

### 3. DiffMean variants — implemented

Three classes added to `axbench/models/mean.py`, all subclassing `MeanActivation`,
all reading activations from **left-padded** batches. The stock `DiffMean` is
deliberately left untouched so its published numbers stay reproducible.

| Class | Vectors per concept | Built from |
|---|---|---|
| `MeanTokenDiffMean` | 1 | every real token (`DiffMean`'s behavior, explicitly named) |
| `LastTokenDiffMean` | 1 | each sequence's final real token only |
| `DiffMeanPositional` | `num_positions` | the token at each offset *k* back from the sequence end |

**Why left padding.** It puts the last real token at column `-1` for *every* row, so
end-aligned indexing is a constant slice instead of a per-row `torch.gather` with
per-`(row, k)` validity masking. A new `LeftPadDataCollator` +
`make_left_padded_data_module` live locally in `mean.py`; `probe.py`'s shared
collator is deliberately not touched (see §8). Padding width is a single global max
(longest example per training call, i.e. per concept), optionally capped by
`max_seq_length`.

**No `position_ids` are passed** — the existing `gather_residual_activations` is
reused unmodified, relying on RoPE shift-invariance (§9).

**`PositionwiseAdditionIntervention`** (`axbench/models/interventions.py`): weight
`[n_concepts, num_positions, hidden]` instead of the stock
`AdditionIntervention`'s `[n_concepts, hidden]`-broadcast-to-all-positions. Applies
`v_0` to the final prompt token, `v_k` k columns earlier. It suppresses itself on
decode steps via `base.shape[1] <= 1` — stateless, so it stays correct across the
batch loop in `predict_steer`, unlike a call counter which would need resetting per
`generate()` call (a hook `Model.predict_steer` does not provide).

**Inference semantics: prompt-only, end-aligned.** Training indexes back from the
last real token of the instruction+response sequence; at inference the vectors are
applied end-aligned to the *prompt* and generation is left unsteered. *Accepted
caveat:* "the end" means end-of-response in training but end-of-prompt at test, so
vectors are applied to a somewhat different distribution than they were fit on.
Revisit if results look weak.

**Other implementation notes:**
- *Latent mode:* the positional matrix is collapsed to a mean-across-positions
  single vector so inherited `predict_latent` / `pre_compute_mean_activations` /
  `get_logits` work unchanged. This path is **required**, not optional (§8).
- *save/load:* both tensors are packed as a **dict** into the standard
  `{model_name}_weight.pt` so they ride `train.py`'s existing dict-merge branch —
  a separate file would silently never be merged across ranks (§8).
  `num_positions` is derived from the checkpoint shape, never from config, so train
  and inference cannot disagree.
- *Config:* `num_positions` and `max_seq_length` added to `ModelParams`,
  `hierarchical_params`, and `_infer_type`'s `int_params` in
  `axbench/scripts/args/training_args.py` — all three are needed (§8).
- *Sweep yamls:* `diffmean_variants.yaml` under `2b/l10`, `9b/l20`, `9b/l31`, each
  matching its sibling `no_grad.yaml`'s model/layer/data settings (note 9B uses
  `steering_batch_size: 5`, not 10).

**Verification status.** Statically verified: everything compiles; the index math and
the save → per-rank-merge → load round trip are unit-tested against the real shipped
source. **Not yet run on a GPU** — the two checks that still matter are (a) `DiffMean`
producing bit-identical weights before/after these edits, and (b) `MeanTokenDiffMean`
matching `DiffMean` within bf16 noise (cosine > ~0.999). Check (b) is the empirical
test of the RoPE assumption in §9 — if it fails, the left-padding premise is wrong,
not just the code.

### 4. Running the pipeline

```bash
CFG=<path/to/config.yaml>
DUMP=<path/to/dump_dir>   # must stay consistent across every command below

# 1. Generate (skip if reusing pre-generated data)
uv run axbench/scripts/generate.py --config $CFG --mode training --dump_dir $DUMP
uv run axbench/scripts/generate.py --config $CFG --mode latent   --dump_dir $DUMP

# 2. Train
uv run torchrun --nproc_per_node=$gpu_count axbench/scripts/train.py \
  --config $CFG --dump_dir $DUMP

# 3. Inference
uv run torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config $CFG --dump_dir $DUMP --mode latent
uv run torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config $CFG --dump_dir $DUMP --mode steering

# 4. Evaluate
uv run axbench/scripts/evaluate.py --config $CFG --dump_dir $DUMP --mode latent
uv run axbench/scripts/evaluate.py --config $CFG --dump_dir $DUMP --mode steering
uv run axbench/scripts/evaluate.py --config $CFG --dump_dir $DUMP --mode steering_test
```

`torchrun --nproc_per_node=$gpu_count` wraps `train.py`/`inference.py` only —
`generate.py` and `evaluate.py` run as plain `uv run` scripts.

### 5. Using pre-generated `concept500` data (skip data generation)

Point `train.py`/`inference.py` at the pre-built data instead of running
`generate.py`:

```bash
DATA=axbench/concept500/<split>   # prod_2b_l10_v1 | prod_2b_l20_v1 | prod_9b_l20_v1 | prod_9b_l31_v1

uv run torchrun --nproc_per_node=$gpu_count axbench/scripts/train.py \
  --config $CFG --dump_dir $DUMP --overwrite_data_dir $DATA/generate

uv run torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config $CFG --dump_dir $DUMP --mode latent \
  --overwrite_metadata_dir $DATA/generate --overwrite_inference_data_dir $DATA/inference

uv run torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config $CFG --dump_dir $DUMP --mode steering \
  --overwrite_metadata_dir $DATA/generate --overwrite_inference_data_dir $DATA/inference
```

`evaluate.py` takes no `--overwrite_*` flags — it just reads whatever `train.py`/
`inference.py` already wrote into `$DUMP`. The four splits correspond to sweep
yamls under `axbench/sweep/wuzhengx/<size>/<layer>/`.

### 6. `concept500` provenance and (non-)reproducibility

- Already committed directly to git (commit `b9ca416`, "[Major] Adding concept 500
  in git"), not gitignored, not fetched by a download script — no regeneration or
  download step is needed to match the benchmark's exact data.
- **Verified from code:** re-running `generate.py` would *not* reproduce this exact
  dataset. Concept *selection* is seeded and deterministic (`generate_training()` in
  `axbench/scripts/generate.py` does `set_seed(args.seed)` then a shuffle, taking
  the first `max_concepts`) — same seed/config gives the same 500 concepts, same
  order. But the LLM-*synthesized example text* is not deterministic:
  `axbench/models/language_models.py:113` defaults `temperature` to `1.0`, and the
  `chat.completions.create` call at `language_models.py:148-149` passes no `seed=`
  parameter. So a fresh `generate.py` run reproduces the same 500 concepts but
  different synthetic sentences.

### 7. Which yaml runs DiffMean steering

`axbench/sweep/wuzhengx/<size>/<layer>/no_grad.yaml` — exists for `2b/l10`,
`2b/l20`, `9b/l20`, `9b/l31`. Trains `DiffMean` alongside `PCA`/`LAT` (the other
closed-form, gradient-free vector methods, hence "no_grad"), and lists `DiffMean` in
both `inference.models` and `evaluate.models` with `steering_evaluators:
["PerplexityEvaluator", "LMJudgeEvaluator"]`.

Distinct from `16k_diffmean.yaml` / `16k_diffmean_crossfit.yaml` (under `9b/l20` and
`2b/l20` only), which run `DiffMean` alone on the full 16K-concept set
(`max_concepts: 16000`), not `concept500` — a different, larger-scale experiment,
not the steering leaderboard.

**Unverified / inferred, flagged explicitly:** it is *believed but not confirmed*
that `no_grad.yaml` is what produced the `DiffMean` row in the top-level
`README.md` leaderboard table (`0.297 | 0.178 | 0.322 | 0.158 | 0.239` for
2B L10 / 2B L20 / 9B L20 / 9B L31 / Avg). Those exact numbers appear nowhere else in
the repo — no results file, log, or notebook output ties them to a specific run.
The inference rests on circumstantial evidence: `no_grad.yaml`'s four splits match
the table's four columns exactly, it's the only `DiffMean` config that uses
`concept500` data, and `experiment_commands.txt` runs it against dump dirs named
`prod_*_concept500_no_grad`. To actually confirm this, run `no_grad.yaml`
end-to-end per section 4/5 above and compare the `evaluate.py --mode steering_test`
output to the table.

### 8. Silent-failure traps in this repo

Each of these fails quietly — wrong results or ignored settings, no error:

- **Unknown per-model YAML keys are dropped without warning.**
  `training_args.py` only copies a key into `ModelParams` if it appears in the
  `hierarchical_params` list (`if param_name in hierarchical_params`). A new
  per-model setting must be added in **three** places — the `ModelParams` dataclass,
  `hierarchical_params`, and `_infer_type`'s type lists — or it is silently ignored
  and the model trains with a default.
- **Steering needs latent mode to have run first.** `inference.py` calls
  `pre_compute_mean_activations` before steering, which reads
  `{ClassName}_max_act` from the latent parquet files. If that column doesn't exist,
  `max_activations` is empty and `predict_steer` silently falls back to `1.0`,
  quietly changing what every entry in the `steering_factors` sweep means. Any new
  method therefore needs a working latent path, not just a steering one.
- **Per-rank checkpoint merging only covers `_weight.pt` / `_bias.pt`.** `train.py`
  saves per-rank files and merges them by concatenating along dim 0. Any extra
  checkpoint file a method writes will never be merged, and inference will fail to
  find it. The merge does handle **dict**-valued weight files (concatenating each
  key), which is the supported way to store more than one tensor.
- **`probe.py`'s `DataCollator` derives `attention_mask` by value**
  (`input_ids != pad_token_id`), not by position. A real content token that happens
  to equal `pad_token_id` would be silently masked out of training.
- **Silent truncation at 1024 tokens** in `probe.make_data_module`
  (`truncation=True`, no logging). For binarized rows this is instruction+response
  concatenated, so long responses can lose their tail before the method sees them.
- **Namespace shadowing via `axbench/__init__.py`.** It star-imports `probe` before
  `mean`, so a symbol in `mean.py` sharing a name with one in `probe.py` silently
  wins in the `axbench` namespace. Name new helpers distinctly (e.g.
  `LeftPadDataCollator`, not `DataCollator`).

### 9. Verified internals (checked against pinned source)

**Left padding is safe without `position_ids` — for Gemma-2/RoPE.** An earlier
assumption that it would "silently corrupt activations" was **wrong**:
- Gemma-2 is RoPE-only; the only `nn.Embedding` is `embed_tokens`, so there are no
  learned absolute position embeddings to shift.
- RoPE attention scores depend only on the *relative* offset, so shifting every
  position by a constant leaves real-token outputs mathematically unchanged.
- No NaN risk: the causal mask fills with `min_dtype` (finite), not `-inf`, so
  fully-masked pad rows softmax to a finite garbage average, and real tokens attend
  to them with weight exactly 0.

Scope: verified for Gemma-2 against `transformers==4.45.1`. Llama-3.1 (present in
`axbench/sweep/wuzhengx/llama_8b/`) is also RoPE-based so the reasoning should carry,
but that was **not** verified. A model with learned absolute position embeddings
would invalidate this.

**`intervene_on_prompt=True` is functionally inert as AxBench calls it.**
`Model.predict_steer` passes `unit_locations=None`; pyvene's
`_broadcast_unit_locations` returns all-`None` for that, which makes the hook skip
its `intervene_on_prompt` gate entirely. The intervention therefore fires on the
prompt prefill **and** on every generated token, regardless of the flag. Anything
that should apply to only one of those must gate itself.

**What `DiffMean` actually trains on.** With `binarize_dataset: true`, `prepare_df`
feeds it the instruction **and** response *concatenated* into one text field (raw
concatenation for non-chat models; a user turn plus an assistant turn via
`apply_chat_template` for chat models). `prefix_length` strips only the fixed
chat-template boilerplate (measured by diffing two single-character messages), **not**
the prompt. So `DiffMean` pools every non-prefix, non-padding token from both halves,
weighted equally — longer examples contribute proportionally more token-vectors.

**`steering_intervention_type`** (in the `inference:` block) selects the intervention
module at load time: `"addition"` → `AdditionIntervention` (add a scaled direction at
every position) vs `"clamping"` → `SubspaceIntervention` (project the direction out,
then replace it with a scaled one, leaving the prompt prefix untouched). It is read
once and applied to **every** model in the run — it is not a per-method setting.

**`GemmaScopeSAEDiffMean` averages differently from `DiffMean`** despite the shared
name and file: it averages per-example first and then across examples (equal weight
per example), whereas `DiffMean` pools all tokens (equal weight per token). Don't
assume they're comparable.
