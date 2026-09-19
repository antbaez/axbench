# CLAUDE.md

## Before making any changes

Before making any changes to this repo (code, config, data files), first tell the user
what you are going to change and why. Do not edit files until they've had a chance to
respond.

Reference notes for working in this repo, accumulated across sessions.

## Benchmarking a custom DiffMean-style steering method

Context: integrating a custom activation-steering method (a DiffMean variant using a
*different steering vector per token position*, rather than one vector per concept)
into AxBench's benchmark pipeline.

### 1. Pipeline overview

Five stages, each its own script under `axbench/scripts/`, all driven by one YAML
with `generate:` / `train:` / `inference:` / `evaluate:` blocks. `generate.py` and
`evaluate.py`/`evaluate_local.py` run as plain `uv run` scripts; `train.py` and
`inference.py` run under `torchrun`.

1. **`generate.py`** — synthesize training/eval data per concept (skip if reusing
   pre-generated data, see concept500 below).

2. **`train.py`** — fits every configured method (`DiffMean`, PCA, LAT, probes,
   the `mean.py` variants, etc.) independently per concept. Concepts (not model
   types) are what get split across `torchrun` ranks — every rank trains *all*
   configured methods, just on its own concept slice — and each rank writes its
   own `rank_{r}_{model_name}_weight.pt`/`_bias.pt`. At the end, rank 0 merges all
   ranks' files by concatenating along the concept dimension into the canonical
   `{model_name}_weight.pt`/`_bias.pt` (dict-valued checkpoints are merged per-key,
   which is how multi-tensor methods like `DiffMeanPositional` survive the merge —
   see §8). Per-rank state (`train_state.pkl_rank_{r}`) records `last_concept_id`
   so a preempted run resumes instead of retraining finished concepts.
   `HyperSteer` is a special case: it skips the per-concept loop entirely and
   trains once on the full dataset from rank 0 only, so the resume/skip machinery
   doesn't apply to it.

3. **`inference.py --mode latent`** — concept-detection scoring: for each trained
   method/concept, scores how strongly its direction fires on held-out text,
   independent of generation. Writes `{ModelName}_max_act` (+ `_max_act_idx`,
   `tokens`) per example, per-rank, to `inference/rank_*_latent[_chunk*]_data.parquet`.
   This `max_act` is the value `pre_compute_mean_activations` later reads to
   calibrate what a given steering factor means in raw activation units for that
   concept — this is why every method needs a working latent path even if the
   real goal is only steering (§8). `--chunk`/`--num_chunks` split the concept-id
   space into deterministic, non-overlapping windows so N independent preemptable
   jobs can each own a slice. The last chunk to finish auto-merges them into the
   single `latent_data.parquet` (`maybe_auto_merge_chunks`), so a separate
   `--merge_chunks` pass is normally unnecessary — but the auto-merge swallows every
   exception and only warns, so a chunk can exit 0 without the merge having happened.
   Every merge (auto or `--merge_chunks`) is verified — full concept coverage, and
   for latent a populated `{Model}_max_act` per model — before it replaces
   `{mode}_data.parquet` via a temp file, so a failed or partial merge (e.g. two
   chunks racing) never overwrites a good one. After a verified auto-merge the
   `rank_*_{mode}_chunk*_data.parquet` files and their resume-state pickles are
   deleted (manual `--merge_chunks` only deletes when `--verify_chunks` is also
   passed); a chunk requeued afterwards sees its window already in the merged file
   and exits instead of redoing it. To force a re-run, delete `{mode}_data.parquet`.
   The exact filename matters: `pre_compute_mean_activations` reads
   `latent_data.parquet` by name, deliberately not a `latent_*` glob, since older
   dumps (and the released `concept500` data) keep `generate.py --mode latent`'s
   `latent_eval_data.parquet` in that same `inference/` directory, and that file has
   no `max_act` column. `generate.py --mode latent` now writes it to `generate/`
   instead, so each stage writes only its own directory; `DatasetFactory` looks for
   it in the `--overwrite_inference_data_dir` it is given, then in a sibling
   `inference/`, and raises if neither has it. `--verify_chunks` (CPU-only, it
   returns before any GPU work) fails loudly if a concept is missing or a model's
   `{Model}_max_act` column is absent/null. A non-positive `max_act` is silently
   coerced to `50` downstream rather than left as-is.

4. **`inference.py --mode steering`** — generates concept-steered text: for every
   concept × eval prompt × configured steering factor, injects a learned
   direction into hidden states during generation. Requires latent mode's merged
   `latent_data.parquet` first; this is now a hard error rather than the silent
   factor-scale fallback to `1.0` it used to be — `pre_compute_mean_activations`
   raises `FileNotFoundError` if the file is absent, and `inference.py` then raises
   `ValueError` naming any `concept_id` the merge doesn't cover (§8). The actual
   injected magnitude is `factor * max_act` (logged as `strength`), so equal factors
   are *not* equal magnitudes across models or concepts: `max_act` varies per
   (concept, model), and which method gets pushed hardest at a given factor changes
   from concept to concept. Compare methods at their own best factor, not at a shared
   one. Which intervention module runs is
   picked once, globally, via the yaml's `steering_intervention_type` and applied
   to *every* model in the run: `"addition"` adds the scaled direction at every
   position (prompt and generated tokens alike); `"clamping"` projects the
   direction out and replaces it with the scaled version, restoring the original
   prefix during prefill only. Because pyvene's `unit_locations=None` handling
   makes it ignore `intervene_on_prompt` entirely, the hook actually fires on both
   prompt and every decode step regardless of that flag — any intervention that
   should be prompt-only or decode-excluded must self-gate on `base.shape[1]`, as
   `PositionwisePromptAdditionIntervention` and `SubspaceIntervention` both do (§9).
   `MeanTokenDiffMean`/`LastTokenDiffMean` opt out of this global selection
   entirely — they override `make_model()` to always use `PromptAdditionIntervention`
   regardless of `steering_intervention_type` (§3). Sharding/chunking and
   `--merge_chunks`/`--verify_chunks` work the same way as latent mode, writing to
   `steering_data.parquet`.

5. **`evaluate.py --mode steering` / `--mode steering_test`** — scores the
   `steering_data.parquet` generations concept-by-concept, fanning out
   `(concept, evaluator, model)` combinations across a process pool. Evaluators
   are config-driven (`steering_evaluators`, typically `PerplexityEvaluator` +
   `LMJudgeEvaluator`); `LMJudgeEvaluator` calls an OpenAI-API judge model three
   times per generation (concept relevance / instruction relevance / fluency, 0-2
   each) and combines them via a harmonic mean, then averages per steering
   factor. Output is `steering[_test].jsonl` (per-concept, per-factor scores) and
   an annotated `steering[_test]_data.parquet`, plus plots and OpenAI usage/cost
   logging. **Important, verified from code:** a `get_best_factors()` function
   (argmax of mean `lm_judge_rating` per concept) exists in this file, but it is
   *not* invoked anywhere in the `--mode steering`/`steering_test` main flow — no
   automatic "pick the best eval-split factor and apply it to the test split"
   step actually runs inside `evaluate.py`. Both modes evaluate every steering
   factor present in the input parquet; the only difference between them is which
   half of each concept's `input_id`s `data_generator` yields (governed by
   `winrate_split_ratio`, off by default). Turning eval-split scores into a single
   best-factor test-split number is therefore a join a downstream consumer
   (`analyses.ipynb`, presumably) must do by hand — treat the "steering picks the
   best factor, steering_test re-evaluates using it" framing as the *intended*
   design, not something this script currently automates. Malformed judge
   responses silently fall back to a rating of `0.0` rather than raising.

6. **`evaluate_local.py --mode steering_test`** — a drop-in variant of
   `evaluate.py` that runs the identical `eval_steering()` code path but routes
   LM-judge calls through one shared, GPU-resident local vLLM Llama-3.1-70B judge
   instead of the OpenAI API (built for a preemptable HPC partition, batching
   requests via vLLM instead of a process pool of API clients). The same
   "`get_best_factors` exists but isn't wired into `steering_test`" caveat above
   applies here too — it re-judges whichever factors are present in its data
   slice, it doesn't filter to a previously-selected best factor. The 500
   concepts are split into 5 fixed, deterministic chunks via the same
   `chunk_bounds()` logic `inference.py` uses (so both stages shard identically);
   `run_preemptable_evaluate_local.sh` self-resubmits one Slurm job per chunk with
   `--requeue`, and per-concept state pickles let a preempted chunk resume rather
   than restart. The last chunk to finish auto-merges all `{mode}_chunk*` files into
   the canonical `{mode}.jsonl`/`{mode}_data.parquet` (`--merge_chunks` does the same
   by hand). The merge is refused — nothing replaced, nothing deleted — unless every
   concept is present in the jsonl and the parquet covers the same concepts; once
   verified, the chunk jsonl/parquet/state files and `temp_*_chunk*.pkl` are deleted,
   and a chunk requeued afterwards exits before loading the judge if its window is
   already in the merged `{mode}.jsonl`. `steering` and `steering_test` chunks/merges are
   entirely independent of each other. For spot-checking, `lm_judge_local.py` dumps
   the first 3 examples of each of the three judge types — full rendered prompt, raw
   judge completion, parsed rating — once per process, so each chunk logs its own
   first concept (9 blocks each). The parsed rating sits next to the raw completion
   on purpose: the parser needs the literal string `Rating:` and otherwise falls back
   to `0.0`, so format drift would otherwise read as uniformly terrible steering.

6b. **`evaluate_short.py --mode steering|steering_test`** — a subsampled OpenAI-judge
   variant of `evaluate.py` for quick, cheap looks. It reuses `evaluate.py`'s client
   construction and the same `LMJudgeEvaluator`, so any cell it reports matches what
   `evaluate.py` would give; only row selection and output format differ.
   `--top_factors N` (the N largest factors, e.g. 4 → 2.5/3.0/4.0/5.0),
   `--factor_stride N` (2 = alternating 7 of 14), `--examples_per_concept N` and
   `--max_concepts N` each take a count or `all`; `--num_workers` runs judge calls
   across threads, each with its own client. Writes `{mode}_short.json` (per-factor
   means) and `{mode}_short_ratings.jsonl` (one line per judged generation, all raw
   ratings) under `{dump}/evaluate/`, then prints a factor × model table and each
   method's best factor. Needs no GPU — `evaluate.py` loads no local model when
   `LMJudgeEvaluator` is the only steering evaluator.

7. **`axbench/scripts/analyses.ipynb`** — turns evaluation output into the
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

| Class | Vectors per concept | Built from | Steers via |
|---|---|---|---|
| `MeanTokenDiffMean` | 1 | every real token (`DiffMean`'s behavior, explicitly named) | `PromptAdditionIntervention` |
| `LastTokenDiffMean` | 1 | each sequence's final real token only | `PromptAdditionIntervention` (inherited) |
| `DiffMeanPositional` | `num_positions` | the token at each offset *k* back from the sequence end | `PositionwisePromptAdditionIntervention` |

The last column is each class's own `make_model()` override, not the yaml's
`steering_intervention_type` — all three hardcode a prefill-only intervention so none
of them ever touches a generated token, unlike the stock `"addition"`/`"clamping"`
choices which apply to every other method in the run (§1, §9).

**Why left padding.** It puts the last real token at column `-1` for *every* row, so
end-aligned indexing is a constant slice instead of a per-row `torch.gather` with
per-`(row, k)` validity masking. A new `LeftPadDataCollator` +
`make_left_padded_data_module` live locally in `mean.py`; `probe.py`'s shared
collator is deliberately not touched (see §8). Padding width is a single global max
(longest example per training call, i.e. per concept), optionally capped by
`max_seq_length`.

**No `position_ids` are passed** — the existing `gather_residual_activations` is
reused unmodified, relying on RoPE shift-invariance (§9).

**`PromptAdditionIntervention`** (`axbench/models/interventions.py`): same
broadcast-to-every-position add as the stock `AdditionIntervention`, but gated with
the same `base.shape[1] <= 1` decode-step no-op described below — so it steers every
*prompt* token during prefill and never touches generated tokens. Used by
`MeanTokenDiffMean` and (by inheritance) `LastTokenDiffMean`.

**`PositionwisePromptAdditionIntervention`** (`axbench/models/interventions.py`,
renamed from `PositionwiseAdditionIntervention`): weight
`[n_concepts, num_positions, hidden]` instead of the stock
`AdditionIntervention`'s `[n_concepts, hidden]`-broadcast-to-all-positions. The stack
is stored in **column order** (slot `-1` = final token, slot `-1-k` = k tokens back),
the same order `DiffMeanPositional.train` slices it out of the left-padded batch, so
inference just does `delta[:, -n:] = v[:, -n:]` with no flip. Checkpoints store it under
the dict key `positional_col`; older checkpoints used `positional` in the reverse order
(slot 0 = final token) and now raise on load rather than steering with reversed
vectors. It suppresses itself on
decode steps via `base.shape[1] <= 1` — stateless, so it stays correct across the
batch loop in `predict_steer`, unlike a call counter which would need resetting per
`generate()` call (a hook `Model.predict_steer` does not provide). Used by
`DiffMeanPositional`.

**Inference semantics: prompt-only, end-aligned.** Training indexes back from the
last real token of the instruction+response sequence; at inference the vectors are
applied end-aligned to the *prompt* and generation is left unsteered. *Accepted
caveat:* "the end" means end-of-response in training but end-of-prompt at test, so
vectors are applied to a somewhat different distribution than they were fit on.
Revisit if results look weak.

**Training semantics: padding is intentionally not masked out.** Position `k`'s
diff-of-means is computed over *every* example in the batch, not just those with a
real, non-prefix token `k` back — deliberate, per explicit user request over an
alternative (masking short examples out of deep positions) that was raised and
rejected as trading a narrower-but-clean estimate for a broader-but-noisier one. For
examples too short to reach position `k`, the contributing "activation" is whatever
the model produces at that column — left-padding or still inside the fixed
chat-template prefix, neither of which is concept-bearing content. This is a real,
acknowledged tradeoff, not an oversight: deep positions (large `k`, few long-enough
examples) are the most exposed, since a bigger share of their contributions there are
non-content. Revisit if deep-position vectors look degenerate or steering quality
drops off sharply with `k`.

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
source.

**Now run end-to-end on GPU** (9B/L20, own `pos_steer_data` dataset): train → latent →
steering all completed, producing 46,480 steering rows over 332 concepts × 10 prompts ×
14 factors for all three variants. Steering demonstrably works and scales with the
factor — a crude concept-keyword rate rises monotonically from a ~21% baseline to 40%
(`DiffMeanPositional`), 36% (`MeanTokenDiffMean`), 28% (`LastTokenDiffMean`). Inspecting
generations shows the expected over-steering arc: concept absent at low factors,
naturally integrated mid-sweep, and by factor 5.0 the model often abandons the
instruction entirely (e.g. refusing on a "gambling" concept) — which is exactly what the
LM judge's harmonic mean is meant to punish. Concept coverage is uneven: a substantial
share of concepts never visibly activate anywhere in the sweep.

Two checks are still **outstanding**: (a) `DiffMean` producing bit-identical weights
before/after these edits, and (b) `MeanTokenDiffMean` matching `DiffMean` within bf16
noise (cosine > ~0.999). Check (b) is the empirical test of the RoPE assumption in §9 —
if it fails, the left-padding premise is wrong, not just the code. Neither is settled by
the run above, which never trained stock `DiffMean`.

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

**On Slurm, use the `run_preemptable_*.sh` wrappers instead of the raw commands
above.** `run_preemptable.sh` holds `CFG`/`DUMP`/`NPROC`/`NUM_CHUNKS` and a
commented-out line per stage; uncomment one stage at a time, since ordering *between*
stages is still manual. Each stage script runs **both** of its modes from a single
call:

- `run_preemptable_generate.sh <CFG> <DUMP> <NUM_WORKERS> <MAX_CONCURRENT>` — training
  then latent, **locally in the calling shell, no sbatch**, ordered by `set -e`. These
  modes never touch the GPU (`generate.py` only moves a model to CUDA in
  `--mode dpo_training`), so they need no allocation.
- `run_preemptable_inference.sh <CFG> <DUMP> <NPROC> <NUM_CHUNKS>` — submits N latent
  chunk jobs, then N steering chunk jobs carrying `--dependency=afterok` on *every*
  latent job, so no steering chunk starts until the whole latent set has succeeded and
  merged. Each steering worker also runs `--mode latent --verify_chunks` first.
- `run_preemptable_evaluate_local.sh <CFG> <DUMP> <NUM_CHUNKS>` — same shape for the
  local judge: N `steering` chunks, then N `steering_test` chunks chained behind them.
  Here the chaining is *not* a data dependency (the two modes are independent); it
  just serialises GPU demand.

The chaining works by capturing ids from `sbatch --parsable` and passing them as
`--dependency=afterok:<id:id:...>`. A preempted-and-requeued job never reaches
`COMPLETED`, so dependents correctly keep waiting through preemption cycles; a genuine
failure leaves them unsatisfiable, and this cluster's `kill_invalid_depend` cancels
them rather than leaving them pending forever. A pending job showing `(BeginTime)` is
just post-preemption backoff, not a problem.

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
- **Steering needs latent mode to have run first** — *no longer silent, but still a
  trap.* `inference.py` calls `pre_compute_mean_activations` before steering, which
  reads `{ClassName}_max_act` from `latent_data.parquet`. This used to fall back to a
  factor scale of `1.0` without complaint; it now raises (missing file →
  `FileNotFoundError` in `model.py`; missing concepts → `ValueError` in
  `inference.py`). Any new method still needs a working latent path, not just a
  steering one. What remains quiet is one layer up: the latent auto-merge catches
  every exception and only warns, so latent chunks can all exit 0 with no merged
  parquet, and the failure only surfaces once steering jobs have already been
  scheduled onto GPUs.
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
