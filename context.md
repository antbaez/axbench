# context.md — behavior-relevant changes on `pos-steer-data`

Reference doc for what actually changed in *pipeline behavior* on this branch
(vs. `origin/main` @ `41c8332`) — what gets generated, trained, steered, or
scored, and how. Excludes pure job-orchestration/scripting glue (`run_preemptable*.sh`,
sbatch flags, log-level cosmetics) and anything already covered by `CLAUDE.md`.

**See `CLAUDE.md` first** for the DiffMean-variant work itself: `MeanTokenDiffMean`
/ `LastTokenDiffMean` / `DiffMeanPositional` in `axbench/models/mean.py`, left-padding,
`PositionwiseAdditionIntervention` (`axbench/models/interventions.py`), the
weight/bias dict save-load scheme, `num_positions`/`max_seq_length` config plumbing
in `training_args.py`, the `diffmean_variants_l20.yaml` sweep, the silent-failure
traps (§8), and the verified RoPE/`intervene_on_prompt` internals (§9). Confirmed
against the current diff — nothing there has changed since CLAUDE.md was written.
This doc covers everything else.

## 1. Latent-mode eval data generation changed shape entirely

`axbench/scripts/generate.py:219` (`create_data_latent`) switched from calling
`DatasetFactory.create_eval_df()` to `DatasetFactory.create_train_df(concept,
num_of_examples, concept_genres_map, output_length=..., split="test")`
(`axbench/utils/dataset.py:442`) — the *same* generator training data now uses,
just pointed at the held-out `"test"` seed-instruction split instead of `"train"`.

Resulting schema change to `latent_eval_data.parquet` (columns
`input, output, output_concept, concept_genre, category, dataset_category[, contrast_concept]`):

| Column | Before (`create_eval_df`) | After (`create_train_df`, `split="test"`) |
|---|---|---|
| `category` | 3 values: `positive` / `negative` / `hard negative` | 2 values only — hard-negative rows are gone |
| `output` | Real LLM-generated response for every row | **Always `""`** for instruction-category data (both positive and negative rows) — no response generated at all |
| `output_concept` | Concept string for positive/negative; `"//"`-joined polysemantic list for hard-negative | Concept string for positive; sentinel `EMPTY_CONCEPT` (`"EEEEE"`, `axbench/utils/constants.py:43`) for negative |
| `concept_genre` | Negatives spread text/math/code (70/15/15) across the whole concept set | Always the concept's own genre, for both positive and negative |
| `contrast_concept` | *(doesn't exist)* | **New**: `""` for positive, one sampled contrastive concept for negative |

This applies to `axbench_data`'s configured `dataset_category: "instruction"`
(`axbench/sweep/antbaez/diffmean_variants_l20.yaml:8`). For `dataset_category:
"continuation"`, `create_train_df` still generates a real response via
`continue_with_concept`, but only for positive rows — there's no continuation
negative branch in the new function at all (not exercised by the current sweep).

**New negative-generation mechanism** (`dataset.py:442-514`): instead of a
separately-sampled, genre-balanced negative pool, negatives are now created by
minimally editing each positive *instruction* itself to relate to a different,
related concept — via three new async helpers in
`axbench/utils/prompt_utils.py`:
- `instruction_with_concept` — rewrites a seed instruction to incorporate the target concept (prompt: `T_INSTRUCTION_WITH_CONCEPT`, `axbench/templates/prompt_templates.py:251`).
- `get_contrastive_concepts` — one LLM call per concept, returns 10 related-but-distinct contrast concepts (`T_GENERATE_CONTRASTIVE_CONCEPTS`), generated once and sampled from per-example rather than invented fresh each time, so negatives don't all converge on the same nearest neighbor.
- `instruction_with_related_concept` — minimally edits a concept-laden instruction to swap in a sampled contrast concept instead (`T_INSTRUCTION_WITH_RELATED_CONCEPT`).

**The old shared negative pool is disabled, not deleted.** `DatasetFactory.__init__`
(`dataset.py:184-246`) used to pre-generate a genre-balanced pool of negatives via
the local base LM (`get_model_continues`) at `concept_id == 0`. That block is now
commented out and `self.negative_df` is set to an always-empty placeholder
DataFrame — kept only because `generate.py`'s `save()` still reads
`dataset_factory.negative_df` unconditionally at `concept_id == 0`. Consequence in
`generate.py:412-420`: the local base LM (`AutoModelForCausalLM.from_pretrained(...)`)
is no longer loaded onto GPU for `--mode training` at all (`model = None`) — nothing
in the new instruction path calls it; only the `gpt-4o-mini`-style API client
(`self.lm_model`) generates anything now.

**Non-text concepts are now skipped during training generation.**
`generate.py:449-453` (`generate_training`): after computing `concept_genres_map`,
if `genre != "text"` the concept is skipped entirely (state advanced, `continue`) —
math/code-genre concepts no longer get training data generated for them at all.

**Minor:** `generate.py`'s `save()` (`generate.py:185-188`) now also writes a
`.json` sibling of every parquet for human inspection — no behavior change,
convenience only.

## 2. Training text-construction bug (from #1) and the fix

Because `output` is now always `""` for instruction-category data, `train.py`'s
`prepare_df` (`axbench/scripts/train.py:104`) building `{"role": "assistant",
"content": row["output"]}` alongside `add_generation_prompt=True` caused Gemma's
chat template to render a **second, spurious generation-prompt header** after the
empty assistant turn's own wrapper. Empirically confirmed tail:
`...<end_of_turn>\n<start_of_turn>model\n<end_of_turn>\n<start_of_turn>...` —
identical boilerplate across every example. `get_suffix_length()`
(`axbench/utils/model_utils.py`) measures its trim length from a bare user-only
probe message with no assistant turn and no generation prompt (2 tokens,
`<end_of_turn>\n`) — the wrong shape for what `prepare_df` actually built, so
`[1:-suffix_length]` under-trimmed and left that boilerplate in the pooled text.

Impact was worst for `LastTokenDiffMean`/`DiffMeanPositional` (index back from
"the end," which was a constant template artifact, not real prompt content) and
minor for `DiffMean`/`MeanTokenDiffMean` (mean-pools over many tokens; boilerplate
is a small dilution).

**Fix** (`train.py:122-149`, both the `HAS_SYSTEM_PROMPT_MODELS` and plain
chat-model branches): dropped the assistant message from `messages` entirely —
now just `[system?, user: input]` + `add_generation_prompt=True` — and changed
the slice from `[1:-suffix_length]` to `[1:]` (no assistant turn means nothing to
trim off the tail). Applies to *every* model trained with `binarize_dataset: true`
on a chat model, not just the three DiffMean variants. This also makes train-time
"end of sequence" coincide with inference-time "end of prompt, about to generate"
(both now end on the same generation-prompt-header state), which resolves what
CLAUDE.md §3 called an "accepted caveat" about train/test distribution mismatch
for `LastTokenDiffMean`/`DiffMeanPositional`.

**Related fix in the same function** (`train.py:116-117`): negatives used to come
from a shared cross-concept `negative_df` argument filtered only by
`concept_genre == genre` (any concept's negative sharing the genre). Now (matching
§1's per-concept negative-generation change), negatives are `original_df[(output_concept
== EMPTY_CONCEPT) & (category == "negative")]` — each concept's *own* paired
minimal-edit negatives, no longer a genre-wide shared pool. `negative_df` is no
longer a separate parameter to `prepare_df` at all (removed from its signature and
from the `main()` call site, `train.py:96-99` / `486`).

## 3. Auto-merge-on-completion for chunked jobs

Previously, `--chunk`/`--num_chunks` sweeps in `inference.py` and `evaluate_local.py`
required a separate manual `--merge_chunks` (and, for `inference.py`,
`--verify_chunks`) invocation after every chunk job finished.

**`axbench/scripts/inference.py`** (new: `chunk_tag`, `chunk_bounds`,
`_chunk_set_complete`, `maybe_auto_merge_chunks`, `merge_chunks`, `verify_chunks` —
all added, ~lines 57-1112 region): `infer_latent`/`infer_steering` now check, right
after their own chunk's rank-0 `dist.barrier()`, whether every concept the dataset
expects now has a row in *some* `rank_*_{mode}_chunk*_data.parquet` file on disk. If
so, that job merges everything itself into `{mode}_data.parquet` — the last chunk to
finish does the merge, no separate step needed. `--merge_chunks`/`--verify_chunks`
CLI flags still work manually. Also fixed in the same pass: latent mode's
`save_state` used to write to the dump root instead of the `inference` subfolder
that `load_state` reads from, so a preempted latent run could never actually resume
(`inference.py:722` comment, `save_state(dump_dir, ...)` now matches `load_state`'s dir).

**`axbench/scripts/evaluate_local.py`** (`_chunk_set_complete:675`,
`maybe_auto_merge_chunks:694`, `merge_chunks:716`): same pattern, called from
`eval_steering()` (`:406`) right after a chunk finishes. Because `evaluate_local.py`'s
`merge_chunks` also interactively prompts (`input(...)`) to offer deleting
now-superseded stale per-chunk files, it gained an `interactive=True` parameter
(default preserves the old manual `--merge_chunks` CLI behavior with the prompt);
the auto-triggered path calls it with `interactive=False`, which skips the prompt
(closed stdin under a non-interactive `sbatch` job would otherwise raise `EOFError`)
and just logs that stale files are left in place.

## 4. `pre_compute_mean_activations` glob bug fix

`axbench/models/model.py:410` (`Model.pre_compute_mean_activations`): the scan for
merged latent results used to match any file `startswith("latent_")` — which also
matches `latent_eval_data.parquet` (generate.py's own eval-data output, no
`{model}_max_act` columns) whenever `--overwrite_inference_data_dir` points at the
same directory. Narrowed to an exact-filename check (`file == "latent_data.parquet"`)
so it can no longer accidentally read the wrong file as if it were the merged
inference-results parquet.

## 5. Other changes checked, no material behavior impact

- **`axbench/models/model.py` / `mean.py`**: `torch.load(..., weights_only=True)`
  added to weight/bias loads (`model.py:100-154`) — safety/deprecation-warning fix
  per newer PyTorch defaults, not a behavior change. Also removed the
  `print(f"Loading {model_name} from {dump_dir}.")` debug line from both
  `model.py`'s `load()` and `mean.py`'s `DiffMeanPositional.load()`.
- **`axbench/models/interventions.py`**: `PositionwiseAdditionIntervention` matches
  CLAUDE.md §3/§9's description exactly — no undocumented behavior found.
- **`axbench/models/language_models.py:158`**: `LanguageModel.chat_completions`'s
  default `batch_size` doubled from `32` to `64` — throughput/concurrency change
  for LLM API calls during generation, not a correctness change.
- **Logging cosmetics** (not behavior): many `logger.warning` → `logger.info` swaps
  across `generate.py`, `dataset.py`, `inference.py`, `evaluate_local.py` for
  routine progress messages (model-loading notices, "using pre-generated data",
  etc.) so they no longer print at the default `WARNING` level. `train.py` also
  gained `logger.propagate = False` to stop the same message double-printing via
  the root logger's own handler (set up separately by `model.py`'s
  `logging.basicConfig(...)` at import time) in two different formats.
- **`training_args.py`**: `num_positions`/`max_seq_length` plumbing — already
  documented in `CLAUDE.md` §3/§8, unchanged since.
- **`axbench/sweep/antbaez/diffmean_variants_l20.yaml`**: new sweep file for this
  work; settings (`dataset_category: "instruction"`, model name
  `google/gemma-2-9b-it`, layer 20, `steering_batch_size: 5`) already covered by
  CLAUDE.md §3/§7's sweep-yaml notes.
