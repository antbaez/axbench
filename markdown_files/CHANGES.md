# CHANGES

High-level summary of how the `pos-steer-data` branch changes the AxBench pipeline.

## Goals

1. **Steering vectors from prompt activations only.** Fit directions on the prompt, not on
   the prompt plus a generated response.
2. **Minimal-pair positives and negatives.** A negative should differ from its positive only
   in which concept it names, so the difference of means isolates the concept rather than
   style, length, or which model wrote the text.
3. **Eval prompts drawn from the training distribution.** Latent and steering evaluation use
   prompts built the same way, from the same material, as the training data.

Evaluation is deliberately **in-sample**: latent and steering prompts come from the same
instruction pool training uses. Treat the numbers as a ceiling/diagnostic, not a
generalization estimate, and not as comparable to the AxBench leaderboard.

## 1. Prompt construction

**Positives and negatives.** A positive is a base instruction rewritten to incorporate the
target concept. A negative is that *positive* minimally rewritten to swap the target concept
for an **unrelated** contrast concept. Deriving the negative from the positive (rather than
from the bare instruction) is what keeps each pair a true minimal pair. No responses are
generated anywhere; every row is a prompt only.

Contrast concepts are unrelated rather than near-neighbors. Near-neighbors leave the rewritten
text sitting in the target concept's own domain, which has two consequences: the difference of
means captures only the narrow gap between neighbors instead of the concept itself, and a
steering prompt built that way still reads as on-concept, so a steered generation is
indistinguishable from an unsteered one.

**Two shared stores**, written once into `$DUMP/generate/` and read by every stage:
- `base_instructions.json` — one pool of instructions (`num_base_instructions`, default 72)
  sampled from `text_train`, shared by every concept and every stage.
- `contrastive_concepts.json` — for each concept, 10 unrelated concepts from a single
  gpt-4o-mini call, each written in the same format as the concept it was generated from
  (same phrasing pattern, structure, length, and specificity) so swapping one in stays a
  minimal edit.

Previously each stage re-sampled its own seed instructions and regenerated its own contrast
concepts at temperature 1.0, so training and eval silently used different negatives. The
stores remove that drift.

**One builder.** `create_train_df` produces prompts for all three stages, taking the base
instructions and contrast concepts as inputs rather than generating them itself. It can emit
positives, negatives, or both.

**Contrast-concept assignment** walks a reshuffled copy of the 10, cycling as needed — random
order with even coverage, instead of independent draws that can leave some unused.

**Text-only concepts.** Non-text concepts (code, math) are skipped at generation time, since
every concept shares one text-genre instruction pool. Roughly 69% of concepts survive.

**Prompts are frozen once generated.** Rerunning a stage reuses its prompts; delete the
corresponding file to regenerate. The LLM response cache is off everywhere — its key includes
a per-run call counter, so it can't guarantee reproducibility, and a partial hit would give a
half-stale dataset. Persisting the generated prompts gives reproducibility directly.

## 2. Where each stage's prompts come from

| stage | prompts | persisted to |
|---|---|---|
| `generate.py --mode training` | every base instruction → positive + negative | `generate/train_data.parquet` |
| `generate.py --mode latent` | random subset (`latent_num_of_examples`) → positive + negative | `generate/latent_eval_data.parquet` (formerly `inference/`, still read as a fallback) |
| `inference.py --mode steering` | random subset (`steering_num_of_examples`) → **negatives only** | `inference/steering_eval_data.parquet` |

Steering uses negatives only: a positive already names the concept, so the model would discuss
it with no intervention and the judge would score the prompt rather than the vector.

**Steering's contrast concepts are held out from training's.** A steering prompt's contrast
concept is drawn from the pool of *other* concepts' stored 10, never from the target concept's
own 10. Those 10 are what its training negatives were built from — the direction was fit to not
respond to them — so evaluating on them would test material the vector was already tuned
against. With a few hundred concepts in the store, the held-out pool runs to a few thousand
candidates.

Steering previously sampled prompts from AlpacaEval, which had no relationship to the training
data. It now uses `steering_datasets: ["ContrastInstructions"]`, generated inside
`inference.py` on first run and reused thereafter.

## 3. Training

- **Prompt-only chat template.** The assistant turn was dropped when building training text,
  so pooled activations end at the generation-prompt header — the same format steering applies
  the vector to. This also fixed a spurious extra template header that was leaking boilerplate
  into the pooled tokens.
- **Paired negatives.** Negatives come from the concept's own minimal pairs rather than a
  genre-wide pool shared across all concepts.
- The old shared negative pool (local Gemma continuations) and the local base-model load in
  generation were disabled; all generation now goes through gpt-4o-mini.

## 4. Inference

- **Latent reads generated data.** `--overwrite_inference_data_dir` must point at
  `$DUMP/inference`; otherwise `inference.py` silently regenerates eval data live with the old
  response-plus-hard-negative schema.
- **Steering calibration file.** `pre_compute_mean_activations` now reads only
  `latent_data.parquet`. Its old `latent_*` prefix glob also matched `latent_eval_data.parquet`,
  which now lives in the same directory, and crashed steering.
- **Missing calibration is now an error, not a silent rescale.** Steering multiplies every
  factor by a concept's `max_act`, and `predict_steer` falls back to `1.0` per row on a lookup
  miss. With unit-norm directions that makes the whole sweep 1–2 orders of magnitude too weak
  to do anything, while still producing fluent output — so the run looks fine and the only
  symptom is generations that don't vary with the factor. Two guards now cover it:
  `pre_compute_mean_activations` raises if `latent_data.parquet` is absent, and steering raises
  if the file is present but missing any concept it is about to steer (the partial case, from
  starting steering before every latent chunk is merged).
- **Chunked, preemptable runs.** Concepts split into disjoint chunks run as independent jobs;
  the last chunk to finish merges them automatically. Merge and verify also run standalone
  without a GPU.

## 5. Evaluation

- Steering evaluation can use a local vLLM Llama-3.1-70B judge instead of the OpenAI API,
  sharded across preemptable jobs with automatic merging.
- The local judge dumps a few complete examples — rendered prompt, raw judge response,
  parsed rating — once per process, so a run can be spot-checked. Malformed judge output
  silently scores 0, so seeing the parse next to the response matters.
- `evaluate_short.py` is a subsampled variant of `evaluate.py` for quick, cheap passes.
  It reuses the same judge and client setup, so its numbers match; what differs is that
  the factors, examples per concept, and concept count are all configurable, judging runs
  across worker threads, and results are written as JSON plus a JSONL of every individual
  rating rather than a parquet. It ends by printing a factor × method table and each
  method's best factor, and needs no GPU.

## 6. Job orchestration

- `run_preemptable.sh` is a flat, top-to-bottom list of pipeline stages, each independently
  commentable. It holds every parameter; the stage scripts take them as arguments rather
  than hardcoding their own, which previously let the dump directory drift out of sync.
- Each stage now runs **both** of its modes from one call, in the right order, so there is
  no manual waiting between the two halves of a stage.
- For the chunked stages, ordering is enforced by Slurm dependencies: the second mode's
  chunks are submitted held, and released only once *every* chunk of the first mode has
  succeeded. This matters most for inference, where steering reads a calibration file that
  only exists after all latent chunks have merged. Preemption doesn't trip the barrier, and
  a real failure cancels the dependents instead of running them on incomplete input.
- Data generation runs locally rather than as a batch job, since those modes never use the
  GPU and were previously holding an accelerator idle for the whole run.
- Whether a stage script dispatches or works is decided by argument presence rather than
  `SLURM_JOB_ID`, and each locates itself by a fixed path, so it can be called from inside
  another running job.

## 7. Bug fixes

- **Concept-id gaps on resume.** Skipping non-text concepts made the data index diverge from
  the loop index, so resuming after a preemption could leave gaps in `concept_id` and misalign
  weights and calibration downstream. The data index is now derived from what has already been
  written.
- **Latent couldn't resume.** Its state was saved to a different directory than it was loaded
  from.
- **New config keys silently ignored.** Keys missing from the args dataclass are dropped with
  no warning; `num_base_instructions` was added there explicitly.
- **Cost logging tied to caching.** Cost reports only wrote when caching was on; now always.
- **Missing steering calibration failed silently.** Steering scales every factor by a
  per-concept value produced during latent evaluation. If that file or a concept within it
  was missing, every factor quietly fell back to a scale of 1, changing what the whole sweep
  meant with no error. It now fails loudly, naming the concepts it can't cover, and the
  calibration file is read by exact name so a similarly-named input file can't be mistaken
  for it.

## 8. Regenerating

Run in order, since each stage reads the previous one's output:

1. `generate.py --mode training` — creates both stores, `metadata.jsonl`, training data
2. `generate.py --mode latent`
3. `train.py`
4. `inference.py --mode latent` (with `--overwrite_inference_data_dir`)
5. `inference.py --mode steering` — generates and persists steering prompts
6. evaluation

To get fresh prompts for one stage, delete its persisted file (§2) and its state file. To
change the instruction pool or contrast concepts, delete the corresponding store — which means
regenerating everything downstream of it.
