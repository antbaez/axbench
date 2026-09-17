# CHANGES

High-level summary of how the `pos-steer-data` branch changes the AxBench pipeline.

## Goals

1. **Steering vectors from prompt activations only.** Fit directions on the prompt, not on
   the prompt plus a generated response.
2. **Minimal-pair positives and negatives.** A negative should differ from its positive only
   in the concept, so the difference of means isolates the concept rather than topic, style,
   or which model wrote the text.
3. **Eval prompts drawn from the training distribution.** Latent and steering evaluation use
   prompts built the same way, from the same material, as the training data.

Evaluation is deliberately **in-sample**: latent and steering prompts come from the same
instruction pool training uses. Treat the numbers as a ceiling/diagnostic, not a
generalization estimate, and not as comparable to the AxBench leaderboard.

## 1. Prompt construction

**Positives and negatives.** A positive is a base instruction rewritten to incorporate the
target concept. A negative is that *positive* minimally rewritten to swap the target concept
for a related-but-distinct contrast concept. Deriving the negative from the positive (rather
than from the bare instruction) is what keeps each pair a true minimal pair. No responses are
generated anywhere; every row is a prompt only.

**Two shared stores**, written once into `$DUMP/generate/` and read by every stage:
- `base_instructions.json` — one pool of instructions (`num_base_instructions`, default 72)
  sampled from `text_train`, shared by every concept and every stage.
- `contrastive_concepts.json` — for each concept, 10 related-but-distinct concepts from a
  single gpt-4o-mini call.

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
| `generate.py --mode latent` | random subset (`latent_num_of_examples`) → positive + negative | `inference/latent_eval_data.parquet` |
| `inference.py --mode steering` | random subset (`steering_num_of_examples`) → **negatives only** | `inference/steering_eval_data.parquet` |

Steering uses negatives only: a positive already names the concept, so the model would discuss
it with no intervention and the judge would score the prompt rather than the vector.

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
- **Steering calibration file.** `pre_compute_mean_activations` now matches only
  `latent_data.parquet`. Its old `latent_*` prefix glob also matched `latent_eval_data.parquet`,
  which now lives in the same directory, and crashed steering.
- **Chunked, preemptable runs.** Concepts split into disjoint chunks run as independent jobs;
  the last chunk to finish merges them automatically. Merge and verify also run standalone
  without a GPU.

## 5. Evaluation

- Steering evaluation can use a local vLLM Llama-3.1-70B judge instead of the OpenAI API,
  sharded across preemptable jobs with automatic merging.

## 6. Job orchestration

- `run_preemptable.sh` is a flat, top-to-bottom list of pipeline stages, each independently
  commentable.
- `run_preemptable_inference.sh <latent|steering>` dispatches one job per chunk. Whether it
  dispatches or works is decided by argument presence rather than `SLURM_JOB_ID`, and it
  locates itself by a fixed path, so it can be called from inside another running job.
- `run_preemptable_evaluate_local.sh` does the same for the local judge.

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
