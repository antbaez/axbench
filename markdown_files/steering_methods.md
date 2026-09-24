# How the DiffMean steering variants work

This covers `MeanTokenDiffMean`, `LastTokenDiffMean` and `DiffMeanPositional`. The code is in `axbench/models/mean.py`, with the two prompt-only interventions in `axbench/models/interventions.py`. It focuses on the implementation and design choices that affect results.

## What all three share

**Training text is the prompt only.** `prepare_df` in `train.py` (committed code) builds each training example as the chat-templated user turn with `add_generation_prompt=True`. There is no assistant turn. So every training sequence ends in `<end_of_turn>\n<start_of_turn>model\n`, which is exactly how the steering prompts end in `dataset.py:852-861`. Training and test are now aligned at the end of the sequence.

This means the §3 caveat in CLAUDE.md is out of date. It says training runs to the end of the response while inference runs to the end of the prompt, and that's no longer true.

**Negatives are paired minimal edits** of each concept's own positives, not a pool drawn from other concepts. With `num_of_examples: 144` that's about 72 positives and 72 negatives per concept. That number matters most for `LastTokenDiffMean`, covered below.

**Left padding** (`LeftPadDataCollator`) puts the last real token in column -1 for every row. The attention mask is built from each row's real length, not by comparing against the pad id, so it avoids the `probe.py` problem where a real token equal to the pad id gets masked. Padding width is the longest example in each concept. Nothing is capped unless you set `max_seq_length`, and if you do, it cuts from the left so the end survives.

**All three steer only during prefill.** Each one overrides `make_model()` and ignores `steering_intervention_type: "addition"` in the yaml. The intervention checks `base.shape[1] <= 1` and does nothing on decode steps, so generated tokens are never pushed directly. The only way steering reaches the output is through the prompt's cached keys and values from layer 20 upward. This is probably the biggest single factor in performance:

- These methods will be weaker at any given factor than stock `DiffMean`, which pushes every generated token.
- The effect probably fades as the generation gets longer.
- If factor 5.0 still looks under-steered, the factor range may need to go higher. Don't read weak steering as a weak direction until you've ruled that out.

**Magnitude is `factor × max_act`**, taken from latent mode (`model.py:402`). There are two catches:

- **`max_act` is a per-concept maximum, so a factor isn't a fixed strength.** Latent mode uses `inference.py`'s own `prepare_df`, which adds an assistant turn and then removes the template suffix. In this dataset every latent row has an empty `output`, so the latent text ends in `<start_of_turn>model\n`, exactly like the training text (checked against the Gemma tokenizer). The calibration is on the right kind of text. Still, `max_act` is the largest projection over any token of any latent row for that concept, so it varies by concept and by method. Compare methods at their own best factor.
- **A fallback of 50.** If a concept's `max_act` is zero or negative, it's replaced with 50 without any warning. That's an arbitrary strength, so those concepts can look badly broken.

The steering vector is also added to every column the intervention covers, including BOS and the chat-template header. Stock `AdditionIntervention` does the same, so that's in line with the baseline.

## MeanTokenDiffMean

**Training.** It averages every real token after the chat-template prefix (`_real_token_mask`) and takes positive mean minus negative mean. That's standard DiffMean on the same data, so it should match stock `DiffMean` within bf16 noise. That check still hasn't been run.

Now that training is prompt-only, the pool includes the five generation-prompt tokens (`<end_of_turn>\n<start_of_turn>model\n`). They're the same in every example, so their share of the average depends on how long the instruction is.

Longer examples contribute more tokens, so they carry more weight. That's the same weighting as `DiffMean`, but it differs from `GemmaScopeSAEDiffMean`, which gives each example equal weight.

**Steering.** `PromptAdditionIntervention` adds the same unit vector, times `factor × max_act`, to every prompt token.

**What affects performance.** This has the most data behind it (thousands of token vectors per class), so it's the lowest-noise estimate. The weakness is prefill-only steering, described above.

## LastTokenDiffMean

**Training.** It takes `activations[:, -1, :]` for each example and computes the difference of means. Because of the prompt-only format, the last token is always the `\n` after `model`, the point where the model is about to start its answer. That's arguably the most useful single position, since it summarizes the whole prompt. The `valid` guard (`attention_mask.sum > prefix_length`) always passes now.

**Steering.** It inherits `PromptAdditionIntervention`, so the vector is added to every prompt token, not just the last one. The docstring says this is deliberate, to distinguish it from `DiffMeanPositional(num_positions=1)`.

**What affects performance:**

- **It's noisy.** Each class mean comes from only about 72 vectors in 3584 dimensions, compared with thousands for MeanToken. Expect more noise from one concept to the next.
- **The vector is used at positions it wasn't fit on.** It was fit on the activation at one template token and then added to every other prompt token. It may be off-distribution at the rest of the prompt.

The earlier run fits this: its keyword rate reached 28%, against 36% for MeanToken and 40% for Positional.

## DiffMeanPositional

**Training.** With `num_positions: 64`, position k takes column `seq_len-1-k` across the batch and computes a separate difference of means for each k. Sums and counts are kept in fp32. Each of the 64 vectors is normalized to unit length on its own.

**Steering.** `PositionwisePromptAdditionIntervention` adds the stack to the last prompt columns in the order it was stored: the final slot goes on the final prompt token, and the vector for k tokens back (`v_k`) lands k columns earlier. It covers `min(64, batch width)` columns, and every column gets the same `factor × max_act`.

**Latent mode.** Latent mode needs a single vector, so training also stores a collapsed direction: the mean of the 64 unit vectors, normalized again. That collapsed direction sets `max_act` for all 64 positions. None of the positions has its own scale.

Weights are saved as a dict (`collapsed` and `positional`) so they survive the per-rank merge. At load time, `num_positions` comes from the checkpoint shape, not the config.

**Design decisions that affect performance:**

1. **Padding is deliberately not masked out.** This was an explicit choice. Deep positions average over every row, including rows where column k is padding or still inside `<bos><start_of_turn>user\n`. A pad column only attends to other pads, so its activation carries no content. That has two effects:
   - **Dilution.** The concept signal at deep k shrinks toward zero.
   - **A length artifact.** If positives and negatives differ in length, deep positions pick up a "real token vs. pad" direction instead of the concept. Paired minimal-edit negatives should keep the lengths close, which is the main thing protecting against this.

2. **Every position gets the same strength.** Each vector is unit length and scaled equally, so a noisy `v_60` gets the same push as a clean `v_0`. Mostly-noise vectors can end up applied at full strength.

3. **Short prompts get steering on the template prefix and BOS.** If a prompt is shorter than 64 tokens, the deep vectors land on `<start_of_turn>user\n` and on `<bos>`. Pad columns are harmless because nothing attends to them. BOS is different: it's an attention sink, so a large addition there at layer 20 can affect everything after it. Short instructions will be affected most.

4. **The calibration comes from a mixed direction.** Magnitude is calibrated on the collapsed mean direction, not on the vectors that are actually added. If the 64 vectors disagree, the collapsed direction may not represent any of them well, and the scale follows.

Being 64 tokens deep means it steers only the last 64 prompt tokens and none of the generation, so its total push is the smallest of the three. Even so, it scored best on the keyword proxy (40%). That suggests the near-end positions are doing most of the work.

## DiffMeanPositional, step by step: building and applying the vectors

The whole method depends on one idea: **count positions backward from the end of the sequence, and do it the same way in training and at inference.** Position `k` always means "k tokens before the last token," never "the k-th token." Everything below is there to keep that true.

The worked example uses `num_positions = 4` to keep it small. The real config uses 64, and the mechanics are identical. The token strings are illustrative Gemma chat-template tokens.

### Step 1: Build the training text

`train.py:prepare_df` runs every example through the chat template with `add_generation_prompt=True` and no assistant turn, removes the template's BOS, and decodes it back to text. When `make_left_padded_data_module` tokenizes that text, the tokenizer adds a single BOS again. So each training sequence looks like this:

```
<bos> <start_of_turn> user \n  ...instruction tokens...  <end_of_turn> \n <start_of_turn> model \n
```

Every example, positive or negative, ends in the same five tokens: `<end_of_turn> \n <start_of_turn> model \n`. That fixed tail is what makes backward counting line up across examples.

### Step 2: Left-pad so the end is always column -1

`make_left_padded_data_module` finds the longest example for the concept (width `W`). `LeftPadDataCollator` then puts pad tokens on the **left** of every shorter row:

```
column:         0     1     2    ...   W-5            W-4   W-3             W-2    W-1
long example:   <bos> <sot> user ...   <end_of_turn>  \n    <start_of_turn> model  \n
short example:  <pad> <pad> <pad> ...  <end_of_turn>  \n    <start_of_turn> model  \n
```

Now the last token (`\n`) is in column `W-1` for **every** row, the one before it in `W-2`, and so on. "k back from the end" becomes a single column, `W-1-k`, for the whole batch, so there's no per-row indexing.

Left padding doesn't change the activations of real tokens. Gemma-2 uses RoPE, which only depends on relative distance between tokens, and real tokens give zero attention weight to pad columns.

### Step 3: Compute one difference of means per position

After running the model up to layer 20 (`gather_residual_activations`), the training loop takes the last `num_positions` columns in a single slice, keeping them in their left-to-right order:

```python
n = min(num_positions, W)                  # W < 64 only if the concept's longest prompt is short
acts = activations[:, -n:, :]              # [batch, n, hidden]: last n columns, left to right
positive_sum[-n:] += acts[labels == 1].sum(0)
negative_sum[-n:] += acts[labels != 1].sum(0)
```

Then `weight[j] = mean_pos[j] - mean_neg[j]`, and each row is normalized to unit length. The result is a `[64, hidden]` matrix stored in **column order**, the same left-to-right order the tokens appear in. The last slot is the last token:

```
weight[-1]  = v_0: direction for the last token     (\n after "model")
weight[-2]  = v_1: direction for 1 back             ("model")
weight[-3]  = v_2: direction for 2 back             (<start_of_turn>)
weight[-4]  = v_3: direction for 3 back             (\n)
weight[-5]  = v_4: direction for 4 back             (<end_of_turn>)
weight[:-5] = directions reaching back into the instruction itself
```

In general, the vector for k tokens back (`v_k`) is in slot `-1-k`, and it comes from training column `W-1-k`. If `W < 64`, the slice fills only the last `W` slots. The leading slots stay zero, and the `empty` warning fires (although its message says there are no examples at all).

`v_0`–`v_4` always come from the same template tokens in every example, so they are clean estimates of "how does the model's state at this template slot differ when the prompt carries the concept." From `k = 5` on, the positions fall inside the instruction. At larger `k`, short examples contribute prefix or pad columns (see design decision 1 above).

### Step 4: Save and merge

`save()` writes `{"positional_col": weight.unsqueeze(0), ...}`, which has shape `[1, 64, hidden]` for one concept. Each rank appends its concepts along dim 0. Rank 0 then concatenates the rank files per dict key, so the final checkpoint holds `positional_col: [n_concepts, 64, hidden]`. The column order inside each concept's `[64, hidden]` block is never changed.

At load time (`mode="steering"`), `num_positions` is read from `shape[1]` of the checkpoint and the tensor goes directly into `PositionwisePromptAdditionIntervention.proj_weight`. Train and inference can't disagree about the number of positions.

Older checkpoints stored the stack under `positional` in the reverse order (slot 0 = last token). `load()` refuses them with a `ValueError` instead of steering with reversed vectors.

Concept alignment works the same way as for every other AxBench method: row `i` of the merged tensor must be `concept_id == i`. The forward pass picks each prompt's stack with `proj_weight[subspaces["idx"]]`, where `idx` is that row's `concept_id`. A single steering batch can mix concepts, and each row still gets its own concept's vectors.

### Step 5: Build the steering prompt the same way

The steering prompts in `dataset.py:852-861` go through the same `apply_chat_template(..., add_generation_prompt=True)[1:]`, so they end in exactly the same five template tokens as the training text. `Model.predict_steer` sets `tokenizer.padding_side = "left"` before tokenizing the batch. The last prompt token is therefore in column `T-1` for every row, where `T` is the batch width, just as in training.

### Step 6: Apply the stack to the last columns, in the same order

The stack is already in column order, so the forward pass in `interventions.py` just lines up the end of the stack with the end of the prompt:

```python
v = self.proj_weight[subspaces["idx"]]      # [bs, 64, hidden], v[:, -1] = last-token vector
n = min(self.num_positions, base.shape[1])  # can't steer more columns than exist
steering_vec = v[:, -n:] * scale
delta = torch.zeros_like(base)
delta[:, -n:] = steering_vec
return base + delta
```

With `num_positions = 4`:

```
stored slots            [  0     1     2     3  ]
holds                   [ v_3   v_2   v_1   v_0 ]
training columns          W-4   W-3   W-2   W-1      where each was fit
inference columns         T-4   T-3   T-2   T-1      where each is added
tokens                    \n    <sot> model \n
```

Slot `-1-k` is read from training column `W-1-k` and written to inference column `T-1-k`. Both are "k back from the end," so `v_0` lands on the same `\n` after `model` it was fit on, and `v_k` in general lands on the same slot it was averaged over in step 3. Nothing is reordered anywhere.

Both sides slice from the end, and that's what keeps them aligned when the widths differ. If the steering batch is narrower than 64 columns (`T < 64`), `v[:, -n:]` keeps the `n` nearest vectors and drops the deepest ones, so `v_0` still lands on the last token.

`scale` is `factor × max_act` for that row. It's the same for every position, and each `v_k` is unit length.

### Step 7: Apply it only once, during prefill

`generate()` calls the intervention on every forward pass, and pyvene ignores `intervene_on_prompt` because `unit_locations=None`. The intervention therefore checks the sequence length itself:

```python
if base.shape[1] <= 1:
    return base   # decode step
```

- **Prefill** (the whole prompt in one pass, `shape[1] = T`): the delta above is added once. Layer 20's output at those prompt positions is now steered, and so are the keys and values that later layers compute from it and store in the KV cache.
- **Each decode step** (`shape[1] = 1` because of the KV cache): the check returns early, so generated tokens are never shifted. They're influenced only by attending back to the steered prompt positions in the cache.

The check has no internal state, so it works the same across the batch loop in `predict_steer` with no reset between `generate()` calls.

### What keeps the vectors aligned, in summary

| Requirement | Training | Inference |
|---|---|---|
| Same final tokens | `add_generation_prompt=True`, no assistant turn | Same template call in `dataset.py` |
| Last token at column -1 | `LeftPadDataCollator` | `padding_side = "left"` in `predict_steer` |
| Position means "k back from the end" | reads column `W-1-k` into slot `-1-k` | writes column `T-1-k` via `delta[:, -n:] = v[:, -n:]` |
| Pad count doesn't matter | RoPE is relative; pads get zero attention | same |
| Same number of positions | set by `num_positions` | read from checkpoint `shape[1]` |
| Right concept per row | merged dim 0 = `concept_id` | `proj_weight[idx]` |

### Where alignment breaks down

- **Short prompts at inference.** Column `T-1-k` is per batch, not per row. For a row with fewer than `k+1` real tokens, `v_k` lands on its left padding (harmless, since nothing attends there) or on its `<bos>`/template prefix (not harmless; see design decision 3). Training had the same mix at that `k`, which is the unmasked-padding decision. So the behavior is consistent, but it isn't concept content.
- **The instruction region only matches on average.** For `k ≥ 5`, `v_k` was fit on whatever instruction token happened to be k back in each training example. At test time it lands on a different instruction with different tokens there. The template positions `k = 0`–`4` are the only places where train and test land on literally the same token type.
- **Unused deep positions.** If a concept's longest training example is shorter than 64 tokens, the training loop never reaches the deeper `k`. Those rows stay zero (the `empty` warning fires, although its message says there are no examples at all), so they add nothing at inference.
- **Precision.** Vectors are built and normalized in fp32, then cast to the activation dtype (bf16) inside `forward` (`steering_vec.to(base.dtype)`).

## Measured prompt lengths and position coverage

Measured on `axbench/results/prod_9b_l20_concept500_diffmean_pos_steer_data_test_split`,
tokenized with `google/gemma-2-9b-it` exactly as `make_left_padded_data_module` does. Both
stages build the prompt identically (gemma-2 is not in `HAS_SYSTEM_PROMPT_MODELS`, so it is
a bare user turn with `add_generation_prompt=True` and no assistant turn), so the training
and inference rows are directly comparable.

### Prompt token lengths

Total tokens, chat-templated, including BOS:

| set | n | min | mean | max | p25 | p50 | p75 | p90 | p99 |
|---|---|---|---|---|---|---|---|---|---|
| training (pos+neg) | 47,808 | 13 | 31.9 | 217 | 26 | 31 | 36 | 42 | 56 |
| training positives | 23,904 | 13 | 31.1 | 141 | 26 | 30 | 35 | 41 | 52 |
| training negatives | 23,904 | 13 | 32.7 | 217 | 27 | 31 | 37 | 43 | 59 |
| inference steering prompts | 3,320 | 14 | 35.6 | 215 | 28 | 33 | 40 | 48 | 82 |

Real content tokens (total minus the 4-token `<bos><start_of_turn>user\n` prefix that
`_real_token_mask` drops): training median 27, inference median 29.

Two consequences:

- **Truncation never fires.** `make_left_padded_data_module` truncates at 1024 and the
  longest example anywhere is 217. Padding width per concept is ~100–220, not 1024. The
  silent-truncation trap in CLAUDE.md §8 does not bite this dataset.
- **Inference prompts run ~12% longer** than training ones (35.6 vs 31.9 mean; p99 82 vs
  56). The cause is the instruction pool, not the contrast concepts: test-pool base
  instructions average 11.5 tokens against 9.6 for the train pool, while contrast-concept
  length is 9.1 either way. Mild extra distribution shift on top of the content shift.

### Position coverage: `num_positions: 64` is about twice as deep as the data supports

`r_k`, the share of training rows with a real token `k` positions back from the end:

| slot | −1 | −9 | −17 | −25 | −32 | −40 | −48 | −64 |
|---|---|---|---|---|---|---|---|---|
| `r_k` | 1.000 | 1.000 | 0.963 | 0.625 | 0.280 | 0.080 | 0.020 | 0.002 |

37 of 64 slots sit below `r_k` = 0.5; 26 below 0.10; 12 below 0.01. Per concept, with 144
rows each:

| slot | concepts with **zero** rows reaching it | median rows reaching it |
|---|---|---|
| −32 | 1 / 332 | 35 of 144 |
| −40 | 19 / 332 | 8 of 144 |
| −48 | 120 / 332 | 1 of 144 |
| −64 | **287 / 332** | **0 of 144** |

For 287 of 332 concepts, slot −64's difference of means is computed over rows where every
contribution is left-padding or prefix. That is not a noisy estimate of a real direction,
it is a padding artifact. This quantifies the "unused deep positions" bullet above: those
rows are not left at zero, they are populated from padding.

Independently recomputing `r_k` from the tokenized training data matches the `real_frac`
tensor stored in the `DiffMeanPositionalWeighted` checkpoint to within 0.0000 across all 64
slots, which confirms both the tokenization above and the stored weights.

`DiffMeanPositionalWeighted` scales those slots by `r_k` ≈ 0.002, so they contribute
essentially nothing to steering or to the collapsed direction. Plain `DiffMeanPositional`
gives the same vectors full unit norm. If the unweighted variant is to be meaningful,
`num_positions` of 24–32 is where the data actually lives.

## Worth testing, roughly in order of payoff

- **Sweep `num_positions` (for example 1, 4, 16, 64).** If 4 or 16 matches or beats 64, the deep, padding-heavy vectors are adding noise.
- **Push the factor range above 5.0.** This tells you whether the weakness is prefill-only steering or bad directions.
- **Run the pending MeanToken vs. `DiffMean` cosine check (> 0.999)** on this dataset.
- **Scale positions by reliability.** Scale each `v_k` by the share of examples that have real content at k, or skip positions that fall on BOS or the prefix at steering time. This leaves the unmasked-training decision alone and only changes how the vectors are applied.
