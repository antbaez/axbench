"""Record layer-`layer` residual-stream norms, per token, during steered generation.

The numbers come straight out of the steering hook itself: pyvene calls the
intervention module with the *pre*-intervention residual stream as its first
positional argument and uses its return value as the *post*-intervention residual
stream, so a single forward hook on that module sees both sides of the edit.

A second, independent hook on layer `layer + 1`'s input records the same tensor from
the other direction (nothing sits between one decoder layer's output and the next
layer's input in Gemma-2, see modeling_gemma2.py: `hidden_states = layer_outputs[0]`).
The two are compared for the first `verify_batches` generate() calls purely as a
check that the post-intervention value really is what the rest of the model consumes.

That second hook has to go on torch's *global* module hook registry rather than on the
layer itself: pyvene calls _cleanup_states() at the top of every generate(), which runs
remove_forward_hooks() over the whole model and pops every module's _forward_hooks and
_forward_pre_hooks. A hook registered on the layer is silently gone before the first
token. The intervention module is not a submodule of the wrapped model, so the hook on
it survives. A global hook fires for every module in the process, so it is only
installed while verification is still pending and removed as soon as it is done.

Shapes: the intervention runs once per forward pass, so each generate() call produces
one [batch, prompt_len] entry for prefill followed by one [batch, 1] entry per decode
step. Concatenating along dim 1 rebuilds the full sequence, prompt first. The final
sampled token has no forward pass of its own, so the captured width is
prompt_len + (tokens_generated - 1).
"""
import numpy as np
import torch
from torch.nn.modules.module import register_module_forward_pre_hook


class SteeringNormRecorder:
    """Collects per-token residual norms at the steering site, one record per row.

    The caller owns the hooks' lifetime (`with recorder:`), so they are removed even if
    generation raises; Model.predict_steer only calls reset() before each generate()
    and record_batch() after it. Records accumulate in `self.records`.
    """

    def __init__(self, model, intervention, layer, model_name="", verify=True,
                 verify_batches=1, logger=None):
        self.model = model
        self.intervention = intervention
        self.layer = layer
        self.model_name = model_name
        self.logger = logger
        self.records = []
        self._pre, self._post, self._next = [], [], []
        self._ax_handle = None
        self._next_handle = None
        # layer+1 may not exist if we ever steer the final layer; the check is optional
        layers = model.model.layers
        self._next_layer = layers[layer + 1] if layer + 1 < len(layers) else None
        self._verify_left = verify_batches if verify else 0
        self._steering_checked = not verify

    # -- hooks ------------------------------------------------------------------
    def _ax_hook(self, module, args, output):
        base = args[0]
        out = getattr(output, "output", output)   # pyvene may wrap in InterventionOutput
        self._pre.append(base.detach().float().norm(dim=-1).cpu())
        self._post.append(out.detach().float().norm(dim=-1).cpu())

    def _next_layer_hook(self, module, args):
        # global hook: fires for every module in the process, keep only the one we want
        if module is self._next_layer and args:
            self._next.append(args[0].detach().float().norm(dim=-1).cpu())
        return None

    # -- lifecycle --------------------------------------------------------------
    def __enter__(self):
        self._ax_handle = self.intervention.register_forward_hook(self._ax_hook)
        if self._verify_left > 0 and self._next_layer is not None:
            self._next_handle = register_module_forward_pre_hook(self._next_layer_hook)
        return self

    def __exit__(self, *exc):
        for h in (self._ax_handle, self._next_handle):
            if h is not None:
                h.remove()
        self._ax_handle = self._next_handle = None
        return False

    def reset(self):
        self._pre, self._post, self._next = [], [], []

    # -- per-batch --------------------------------------------------------------
    def record_batch(self, batch_examples, attention_mask, generations, mags, max_acts,
                     tokenizer):
        """Turn the norms captured during one generate() call into per-row records.

        Each record holds only real tokens: left padding is dropped from the prompt
        and generation is cut at the first end-of-turn/EOS token.
        """
        if not self._post:
            raise RuntimeError(
                "No activations captured -- the intervention hook never fired. Check "
                "that the intervention passed in is the module pyvene is calling.")
        pre = torch.cat(self._pre, dim=1)
        post = torch.cat(self._post, dim=1)
        prompt_len = attention_mask.shape[1]

        if self._verify_left > 0:
            self._check_next_layer(post)
            self._verify_left -= 1
            if self._verify_left == 0 and self._next_handle is not None:
                self._next_handle.remove()
                self._next_handle = None
        if not self._steering_checked:
            self._check_steering(pre, post, mags, prompt_len)
            self._steering_checked = True

        stop_ids = _stop_token_ids(tokenizer)
        pad_id = tokenizer.pad_token_id
        n_pad = (prompt_len - attention_mask.sum(dim=1)).tolist()
        max_gen = post.shape[1] - prompt_len
        mags = mags.detach().float().cpu().tolist()
        max_acts = max_acts.detach().float().cpu().tolist()
        gen_cpu = generations[:, prompt_len:].cpu()
        for i, row in enumerate(batch_examples.itertuples(index=False)):
            n_gen = min(_generated_length(gen_cpu[i].tolist(), stop_ids, pad_id), max_gen)
            lo = int(n_pad[i])
            hi = prompt_len + n_gen
            self.records.append({
                "model": self.model_name,
                "concept_id": int(row.concept_id),
                "input_id": int(row.input_id),
                "factor": float(mags[i]),
                "max_act": float(max_acts[i]),
                "strength": float(mags[i] * max_acts[i]),
                "n_prompt": prompt_len - lo,
                "n_gen": int(n_gen),
                "norms_pre": pre[i, lo:hi].numpy().astype(np.float32),
                "norms_post": post[i, lo:hi].numpy().astype(np.float32),
            })

    # -- verification prints ----------------------------------------------------
    def _check_next_layer(self, post):
        """Confirm the post-intervention value is exactly what layer+1 receives."""
        tag = f"[norm-check] {self.model_name}"
        if self._next_layer is None:
            self._log(f"{tag}: layer {self.layer} is the last layer; skipping.")
            return
        if len(self._next) != len(self._post):
            self._log(
                f"{tag}: MISMATCH in call counts -- intervention fired {len(self._post)} "
                f"times, layer {self.layer + 1} {len(self._next)} times.")
            return
        nxt = torch.cat(self._next, dim=1)
        if nxt.shape != post.shape:
            self._log(f"{tag}: MISMATCH in shape {tuple(post.shape)} vs {tuple(nxt.shape)}")
            return
        max_diff = (post - nxt).abs().max().item()
        status = "MATCH" if max_diff == 0.0 else ("close" if max_diff < 1e-3 else "MISMATCH")
        self._log(
            f"{tag}: layer {self.layer} post-intervention vs layer {self.layer + 1} input: "
            f"{status} (max |diff| = {max_diff:.3e} over {post.numel()} norms, "
            f"{len(self._post)} forward passes)")

    def _check_steering(self, pre, post, mags, prompt_len):
        """Confirm the captured tensors really are the steered activations.

        A steered row's prompt norms must differ between pre and post, and a factor-0
        row must be untouched everywhere. A steered row's *generated* norms should be
        unchanged for the prompt-only interventions (MeanTokenDiffMean,
        LastTokenDiffMean, DiffMeanPositional*) and changed for stock addition/clamping.
        """
        tag = f"[norm-capture] {self.model_name}"
        mags = mags.detach().float().cpu()
        d = (post - pre).abs()
        self._log(f"{tag}: batch of {post.shape[0]}, captured {post.shape[1]} positions "
                  f"({prompt_len} prompt + {post.shape[1] - prompt_len} decode)")
        hot = (mags > 0).nonzero().flatten()
        if len(hot):
            i = int(hot[0])
            p_max = d[i, :prompt_len].max().item()
            g_max = d[i, prompt_len:].max().item() if post.shape[1] > prompt_len else 0.0
            self._log(
                f"{tag}:   factor={mags[i].item():g} row: prompt max|post-pre|={p_max:.4g} "
                f"(must be > 0: steering applied), decode max|post-pre|={g_max:.4g} "
                f"(0 for prompt-only interventions, > 0 for stock addition/clamping)")
            if p_max == 0.0:
                self._log(f"{tag}:   WARNING: a steered row shows no change -- the "
                          f"captured activations are NOT steered.")
        else:
            self._log(f"{tag}:   no nonzero factor in this batch to check against.")
        cold = (mags == 0).nonzero().flatten()
        if len(cold):
            i = int(cold[0])
            self._log(f"{tag}:   factor=0 row: max|post-pre|={d[i].max().item():.4g} "
                      f"(must be 0: unsteered baseline)")

    def _log(self, msg):
        if self.logger is not None:
            self.logger.warning(msg)
        else:
            print(msg, flush=True)


def _stop_token_ids(tokenizer):
    ids = {tokenizer.eos_token_id}
    for tok in ("<end_of_turn>", "<eos>", "<|eot_id|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            ids.add(tid)
    ids.discard(None)
    return ids


def _generated_length(gen_ids, stop_ids, pad_id):
    """Real generated tokens: up to and including the first stop token."""
    for i, t in enumerate(gen_ids):
        if t in stop_ids:
            return i + 1
        if pad_id is not None and t == pad_id:
            return i
    return len(gen_ids)
