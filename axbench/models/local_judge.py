"""Local vLLM-backed LM judge (Llama-3.1-70B-Instruct), drop-in for LanguageModel.

Deliberately NOT star-imported by axbench/__init__.py -- vllm is a heavy,
GPU-only dependency and axbench/__init__.py is pulled in by every script
(train.py, inference.py, generate.py), so importing vllm here would make it
a hard dependency for environments that never touch the local judge.
evaluate_local.py imports this module directly instead.

Follows the same pattern as /home/acbaez/gcm-interp/judge-evals/config.py +
evaluator.py (make_llm / generate_in_batches).
"""

import os
import logging
import re

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")
os.environ.setdefault("GLOO_LOG_LEVEL", "ERROR")
logging.getLogger("vllm").setLevel(logging.WARNING)

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

SEED = 42
JUDGE_MODEL_NAME = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
TOKENIZER_MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"

# A judge answer that never reaches "Rating:" parses as 0.0 (lm_judge.py's
# DEFAULT_RATING) -- indistinguishable from a genuine "concept absent" verdict.
# So any completion that ran out of budget mid-explanation is continued with
# FORCE_RATING_SUFFIX appended to its own text, which leaves the model no choice
# but to emit the score next. Only the number is kept from that continuation.
FORCE_RATING_SUFFIX = "\n\nRating: [["
FORCE_RATING_MAX_TOKENS = 6


class LocalLanguageModel(object):
    """Same call surface as axbench.models.language_models.LanguageModel,
    backed by a local vLLM engine instead of the OpenAI API."""

    def __init__(
        self,
        model_name=None,
        tokenizer_name=None,
        temperature=0.0,
        max_tokens=256,
        batch_size=256,
        **kwargs,
    ):
        self.model = model_name or JUDGE_MODEL_NAME
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.total_calls = 0
        # Runs sequentially process one ~210-prompt batch per (concept, model)
        # task, so this counts batches across every chat_completions() call on
        # this instance, not just within one -- logging every batch would mean
        # one print per task (thousands per run).
        self._batch_count = 0
        self._log_every_n_batches = 10

        num_gpus = self._count_gpus()
        if num_gpus == 0:
            raise RuntimeError("No GPUs detected -- LocalLanguageModel requires a GPU.")
        logger.warning(f"Detected {num_gpus} GPU(s). Loading local judge model: {self.model}")

        self.llm = LLM(
            model=self.model,
            quantization="bitsandbytes",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            dtype="auto",
            max_num_seqs=210,
            max_model_len=4096,
            seed=SEED,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or TOKENIZER_MODEL_NAME)

    @staticmethod
    def _count_gpus():
        return torch.cuda.device_count()

    def _render_prompt(self, prompt):
        chat = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)

    def _generate_in_batches(self, rendered_prompts, sampling_params, batch_size):
        peak_used_gb = 0.0
        n_batches = (len(rendered_prompts) + batch_size - 1) // batch_size
        for batch_idx, i in enumerate(range(0, len(rendered_prompts), batch_size)):
            batch = rendered_prompts[i:i + batch_size]
            self._batch_count += 1
            should_log = self._batch_count % self._log_every_n_batches == 0
            if should_log:
                max_prompt_len = max(len(ids) for ids in self.tokenizer(batch).input_ids)
                print(f"[local_judge] batch {batch_idx + 1}/{n_batches}: max prompt length after tokenization = {max_prompt_len}", flush=True)
            results = self.llm.generate(batch, sampling_params)

            # VRAM: query the driver directly (cudaMemGetInfo) rather than
            # torch's caching-allocator stats, since vLLM manages its own KV
            # cache pool outside torch's allocator.
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_gb = (total_bytes - free_bytes) / 1e9
            total_gb = total_bytes / 1e9
            peak_used_gb = max(peak_used_gb, used_gb)
            if should_log:
                try:
                    util_pct = torch.cuda.utilization()
                except Exception:
                    util_pct = "n/a"
                print(
                    f"[local_judge] batch {batch_idx + 1}/{n_batches} (size={len(batch)}): "
                    f"VRAM {used_gb:.1f}/{total_gb:.1f} GB (peak {peak_used_gb:.1f} GB), "
                    f"GPU util {util_pct}%",
                    flush=True,
                )

            yield [(r.outputs[0].text, r.outputs[0].finish_reason) for r in results]

    async def chat_completions(self, api_names, prompts, batch_size=None):
        """Same signature as LanguageModel.chat_completions: returns a list of
        completion strings, one per prompt, in input order. Declared async
        purely so the existing `await self.lm_model.chat_completions(...)`
        call site in lm_judge.py keeps working unmodified -- internally this
        is fully synchronous, matching vLLM's offline/batch API.

        The caller (lm_judge.py) hardcodes batch_size=16, sized for OpenAI's
        concurrent-HTTP-request model -- that's ignored here in favor of
        self.batch_size, sized for vLLM's GPU-side continuous batching,
        rather than editing lm_judge.py (shared with the OpenAI path)."""
        batch_size = self.batch_size
        rendered_prompts = [self._render_prompt(p) for p in prompts]
        # Reserve the last FORCE_RATING_MAX_TOKENS of the budget: the reasoning pass
        # stops that much short of max_tokens, so an answer that is about to run out
        # of room still has room left for a forced rating (see _force_missing_ratings).
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=1.0,
            top_k=-1,
            max_tokens=max(self.max_tokens - FORCE_RATING_MAX_TOKENS, 1),
            seed=SEED,
        )
        all_completions, finish_reasons = [], []
        for batch in self._generate_in_batches(rendered_prompts, sampling_params, batch_size):
            all_completions.extend(text.strip() for text, _ in batch)
            finish_reasons.extend(reason for _, reason in batch)
        all_completions = self._force_missing_ratings(
            rendered_prompts, all_completions, finish_reasons, batch_size)
        self.total_calls += len(prompts)
        return all_completions

    def _force_missing_ratings(self, rendered_prompts, completions, finish_reasons, batch_size):
        """Append the rating to any answer that was cut off mid-reasoning.

        An answer is continued when it stopped because it ran out of budget
        (finish_reason == "length") and has not written "Rating:" yet. The judge's own
        partial reasoning is fed back with FORCE_RATING_SUFFIX appended, so the very next
        tokens it writes are the score -- forced into its trace at the point it would
        otherwise have been truncated, rather than after the fact. Reasoning and forced
        rating together stay within max_tokens. Only the number is kept from the
        continuation, and the reasoning the judge did write is left untouched.
        """
        missing = [i for i, (c, r) in enumerate(zip(completions, finish_reasons))
                   if "Rating:" not in c and r == "length"]
        stopped_without_rating = sum(
            1 for c, r in zip(completions, finish_reasons) if "Rating:" not in c and r != "length")
        if stopped_without_rating:
            print(f"[local_judge] {stopped_without_rating}/{len(completions)} completions ended on "
                  f"their own without a rating -- those score 0", flush=True)
        if not missing:
            return completions
        print(f"[local_judge] {len(missing)}/{len(completions)} completions ran out of reasoning "
              f"budget ({self.max_tokens - FORCE_RATING_MAX_TOKENS} tokens); forcing the rating "
              f"into the trace", flush=True)
        forced_prompts = [rendered_prompts[i] + completions[i] + FORCE_RATING_SUFFIX for i in missing]
        sampling_params = SamplingParams(
            temperature=self.temperature, top_p=1.0, top_k=-1,
            max_tokens=FORCE_RATING_MAX_TOKENS, seed=SEED)
        continuations = []
        for batch in self._generate_in_batches(forced_prompts, sampling_params, batch_size):
            continuations.extend(text for text, _ in batch)
        unparsed = 0
        for i, cont in zip(missing, continuations):
            m = re.search(r"\d+(?:\.\d+)?", cont)
            if m:
                completions[i] = f"{completions[i]}{FORCE_RATING_SUFFIX}{m.group()}]]"
            else:
                unparsed += 1
        if unparsed:
            print(f"[local_judge] {unparsed} forced continuation(s) still had no number "
                  f"-- those score 0", flush=True)
        return completions

    def get_report(self):
        return {
            "total_calls": self.total_calls,
            "total_cache_hits": 0,
            "total_price": 0.0,
        }
