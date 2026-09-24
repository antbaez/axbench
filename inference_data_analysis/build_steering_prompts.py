#!/usr/bin/env python
"""
Builds a single-step concept-injection edit -- a random held-out contrast concept is
injected directly into the base instruction, in one LLM call, reusing the exact
wording of T_INSTRUCTION_WITH_CONCEPT (the same prompt inference.py's own
ContrastInstructions uses for its first, target-concept-injection step) -- applied
uniformly to a configurable raw instruction pool, and without ever loading the target
model onto GPU. This only exercises instruction_with_concept, which needs a tokenizer
(for chat-template formatting) and the OpenAI-backed lm_model, the same GPU-free shape
as generate.py's training/latent modes.

This deliberately does NOT reproduce inference.py's own two-step ContrastInstructions
edit (inject the target concept, then minimally edit toward the contrast concept via
a second, differently-worded prompt) -- see steering_prompts_two_step.json for that
version. Here there is no target-concept injection step at all: the contrast concept
is injected straight into the base instruction, one prompt, one call.

--dataset selects which raw instruction pool the base instructions are drawn from,
so the same one-step edit can be compared across pools:
    train    {dump_dir}/generate/base_instructions.json      (Dolly open_qa, train split)
    test     {dump_dir}/generate/base_instructions_test.json (Dolly open_qa, held-out split)
    alpaca   {master_data_dir}/alpaca_eval.json's "instruction" field

--exclude_dump_dir, applied to the train pool only: drops any instruction already
present in {exclude_dump_dir}/generate/base_instructions.json, so this analysis never
samples an instruction generate.py actually trained on for that dump (create_train_df
uses every instruction in a dump's base_instructions.json for training, so "used in
generate.py's output" and "present in that file" are the same set).

For each concept x dataset, --num_instructions instructions are sampled uniformly at
random (seeded by --seed, concept_id and dataset for reproducibility) from that pool.

Requires {dump_dir}/generate/contrastive_concepts.json (written by generate.py
--mode training) to draw held-out contrast concepts from other concepts' pools.

Writes {this dir}/steering_prompts.json:
    {concept_id: {"concept": ..., "datasets": {dataset_name: [
        {"instruction_id", "base_instruction", "contrast_concept",
         "concept_injected", "formatted_prompt"}, ...
    ]}}}

Usage:
    uv run inference_data_analysis/build_steering_prompts.py \\
        --dump_dir <dump> --num_concepts 10 --num_instructions 10 \\
        --exclude_dump_dir axbench/results/prod_9b_l20_concept500_diffmean_pos_steer_data
"""
import argparse
import asyncio
import json
import os
import random
from pathlib import Path

import httpx
import pandas as pd
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from axbench.utils.constants import HAS_SYSTEM_PROMPT_MODELS
from axbench.utils.dataset import (
    LanguageModel,
    assign_contrast_concepts,
    load_base_instructions,
    load_contrastive_concepts,
    run_tasks,
)
from axbench.utils.prompt_utils import instruction_with_concept

# generate.py defines equivalents of these three, but importing it as a package
# (rather than running it as __main__) breaks its `from args.dataset_args import
# ...` line, which only resolves because axbench/scripts/ gets added to sys.path
# when generate.py itself is the entry point. Duplicated here to stay decoupled.
MODEL_NAME_MAP = {
    "gemma-2-2b": "google/gemma-2-2b-it",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}


def make_openai_client():
    return AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=1000),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )


def load_metadata_flatten(generate_dir):
    """Same shape as generate.py's own load_metadata_flatten: metadata.jsonl -> list of
    {concept, ref, concept_genres_map, concept_id}."""
    metadata = []
    with open(Path(generate_dir) / "metadata.jsonl") as f:
        for line in f:
            data = json.loads(line)
            concept = data["concept"]
            metadata.append({
                "concept": concept,
                "ref": data["ref"],
                "concept_genres_map": {concept: data["concept_genres_map"][concept]},
                "concept_id": data["concept_id"],
            })
    return metadata


DATASET_CHOICES = ("train", "test", "alpaca")


def load_pool(dataset, dump_dir, master_data_dir, exclude=None):
    if dataset == "train":
        pool = load_base_instructions(Path(dump_dir) / "generate", dist="train")
        if exclude:
            pool = [x for x in pool if x not in exclude]
        return pool
    if dataset == "test":
        return load_base_instructions(Path(dump_dir) / "generate", dist="test")
    if dataset == "alpaca":
        df = pd.read_json(Path(master_data_dir) / "alpaca_eval.json")
        return df["instruction"].tolist()
    raise ValueError(f"unknown dataset {dataset!r}")


async def build_rows(lm_model, tokenizer, concept, sampled, contrastive_store, rng,
                      steering_model_name, api_tag):
    """Inject a random held-out contrast concept directly into `sampled` base
    instructions, in one call -- reusing instruction_with_concept (the same prompt/
    wording ContrastInstructions' own target-concept-injection step uses), just
    pointed at the contrast concept instead of the target concept, applied straight
    to the base instruction instead of to an already target-concept-injected one."""
    own = set(contrastive_store[concept])
    held_out = sorted({
        c for other, items in contrastive_store.items() if other != concept
        for c in items if c not in own
    })
    if not held_out:
        raise ValueError(
            f"no held-out contrast concepts for '{concept}' -- contrastive_concepts.json "
            f"needs entries for more than one concept.")
    contrast_concepts = assign_contrast_concepts(held_out, len(sampled), rng=rng)

    injected = (await run_tasks([instruction_with_concept(
        lm_model, tokenizer, concepts=contrast_concepts, content=sampled,
        api_tag=api_tag)]))[0]

    system_messages = []
    if steering_model_name in HAS_SYSTEM_PROMPT_MODELS:
        system_messages = [{"role": "system", "content": "You are a helpful assistant."}]

    rows = []
    for i, (base, contrast_concept, concept_injected) in enumerate(
            zip(sampled, contrast_concepts, injected)):
        formatted_prompt = tokenizer.decode(
            tokenizer.apply_chat_template(
                system_messages + [{"role": "user", "content": concept_injected}],
                tokenize=True, add_generation_prompt=True)[1:])  # drop bos
        rows.append({
            "instruction_id": i,
            "base_instruction": base,
            "contrast_concept": contrast_concept,
            "concept_injected": concept_injected,
            "formatted_prompt": formatted_prompt,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--master_data_dir", default="axbench/data")
    parser.add_argument("--num_concepts", type=int, default=10)
    parser.add_argument("--num_instructions", type=int, default=10)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES,
                         default=list(DATASET_CHOICES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lm_model", default="gpt-4o-mini")
    parser.add_argument("--steering_model_name", default=None,
                         help="Defaults to whichever model the dump's concepts were built for.")
    parser.add_argument("--exclude_dump_dir", default=None,
                         help="Train pool only: drop any instruction already present in "
                              "this dump's generate/base_instructions.json, so the analysis "
                              "never samples something generate.py actually trained on.")
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    generate_dir = dump_dir / "generate"

    metadata = sorted(load_metadata_flatten(generate_dir), key=lambda e: e["concept_id"])
    metadata = metadata[:args.num_concepts]
    if not metadata:
        raise FileNotFoundError(f"No concepts found in {generate_dir}/metadata.jsonl")

    model_name = args.steering_model_name or MODEL_NAME_MAP[metadata[0]["ref"].split("/")[3]]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    contrastive_store = load_contrastive_concepts(generate_dir)

    client = make_openai_client()
    lm_model = LanguageModel(
        args.lm_model, client, dump_dir=str(generate_dir),
        use_cache=True, master_data_dir=args.master_data_dir,
    )

    exclude = None
    if args.exclude_dump_dir:
        exclude = set(load_base_instructions(
            Path(args.exclude_dump_dir) / "generate", dist="train"))
        print(f"Excluding {len(exclude)} instructions already used in "
              f"{args.exclude_dump_dir}/generate/base_instructions.json from the train pool.")

    pools = {
        name: load_pool(name, dump_dir, args.master_data_dir,
                         exclude=exclude if name == "train" else None)
        for name in args.datasets
    }

    out = {}
    for entry in metadata:
        concept_id, concept = entry["concept_id"], entry["concept"]
        print(f"concept {concept_id}: {concept}")
        datasets_out = {}
        for dataset in args.datasets:
            pool = pools[dataset]
            rng = random.Random(f"{args.seed}:{concept_id}:{dataset}")
            sampled = rng.sample(pool, min(args.num_instructions, len(pool)))
            rows = asyncio.run(build_rows(
                lm_model, tokenizer, concept, sampled, contrastive_store, rng,
                model_name, api_tag=f"inference_data_analysis.{dataset}"))
            datasets_out[dataset] = rows
        out[str(concept_id)] = {"concept": concept, "datasets": datasets_out}

    lm_model.save_cache()

    out_path = Path(__file__).parent / "steering_prompts.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
