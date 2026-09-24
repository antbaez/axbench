#!/usr/bin/env python
"""
Join sample_generations_debug_chunk0_rank0.json (the first-10-concepts x
first-10-input_ids x every-factor generation preview inference.py writes) against
the per-generation ratings produced by evaluate_local.py and evaluate_subset.py,
so ratings can be read side by side with the exact prompt/generation they scored.

Writes two files, one per evaluator (skipped if that evaluator has no output),
under {dump_dir}/evaluate/:
    sample_generations_preview_with_evaluate_local_ratings.json
    sample_generations_preview_with_evaluate_subset_ratings.json

Only valid within a single dump: the debug/preview file and the evaluate outputs
must come from the same inference.py run, since ContrastInstructions samples a
fresh prompt per run (see the v4-vs-v5 mismatch this script was built to avoid).

evaluate_local.py and evaluate_subset.py call build_previews() themselves at the
end of a steering-eval run, so this file's CLI is mostly for rebuilding the
preview by hand (e.g. after reorganizing an old dump's evaluate/ directory).

Usage:
    uv run axbench/scripts/build_preview_ratings.py --dump_dir <dump>
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def _resolve(evaluate_dir, subdir_name):
    """evaluate_dir/subdir_name if it exists (a dump reorganized like v5), else evaluate_dir itself
    (the flat layout evaluate_local.py/evaluate_subset.py actually write)."""
    sub = Path(evaluate_dir) / subdir_name
    return sub if sub.is_dir() else Path(evaluate_dir)


def find_preview_path(inference_dir):
    """The first-10-concepts debug/preview file inference.py writes, or None if absent."""
    for name in ("sample_generations_preview_chunk0_rank0.json",
                 "sample_generations_debug_chunk0_rank0.json"):
        path = Path(inference_dir) / name
        if path.exists():
            return path
    return None


def load_local_ratings(evaluate_dir):
    """concept_id, input_id, factor, model -> rating dict, from evaluate_local.py's steering_data.parquet.

    steering_data.parquet and steering_test_data.parquet cover disjoint input_ids per
    concept (the eval/test split evaluate.py's data_generator draws), so both are
    loaded and merged -- otherwise half of any concept's input_ids would silently
    show no rating.
    """
    base = _resolve(evaluate_dir, "evaluate_local")
    ratings = {}
    for filename in ("steering_data.parquet", "steering_test_data.parquet"):
        path = base / filename
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        model_names = sorted({
            col.rsplit("_LMJudgeEvaluator", 1)[0]
            for col in df.columns if col.endswith("_LMJudgeEvaluator")
        })
        for _, row in df.iterrows():
            key = (int(row["concept_id"]), int(row["input_id"]), float(row["factor"]))
            for model in model_names:
                prefix = f"{model}_LMJudgeEvaluator"
                if prefix not in df.columns:
                    continue
                ratings.setdefault(key, {})[model] = {
                    "split": filename.removesuffix("_data.parquet"),
                    "lm_judge_rating": row.get(prefix),
                    "relevance_concept_rating": row.get(f"{prefix}_relevance_concept_ratings"),
                    "relevance_concept_why": row.get(f"{prefix}_relevance_concept_completions"),
                    "relevance_instruction_rating": row.get(f"{prefix}_relevance_instruction_ratings"),
                    "relevance_instruction_why": row.get(f"{prefix}_relevance_instruction_completions"),
                    "fluency_rating": row.get(f"{prefix}_fluency_ratings"),
                    "fluency_why": row.get(f"{prefix}_fluency_completions"),
                }
    return ratings


def load_subset_ratings(evaluate_dir):
    """concept_id, input_id, factor, model -> rating dict, from evaluate_subset.py's steering_ratings.jsonl.

    Same eval/test split merge as load_local_ratings, from steering_ratings.jsonl
    and steering_test_ratings.jsonl.
    """
    base = _resolve(evaluate_dir, "evaluate_subset")
    ratings = {}
    for filename in ("steering_ratings.jsonl", "steering_test_ratings.jsonl"):
        path = base / filename
        if not path.exists():
            continue
        split = filename.removesuffix("_ratings.jsonl")
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                key = (int(row["concept_id"]), int(row["input_id"]), float(row["factor"]))
                ratings.setdefault(key, {})[row["model"]] = {
                    "split": split,
                    "lm_judge_rating": row.get("lm_judge_rating"),
                    "relevance_concept_rating": row.get("relevance_concept"),
                    "relevance_instruction_rating": row.get("relevance_instruction"),
                    "fluency_rating": row.get("fluency"),
                    # evaluate_subset.py only stores the numeric ratings, not the
                    # judge's rationale text -- unlike evaluate_local.py's *_completions
                    # columns, so there is no "_why" field to attach here.
                }
    return ratings


def build_one(evaluate_dir, preview_path, ratings, source_name):
    """Join `preview_path` against a single evaluator's ratings dict and write its own file."""
    with open(preview_path) as f:
        preview = json.load(f)

    out = {}
    for concept_id, entry in preview.items():
        out_generations = []
        for gen in entry["generations"]:
            key = (int(concept_id), gen["input_id"], gen["factor"])
            row = {
                "input_id": gen["input_id"],
                "factor": gen["factor"],
                "prompt": gen["prompt"],
                "models": {},
            }
            for model, generation_text in gen["generations"].items():
                model_entry = {"generation": generation_text}
                rating = ratings.get(key, {}).get(model)
                if rating is not None:
                    model_entry.update(rating)
                row["models"][model] = model_entry
            out_generations.append(row)
        out[concept_id] = {"concept": entry["concept"], "generations": out_generations}

    out_path = Path(evaluate_dir) / f"sample_generations_preview_with_{source_name}_ratings.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out_path


def build_previews(evaluate_dir, inference_dir):
    """Build both preview-with-ratings files, skipping any evaluator with no output yet.

    Safe to call repeatedly (e.g. once per evaluate_local.py chunk merge): each call
    just overwrites the file with whatever ratings currently exist. Returns the list
    of paths written.
    """
    preview_path = find_preview_path(inference_dir)
    if preview_path is None:
        return []

    written = []
    local_ratings = load_local_ratings(evaluate_dir)
    if local_ratings:
        written.append(build_one(evaluate_dir, preview_path, local_ratings, "evaluate_local"))
    subset_ratings = load_subset_ratings(evaluate_dir)
    if subset_ratings:
        written.append(build_one(evaluate_dir, preview_path, subset_ratings, "evaluate_subset"))
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", required=True)
    args = parser.parse_args()

    evaluate_dir = Path(args.dump_dir) / "evaluate"
    inference_dir = Path(args.dump_dir) / "inference"
    written = build_previews(evaluate_dir, inference_dir)
    if not written:
        print(f"Nothing written -- no preview file under {inference_dir} or no evaluate "
              f"output under {evaluate_dir}.")
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
