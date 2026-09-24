#!/usr/bin/env python
"""
Build the same "prompt + generation + ratings" preview that
axbench/scripts/build_preview_ratings.py produces, but not capped at
inference.py's hardcoded first-10-concepts x first-10-input_ids debug dump: this
pulls concepts/instructions directly from the evaluate-stage parquet/jsonl files
themselves, so how many of each to include is a command-line argument.

Two independent outputs, one per evaluator (skipped if that evaluator's files are
absent), written under {dump_dir}/evaluate/:
    sample_generations_compiled_with_evaluate_local_ratings.json
    sample_generations_compiled_with_evaluate_subset_ratings.json

Concepts are the --num_concepts lowest concept_ids present in that evaluator's own
output; instructions are, per selected concept, the --num_instructions lowest
input_ids present for it (steering_data.parquet's eval split covers ~half of a
concept's input_ids and steering_test_data.parquet the other half, so both are
combined first -- see build_preview_ratings.load_local_ratings for the same
reasoning applied there).

Usage:
    python compile_evaluation.py --dump_dir <dump> --num_concepts 20 --num_instructions 5
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from axbench.scripts.build_preview_ratings import _resolve


def compile_local(evaluate_dir, num_concepts, num_instructions):
    base = _resolve(evaluate_dir, "evaluate_local")
    frames = []
    for filename in ("steering_data.parquet", "steering_test_data.parquet"):
        path = base / filename
        if path.exists():
            df = pd.read_parquet(path)
            df["split"] = filename.removesuffix("_data.parquet")
            frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)

    model_names = sorted({
        col.rsplit("_LMJudgeEvaluator", 1)[0]
        for col in df.columns if col.endswith("_LMJudgeEvaluator")
    })

    concept_ids = sorted(df["concept_id"].unique())[:num_concepts]
    out = {}
    for concept_id in concept_ids:
        cdf = df[df["concept_id"] == concept_id]
        concept_name = cdf["input_concept"].iloc[0]
        input_ids = sorted(cdf["input_id"].unique())[:num_instructions]
        generations = []
        for input_id in input_ids:
            idf = cdf[cdf["input_id"] == input_id].sort_values("factor")
            for _, row in idf.iterrows():
                models = {}
                for model in model_names:
                    gen_col = f"{model}_steered_generation"
                    if gen_col not in df.columns:
                        continue
                    prefix = f"{model}_LMJudgeEvaluator"
                    models[model] = {
                        "generation": row.get(gen_col),
                        "split": row.get("split"),
                        "lm_judge_rating": row.get(prefix),
                        "relevance_concept_rating": row.get(f"{prefix}_relevance_concept_ratings"),
                        "relevance_concept_why": row.get(f"{prefix}_relevance_concept_completions"),
                        "relevance_instruction_rating": row.get(f"{prefix}_relevance_instruction_ratings"),
                        "relevance_instruction_why": row.get(f"{prefix}_relevance_instruction_completions"),
                        "fluency_rating": row.get(f"{prefix}_fluency_ratings"),
                        "fluency_why": row.get(f"{prefix}_fluency_completions"),
                    }
                generations.append({
                    "input_id": int(input_id),
                    "factor": float(row["factor"]),
                    "prompt": row.get("original_prompt"),
                    "models": models,
                })
        out[str(int(concept_id))] = {"concept": concept_name, "generations": generations}
    return out


def compile_subset(evaluate_dir, num_concepts, num_instructions):
    base = _resolve(evaluate_dir, "evaluate_subset")
    rows = []
    for filename in ("steering_ratings.jsonl", "steering_test_ratings.jsonl"):
        path = base / filename
        if not path.exists():
            continue
        split = filename.removesuffix("_ratings.jsonl")
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                row["split"] = split
                rows.append(row)
    if not rows:
        return None

    by_concept = defaultdict(list)
    for row in rows:
        by_concept[row["concept_id"]].append(row)

    concept_ids = sorted(by_concept)[:num_concepts]
    out = {}
    for concept_id in concept_ids:
        crows = by_concept[concept_id]
        concept_name = crows[0]["concept"]
        input_ids = set(sorted({r["input_id"] for r in crows})[:num_instructions])

        by_key = defaultdict(dict)  # (input_id, factor) -> {model: row}
        for r in crows:
            if r["input_id"] not in input_ids:
                continue
            by_key[(r["input_id"], r["factor"])][r["model"]] = r

        generations = []
        for (input_id, factor), model_rows in sorted(by_key.items()):
            models = {}
            prompt = None
            for model, r in model_rows.items():
                prompt = r["instruction"]
                models[model] = {
                    "generation": r["generation"],
                    "split": r["split"],
                    "lm_judge_rating": r.get("lm_judge_rating"),
                    "relevance_concept_rating": r.get("relevance_concept"),
                    "relevance_instruction_rating": r.get("relevance_instruction"),
                    "fluency_rating": r.get("fluency"),
                    # evaluate_subset.py stores no judge rationale text, so there is
                    # no "_why" field to carry here -- see build_preview_ratings.py.
                }
            generations.append({
                "input_id": int(input_id), "factor": float(factor),
                "prompt": prompt, "models": models,
            })
        out[str(concept_id)] = {"concept": concept_name, "generations": generations}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--num_concepts", type=int, required=True,
                         help="How many of the lowest concept_ids to include.")
    parser.add_argument("--num_instructions", type=int, required=True,
                         help="Per included concept, how many of its lowest input_ids to include.")
    args = parser.parse_args()

    evaluate_dir = Path(args.dump_dir) / "evaluate"

    local = compile_local(evaluate_dir, args.num_concepts, args.num_instructions)
    subset = compile_subset(evaluate_dir, args.num_concepts, args.num_instructions)

    if local is None and subset is None:
        raise FileNotFoundError(
            f"No evaluate_local or evaluate_subset output found under {evaluate_dir}")

    if local is not None:
        path = evaluate_dir / "sample_generations_compiled_with_evaluate_local_ratings.json"
        with open(path, "w") as f:
            json.dump(local, f, indent=2, ensure_ascii=False)
        print(f"Wrote {path}")

    if subset is not None:
        path = evaluate_dir / "sample_generations_compiled_with_evaluate_subset_ratings.json"
        with open(path, "w") as f:
            json.dump(subset, f, indent=2, ensure_ascii=False)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
