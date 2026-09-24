#!/usr/bin/env python
"""
A subsampled variant of evaluate.py --mode steering / steering_test, plus a
select_best mode that halves the request count for steering_test by only
scoring each concept's own already-selected factor instead of every factor.

Scoring is not reimplemented: this builds the same LanguageModel client as
evaluate.py (see evaluate.py:352-379) and calls the same LMJudgeEvaluator, so a
given (concept, model, factor) cell gets exactly the number evaluate.py would
give it. What changes is which rows are judged and how results are written.

    --mode {steering, steering_test, select_best}   default: select_best
    --factors "0.5,1.0,2.5,5.0"   candidate steering factors. 'all' = every
                                  factor present in the data. Default: all.
    --examples_per_concept N     keep the first N input_ids per concept/split.
    --max_concepts N             stop after N concepts.
    --num_workers N              threads issuing judge calls in parallel.
    --eval_models "A,B"          steering methods to judge. Default: all of the
                                 yaml's evaluate.models.

select_best runs --factors against the steering (eval) split, picks each
(concept, model)'s own best-scoring factor from that (mirroring
evaluate.py's get_best_factors(), which exists but is never wired into its
steering/steering_test flow), then judges steering_test using *only* that one
factor per (concept, model) instead of every candidate factor -- an N-factor
candidate list costs N times fewer steering_test requests than judging all of
them, since each concept only pays for the factor it actually won on.

Outputs, all under {dump_dir}/evaluate/:
    steering_subset.json / steering_test_subset.json   per-phase detail
    {phase}_ratings.jsonl                               one line per judged
                                                         generation, every raw
                                                         rating
    select_best_summary.json   (select_best only) the best factor chosen per
                                (concept, model) and both phases' summaries

Usage:
    uv run axbench/scripts/evaluate_subset.py --config <cfg> --dump_dir <dump> \
        --factors "0.5,1.5,3.0,5.0" --max_concepts 20

Whole-number-factors example, first 250 concepts, gpt-4o-mini judge:
    cd ~/axbench && set -a && source .env && set +a && \
    uv run axbench/scripts/evaluate_subset.py \
        --config axbench/sweep/antbaez/diffmean_variants_l20.yaml \
        --dump_dir axbench/results/prod_9b_l20_concept500_diffmean_pos_steer_data_v2 \
        --mode select_best --factors "1.0,2.0,3.0,4.0,5.0" \
        --max_concepts 250 --num_workers 50

Needs no GPU -- evaluate.py loads no local model when LMJudgeEvaluator is the
only steering evaluator, so this runs fine in a plain CPU allocation.
"""
import os, sys, json, math, asyncio, logging, threading, warnings
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd
from openai import AsyncOpenAI
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", message=r"pyreft not installed.*")
warnings.filterwarnings("ignore", message=r"HyperSteer unavailable.*")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import axbench
from axbench.models.language_models import LanguageModel
from axbench.scripts.args.eval_args import EvalArgs
from axbench.scripts.build_preview_ratings import build_previews

logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.WARN)
logger = logging.getLogger(__name__)


def make_lm_model(lm_model_name, dump_dir):
    """Same client construction as evaluate.py:358-379, minus the process pool."""
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=1000),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )
    lm_model = LanguageModel(
        lm_model_name, client, dump_dir=dump_dir, use_cache=True,
        cache_level="prompt", cache_tag="evaluate_subset",
        master_data_dir="axbench/data", temperature=0.0, rate_limit_retries=8)
    return client, lm_model


def opt_int(value):
    """Parse a knob that takes either a count or 'all' (= no limit -> None)."""
    if value is None or isinstance(value, int):
        return value
    text = str(value).strip().lower()
    return None if text in ("all", "none", "", "-1") else int(text)


def parse_factors(value, available):
    """'all' (default) -> every factor present in the data; otherwise a manual
    comma-separated list, checked against what's actually there so a typo
    fails loudly instead of silently evaluating nothing for it."""
    text = str(value).strip().lower()
    if text in ("all", "none", ""):
        return sorted(available)
    requested = sorted({float(x) for x in str(value).split(",") if x.strip()})
    missing = sorted(set(requested) - set(available))
    if missing:
        raise SystemExit(
            f"--factors requested {missing} not present in the data. "
            f"Available factors: {sorted(available)}")
    return requested


def select_rows(df, mode, winrate_split_ratio, factors, examples_per_concept, max_concepts):
    """
    Apply the mode split, then restrict to the requested factors and example count.

    The mode split mirrors evaluate.py's data_generator: with a winrate_split_ratio
    set, `steering` takes the low input_ids and `steering_test` the high ones, so
    the two modes never judge the same generation.
    """
    concept_ids = sorted(df["concept_id"].unique())
    if max_concepts:
        concept_ids = concept_ids[:max_concepts]
    df = df[df["concept_id"].isin(concept_ids)]

    if winrate_split_ratio is not None and float(winrate_split_ratio) > 0:
        n_input_ids = df["input_id"].max() + 1
        n_steering_ids = n_input_ids - round(n_input_ids * float(winrate_split_ratio))
        if mode == "steering":
            df = df[df["input_id"] < n_steering_ids]
        else:
            df = df[df["input_id"] >= n_steering_ids]

    df = df[df["factor"].isin(factors)]

    if examples_per_concept:
        keep = sorted(df["input_id"].unique())[:examples_per_concept]
        df = df[df["input_id"].isin(keep)]

    return df, sorted(df["concept_id"].unique())


def run_phase(df, concept_ids, models, factor_lookup, dump_dir, label,
              evaluator_class, get_lm_model, num_workers, args):
    """
    Judge every (concept, model) pair, each restricted to factor_lookup(concept, model)
    -- the full candidate list for a plain sweep, or a single already-chosen factor for
    select_best's steering_test phase. A pair whose lookup returns an empty list is
    skipped (e.g. select_best when phase 1 produced no result for that pair).

    Returns (per_concept detail list, acc grouped by factor, acc_flat ungrouped --
    the latter is what select_best's held-out score averages over, since in that
    phase different concepts contribute rows under different factor values).
    """
    tasks = [(c, m, factor_lookup(c, m)) for c in concept_ids for m in models]
    tasks = [(c, m, f) for c, m, f in tasks if f]
    n_concepts = len({c for c, _, _ in tasks})

    ratings_path = dump_dir / f"{label}_ratings.jsonl"
    per_concept = []
    acc = {m: defaultdict(lambda: defaultdict(list)) for m in models}
    acc_flat = {m: [] for m in models}

    def judge(task):
        concept_id, model_name, task_factors = task
        current_df = df[
            (df["concept_id"] == concept_id) & (df["factor"].isin(task_factors))
        ].reset_index(drop=True)
        evaluator = evaluator_class(
            model_name, dump_dir=str(dump_dir), concept_id=int(concept_id),
            lm_model=get_lm_model(), winrate_baseline=args.winrate_baseline,
            steer_dataset_type=getattr(args, "steer_data_type", None))
        return concept_id, model_name, current_df, evaluator.compute_metrics(current_df)

    with open(ratings_path, "w") as raw_f, \
            tqdm(total=n_concepts, desc=f"{label} concepts", unit="concept") as pbar, \
            ThreadPoolExecutor(max_workers=num_workers) as pool:
        entry = None
        # pool.map yields in submission order (concept-major, since tasks was built
        # that way), so every write below happens deterministically -- no output lock.
        for concept_id, model_name, current_df, r in pool.map(judge, tasks):
            if entry is None or entry["concept_id"] != int(concept_id):
                entry = {"concept_id": int(concept_id),
                         "concept": current_df["input_concept"].iloc[0], "models": {}}
                per_concept.append(entry)

            entry["models"][model_name] = {
                k: [float(v) for v in r[k]]
                for k in ("factor", "lm_judge_rating", "relevance_concept_ratings",
                          "relevance_instruction_ratings", "fluency_ratings")}
            for f, lj, rc, ri, fl in zip(
                    r["factor"], r["lm_judge_rating"], r["relevance_concept_ratings"],
                    r["relevance_instruction_ratings"], r["fluency_ratings"]):
                acc[model_name]["lm_judge_rating"][f].append(lj)
                acc[model_name]["relevance_concept_ratings"][f].append(rc)
                acc[model_name]["relevance_instruction_ratings"][f].append(ri)
                acc[model_name]["fluency_ratings"][f].append(fl)
                acc_flat[model_name].append(lj)

            # One line per judged generation: every raw rating, ungrouped.
            for i in range(len(current_df)):
                raw_f.write(json.dumps({
                    "concept_id": int(concept_id),
                    "concept": current_df["input_concept"].iloc[i],
                    "model": model_name,
                    "input_id": int(current_df["input_id"].iloc[i]),
                    "factor": float(current_df["factor"].iloc[i]),
                    "instruction": current_df["original_prompt"].iloc[i],
                    "generation": current_df[f"{model_name}_steered_generation"].iloc[i],
                    "relevance_concept": r["raw_relevance_concept_ratings"][i],
                    "relevance_instruction": r["raw_relevance_instruction_ratings"][i],
                    "fluency": r["raw_fluency_ratings"][i],
                    "lm_judge_rating": r["raw_aggregated_ratings"][i],
                }) + "\n")
            raw_f.flush()
            # pool.map yields a concept's models consecutively, so the concept
            # is finished once its last model lands.
            if len(entry["models"]) == len(models):
                pbar.update(1)

    return per_concept, acc, acc_flat, ratings_path


def summarize(acc, acc_flat, models):
    summary = {}
    for m in models:
        by_factor = {f: sum(v) / len(v) for f, v in acc[m]["lm_judge_rating"].items()}
        best_f = max(by_factor, key=by_factor.get) if by_factor else None
        mean_at = lambda key, f: sum(acc[m][key][f]) / len(acc[m][key][f])
        summary[m] = {
            "by_factor": by_factor,
            "by_factor_concept_relevance": {f: mean_at("relevance_concept_ratings", f) for f in by_factor},
            "by_factor_instruction_relevance": {f: mean_at("relevance_instruction_ratings", f) for f in by_factor},
            "by_factor_fluency": {f: mean_at("fluency_ratings", f) for f in by_factor},
            "best_factor": best_f,
            "best_rating": by_factor[best_f] if best_f is not None else None,
            # Ungrouped mean: in select_best's steering_test phase, different concepts
            # contribute rows under different (their own chosen) factor values, so the
            # held-out score is the flat mean over all rows, not any single by_factor entry.
            "mean_rating": sum(acc_flat[m]) / len(acc_flat[m]) if acc_flat[m] else None,
            "n_ratings": len(acc_flat[m]),
        }
    return summary


def write_phase_json(dump_dir, label, factors, args, models, summary, per_concept):
    out = {
        "mode": label,
        "filters": {
            "factors": [float(f) for f in factors],
            "examples_per_concept": args.examples_per_concept,
            "max_concepts": args.max_concepts,
        },
        "lm_model": args.lm_model,
        "models": models,
        "summary": summary,
        "per_concept": per_concept,
    }
    path = dump_dir / f"{label}_subset.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


def print_factor_table(summary, models, factors, title):
    width = max(len(m) for m in models) + 2
    print("\n" + "=" * (14 + width * len(models)))
    print(title)
    print("=" * (14 + width * len(models)))
    print(f"{'factor':>10}  " + "".join(f"{m:>{width}}" for m in models))
    for f in factors:
        row = "".join(f"{summary[m]['by_factor'].get(f, float('nan')):>{width}.3f}" for m in models)
        print(f"{f:>10}  {row}")


def print_single_mode_report(summary, models, factors):
    print_factor_table(summary, models, factors, "MEAN LM JUDGE RATING BY FACTOR")
    print("\n" + "-" * 58)
    print("BEST FACTOR PER METHOD (by mean lm_judge_rating)")
    print("-" * 58)
    for m in models:
        s = summary[m]
        print(f"  {m:<22} best factor {s['best_factor']:>5}  rating {s['best_rating']:.3f}"
              f"   (concept {s['by_factor_concept_relevance'][s['best_factor']]:.3f}, "
              f"instruct {s['by_factor_instruction_relevance'][s['best_factor']]:.3f}, "
              f"fluency {s['by_factor_fluency'][s['best_factor']]:.3f})")
    print()


def print_select_best_report(summary_eval, summary_test, best_factor, models, factors,
                             best_ties=None):
    best_ties = best_ties or {}
    print_factor_table(
        summary_eval, models, factors,
        "PHASE 1: STEERING (eval split) -- MEAN LM JUDGE RATING BY FACTOR")

    print("\n" + "-" * 58)
    print("FACTOR SELECTED PER CONCEPT (how often each candidate factor won)")
    print("[k tie-broken] = won only because ties go to the smallest tied factor")
    print("-" * 58)
    for m in models:
        chosen = [(c, f) for (c, mm), f in best_factor.items() if mm == m]
        for f in factors:
            won = [c for c, x in chosen if x == f]
            if won:
                k = sum(1 for c in won if (c, m) in best_ties)
                tie_note = f"   [{k} tie-broken]" if k else ""
                print(f"  {m:<22} factor {f:>5}: {len(won):>4} concepts "
                      f"({100 * len(won) / len(chosen):.1f}%){tie_note}")
        ties = [t for (c, mm), t in best_ties.items() if mm == m]
        zero = sum(1 for t in ties if t["rating"] == 0)
        print(f"  {m:<22} tie-broken  : {len(ties):>4} concepts "
              f"({100 * len(ties) / max(len(chosen), 1):.1f}%) -- {zero} rated 0 at every "
              f"factor (never steered), {len(ties) - zero} tied at a positive rating")

    print("\n" + "-" * 58)
    print("PHASE 2: HELD-OUT SCORE (best factor per concept, scored on steering_test)")
    print("-" * 58)
    for m in models:
        s = summary_test[m]
        print(f"  {m:<22} held-out rating {s['mean_rating']:.3f}  (n={s['n_ratings']})")
    print()


def main():
    custom_args = [
        {'args': ['--mode'], 'kwargs': {
            'type': str, 'default': 'select_best',
            'help': "steering | steering_test | select_best. select_best evaluates "
                    "--factors on steering, picks each (concept, model)'s own best "
                    "factor, then evaluates steering_test using only that one factor "
                    "per pair instead of every candidate factor."}},
        {'args': ['--factors'], 'kwargs': {
            'type': str, 'default': 'all',
            'help': "Comma-separated steering factors to evaluate, e.g. "
                    "'0.5,1.0,2.5,5.0'. 'all' = every factor present in the data."}},
        {'args': ['--examples_per_concept'], 'kwargs': {
            'type': str, 'default': 'all',
            'help': "Judge only the first N input_ids per concept/split, or 'all'."}},
        {'args': ['--max_concepts'], 'kwargs': {
            'type': str, 'default': 'all',
            'help': "Judge only the first N concepts, or 'all'."}},
        {'args': ['--num_workers'], 'kwargs': {
            'type': int, 'default': 50,
            'help': 'Worker threads issuing judge calls in parallel. The work is '
                    'API-latency bound, so raise this to go faster and lower it if '
                    'the account starts returning 429s.'}},
        {'args': ['--eval_models'], 'kwargs': {
            'type': str, 'default': 'all',
            'help': "Comma-separated steering methods to judge, e.g. "
                    "'MeanTokenDiffMean,DiffMeanPositionalWeighted'. 'all' = every "
                    "model in the yaml's evaluate.models."}},
    ]
    args = EvalArgs(custom_args=custom_args, section="evaluate", ignore_unknown=True)
    for name, default in (("mode", "select_best"), ("factors", "all"),
                          ("examples_per_concept", "all"), ("max_concepts", "all"),
                          ("num_workers", 50), ("eval_models", "all")):
        if not hasattr(args, name):
            setattr(args, name, default)
    args.examples_per_concept = opt_int(args.examples_per_concept)
    args.max_concepts = opt_int(args.max_concepts)

    if args.mode not in ("steering", "steering_test", "select_best"):
        raise SystemExit(
            f"--mode must be steering, steering_test, or select_best, got {args.mode!r}")

    data_dir = Path(args.dump_dir) / "inference"
    dump_dir = Path(args.dump_dir) / "evaluate"
    dump_dir.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_parquet(data_dir / "steering_data.parquet")
    factors = parse_factors(args.factors, df_all["factor"].unique())
    models = [m for m in args.models]
    if args.eval_models != "all":
        requested = [m.strip() for m in args.eval_models.split(",") if m.strip()]
        unknown = [m for m in requested if m not in models]
        if unknown:
            raise SystemExit(f"--eval_models {unknown} not in the yaml's evaluate.models {models}")
        models = requested
    missing_cols = [m for m in models if f"{m}_steered_generation" not in df_all.columns]
    if missing_cols:
        raise SystemExit(f"no generations for {missing_cols} in {data_dir / 'steering_data.parquet'}")
    winrate_split_ratio = getattr(args, "winrate_split_ratio", None)

    evaluator_class = getattr(axbench, "LMJudgeEvaluator")
    # The judge logs a line per task, which would shred the progress bar.
    logging.getLogger("axbench.evaluators.lm_judge").setLevel(logging.ERROR)

    # One client per worker thread, not one shared across them: an AsyncOpenAI's
    # httpx client is bound to the event loop that drives it, and each judge call
    # runs its own asyncio.run(). Reused across both phases of select_best.
    thread_local = threading.local()
    open_clients = []
    clients_lock = threading.Lock()

    def get_lm_model():
        if not hasattr(thread_local, "lm_model"):
            client, lm_model = make_lm_model(args.lm_model, str(dump_dir))
            thread_local.lm_model = lm_model
            with clients_lock:
                open_clients.append((client, lm_model))
        return thread_local.lm_model

    try:
        if args.mode in ("steering", "steering_test"):
            df, concept_ids = select_rows(
                df_all, args.mode, winrate_split_ratio, factors,
                args.examples_per_concept, args.max_concepts)
            print(f"mode={args.mode}  concepts={len(concept_ids)}  factors={factors}\n"
                  f"models={models}  rows={len(df):,}  judge calls={len(df) * len(models) * 3:,}")

            per_concept, acc, acc_flat, ratings_path = run_phase(
                df, concept_ids, models, lambda c, m: factors, dump_dir, args.mode,
                evaluator_class, get_lm_model, args.num_workers, args)
            summary = summarize(acc, acc_flat, models)
            json_path = write_phase_json(dump_dir, args.mode, factors, args, models, summary, per_concept)

            print_single_mode_report(summary, models, factors)
            print(f"detail : {json_path}")
            print(f"ratings: {ratings_path}")

        else:  # select_best
            df_eval, concept_ids = select_rows(
                df_all, "steering", winrate_split_ratio, factors,
                args.examples_per_concept, args.max_concepts)
            print(f"[phase 1/2] steering  concepts={len(concept_ids)}  factors={factors}\n"
                  f"models={models}  rows={len(df_eval):,}  "
                  f"judge calls={len(df_eval) * len(models) * 3:,}")
            per_concept_eval, acc_eval, acc_flat_eval, ratings_path_eval = run_phase(
                df_eval, concept_ids, models, lambda c, m: factors, dump_dir, "steering",
                evaluator_class, get_lm_model, args.num_workers, args)
            summary_eval = summarize(acc_eval, acc_flat_eval, models)
            write_phase_json(dump_dir, "steering", factors, args, models, summary_eval, per_concept_eval)

            # Best factor per (concept, model), straight from phase 1's own per-task
            # results -- the same argmax evaluate.py's get_best_factors() computes but
            # never wires into steering_test.
            # list.index(max) breaks ties toward the first (smallest) factor, so a
            # concept rated 0 at every factor "picks" the smallest one. best_ties records
            # every factor that shares the max, so the report can separate those.
            best_factor, best_ties = {}, {}
            for entry in per_concept_eval:
                c = entry["concept_id"]
                for m in models:
                    if m not in entry["models"]:
                        continue
                    fr = entry["models"][m]["factor"]
                    lj = entry["models"][m]["lm_judge_rating"]
                    if fr:
                        best_factor[(c, m)] = fr[lj.index(max(lj))]
                        tied = [f for f, r in zip(fr, lj) if math.isclose(r, max(lj), abs_tol=1e-9)]
                        if len(tied) > 1:
                            best_ties[(c, m)] = {"factors": tied, "rating": max(lj)}

            df_test, concept_ids_test = select_rows(
                df_all, "steering_test", winrate_split_ratio, factors,
                args.examples_per_concept, args.max_concepts)
            n_pairs = sum(1 for c in concept_ids_test for m in models if (c, m) in best_factor)
            print(f"\n[phase 2/2] steering_test  {n_pairs} (concept, model) pairs, "
                  f"1 factor each (vs {len(concept_ids_test) * len(models) * len(factors)} "
                  f"judge-row-sets if all {len(factors)} candidate factors were kept)")
            per_concept_test, acc_test, acc_flat_test, ratings_path_test = run_phase(
                df_test, concept_ids_test, models,
                lambda c, m: [best_factor[(c, m)]] if (c, m) in best_factor else [],
                dump_dir, "steering_test", evaluator_class, get_lm_model, args.num_workers, args)
            summary_test = summarize(acc_test, acc_flat_test, models)
            write_phase_json(dump_dir, "steering_test", factors, args, models, summary_test, per_concept_test)

            summary_path = dump_dir / "select_best_summary.json"
            with open(summary_path, "w") as f:
                json.dump({
                    "mode": "select_best",
                    "filters": {"factors": [float(x) for x in factors],
                                "examples_per_concept": args.examples_per_concept,
                                "max_concepts": args.max_concepts},
                    "lm_model": args.lm_model,
                    "models": models,
                    "best_factor_per_concept_model": {
                        f"{c}:{m}": f for (c, m), f in best_factor.items()},
                    "best_factor_ties": {
                        f"{c}:{m}": t for (c, m), t in best_ties.items()},
                    "summary_steering": summary_eval,
                    "summary_steering_test": summary_test,
                }, f, indent=2)

            print_select_best_report(summary_eval, summary_test, best_factor, models, factors,
                                     best_ties)
            print(f"steering detail      : {dump_dir / 'steering_subset.json'}")
            print(f"steering_test detail : {dump_dir / 'steering_test_subset.json'}")
            print(f"summary               : {summary_path}")
    finally:
        for client, _ in open_clients:
            asyncio.run(client.close())

    total_price = sum(lm.stats.get_total_price() for _, lm in open_clients)
    print(f"\ntotal price      : ${total_price:.2f}  ({len(open_clients)} worker client(s))")

    for path in build_previews(dump_dir, data_dir):
        print(f"preview          : {path}")


if __name__ == "__main__":
    main()
