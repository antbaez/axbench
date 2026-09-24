"""Summarize the residual-stream norms collected by collect_norms.py.

Two regimes, because they are not equally comparable:

  Prompt tokens are identical across factors (same prompt, same tokenization) and are
  where the prompt-only interventions actually add their vector, so every position can
  be paired with its factor-0 counterpart: the ratio norm(factor)/norm(0) is exact.

  Generated tokens differ across factors, since the text itself differs, so there is no
  per-position pairing. These are summarized as plain distribution statistics (mean,
  median, std, min, max, percentiles) and compared against the same method's factor-0
  row.

The question being asked is which method's generated-token norms stay closest to its
own unsteered baseline at comparable steering strength.

Input is what inference.py --mode steering writes when the yaml's inference block sets
capture_norms: true -- one concept{cid}_rank{r}[_chunk{c}].parquet per concept under
{dump}/inference/norms/, one row per (model, concept, prompt, factor).

Example:
    uv run python mech_analysis/norms/analyze_norms.py \
        --norms_dir <dump>/inference/norms --plots
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--norms_dir", required=True, help="{dump}/inference/norms")
    p.add_argument("--out_dir", default=None, help="default: {norms_dir}/analysis")
    p.add_argument("--field", default="norms_post", choices=["norms_post", "norms_pre"],
                   help="norms_post is the steered residual stream the model consumes")
    p.add_argument("--plots", action="store_true", help="also write figures (needs matplotlib)")
    return p.parse_args()


def load(norms_dir):
    # oldest first, so a concept redone after preemption keeps its latest records
    files = sorted(Path(norms_dir).glob("*.parquet"), key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no *.parquet under {norms_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["model", "concept_id", "input_id", "factor"], keep="last")
    if len(df) < before:
        print(f"dropped {before - len(df)} duplicate records (concepts written twice)")
    print(f"loaded {len(df)} sequences from {len(files)} files: "
          f"{df['model'].nunique()} models, {df['concept_id'].nunique()} concepts, "
          f"factors {sorted(df['factor'].unique())}")
    return df


def split(row, field):
    """(prompt norms, generated norms) for one sequence."""
    v = np.asarray(row[field], dtype=np.float64)
    n_p = int(row["n_prompt"])
    return v[:n_p], v[n_p:]


def simple_stats(values, prefix=""):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {}
    return {
        f"{prefix}n": int(v.size),
        f"{prefix}mean": float(v.mean()),
        f"{prefix}median": float(np.median(v)),
        f"{prefix}std": float(v.std(ddof=0)),
        f"{prefix}min": float(v.min()),
        f"{prefix}max": float(v.max()),
        f"{prefix}p5": float(np.percentile(v, 5)),
        f"{prefix}p95": float(np.percentile(v, 95)),
    }


def generated_summary(df, field):
    """Per (model, factor): plain stats over all generated-token norms, vs factor 0."""
    rows = []
    for (model, factor), grp in df.groupby(["model", "factor"]):
        gen = np.concatenate([split(r, field)[1] for _, r in grp.iterrows()] or [np.array([])])
        rec = {"model": model, "factor": factor,
               "mean_strength": float(grp["strength"].mean()),
               "n_sequences": len(grp)}
        rec.update(simple_stats(gen, "gen_"))
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(["model", "factor"]).reset_index(drop=True)

    # ratios against the same model's factor-0 row
    base = out[out["factor"] == 0.0].set_index("model")
    if base.empty:
        print("WARNING: no factor 0.0 rows -- ratios to the unsteered baseline are skipped.")
        return out
    out["gen_mean_ratio"] = out.apply(lambda r: r["gen_mean"] / base.loc[r["model"], "gen_mean"], axis=1)
    out["gen_std_ratio"] = out.apply(lambda r: r["gen_std"] / base.loc[r["model"], "gen_std"], axis=1)
    # share of generated tokens outside the unsteered [p5, p95] band
    frac = []
    for _, r in out.iterrows():
        lo, hi = base.loc[r["model"], "gen_p5"], base.loc[r["model"], "gen_p95"]
        grp = df[(df["model"] == r["model"]) & (df["factor"] == r["factor"])]
        gen = np.concatenate([split(g, field)[1] for _, g in grp.iterrows()] or [np.array([])])
        frac.append(float(((gen < lo) | (gen > hi)).mean()) if gen.size else np.nan)
    out["gen_frac_outside_baseline_p5_p95"] = frac
    return out


def prompt_summary(df, field):
    """Per (model, factor): paired norm(factor)/norm(0) over prompt positions.

    Pairing is by (concept_id, input_id) and position index, which is valid because the
    prompt tokens are identical at every factor.
    """
    base = {}
    for _, r in df[df["factor"] == 0.0].iterrows():
        base[(r["model"], r["concept_id"], r["input_id"])] = split(r, field)[0]
    if not base:
        print("WARNING: no factor 0.0 rows -- the paired prompt analysis is skipped.")
        return pd.DataFrame()

    rows = []
    for (model, factor), grp in df.groupby(["model", "factor"]):
        ratios, deltas = [], []
        for _, r in grp.iterrows():
            b = base.get((r["model"], r["concept_id"], r["input_id"]))
            if b is None:
                continue
            p = split(r, field)[0]
            n = min(len(b), len(p))
            if n == 0:
                continue
            ratios.append(p[-n:] / b[-n:])
            deltas.append(p[-n:] - b[-n:])
        if not ratios:
            continue
        ratios = np.concatenate(ratios)
        deltas = np.concatenate(deltas)
        rec = {"model": model, "factor": factor,
               "mean_strength": float(grp["strength"].mean())}
        rec.update(simple_stats(ratios, "prompt_ratio_"))
        rec.update(simple_stats(deltas, "prompt_delta_"))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["model", "factor"]).reset_index(drop=True)


def position_curve(df, field, max_positions=256):
    """Mean generated-token norm as a function of generation step."""
    rows = []
    for (model, factor), grp in df.groupby(["model", "factor"]):
        acc = np.zeros(max_positions)
        cnt = np.zeros(max_positions)
        for _, r in grp.iterrows():
            g = split(r, field)[1][:max_positions]
            acc[:len(g)] += g
            cnt[:len(g)] += 1
        valid = cnt > 0
        for pos in np.nonzero(valid)[0]:
            rows.append({"model": model, "factor": factor, "gen_position": int(pos),
                         "mean_norm": acc[pos] / cnt[pos], "n": int(cnt[pos])})
    return pd.DataFrame(rows)


def prompt_position_curve(df, field, max_positions=128):
    """Mean paired ratio by distance back from the last prompt token (0 = last)."""
    base = {}
    for _, r in df[df["factor"] == 0.0].iterrows():
        base[(r["model"], r["concept_id"], r["input_id"])] = split(r, field)[0]
    if not base:
        return pd.DataFrame()
    rows = []
    for (model, factor), grp in df.groupby(["model", "factor"]):
        acc = np.zeros(max_positions)
        cnt = np.zeros(max_positions)
        for _, r in grp.iterrows():
            b = base.get((r["model"], r["concept_id"], r["input_id"]))
            if b is None:
                continue
            p = split(r, field)[0]
            n = min(len(b), len(p), max_positions)
            if n == 0:
                continue
            ratio = (p[-n:] / b[-n:])[::-1]      # index 0 = last prompt token
            acc[:n] += ratio
            cnt[:n] += 1
        for pos in np.nonzero(cnt > 0)[0]:
            rows.append({"model": model, "factor": factor, "tokens_from_end": int(pos),
                         "mean_ratio": acc[pos] / cnt[pos], "n": int(cnt[pos])})
    return pd.DataFrame(rows)


def make_plots(gen_curve, prompt_curve, gen_df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, curve, xcol, ycol, xlabel, ylabel in [
        ("gen_norm_by_position", gen_curve, "gen_position", "mean_norm",
         "generated token index", "mean layer-L norm"),
        ("prompt_ratio_by_position", prompt_curve, "tokens_from_end", "mean_ratio",
         "tokens back from last prompt token", "mean norm / unsteered norm"),
    ]:
        if curve is None or curve.empty:
            continue
        models = sorted(curve["model"].unique())
        fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4), squeeze=False)
        for ax, m in zip(axes[0], models):
            sub = curve[curve["model"] == m]
            for f in sorted(sub["factor"].unique()):
                s = sub[sub["factor"] == f]
                ax.plot(s[xcol], s[ycol], label=f"factor {f:g}", linewidth=1)
            ax.set_title(m, fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        axes[0][-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"{name}.png", dpi=150)
        plt.close(fig)

    if "gen_mean_ratio" in gen_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in sorted(gen_df["model"].unique()):
            s = gen_df[gen_df["model"] == m].sort_values("mean_strength")
            ax.plot(s["mean_strength"], s["gen_mean_ratio"], marker="o", label=m)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("mean steering strength (factor x max_act)")
        ax.set_ylabel("generated-token mean norm / unsteered")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(Path(out_dir) / "gen_mean_ratio_vs_strength.png", dpi=150)
        plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.norms_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(args.norms_dir)
    gen = generated_summary(df, args.field)
    prompt = prompt_summary(df, args.field)
    gen_curve = position_curve(df, args.field)
    prompt_curve = prompt_position_curve(df, args.field)

    gen.to_csv(out_dir / "generated_token_stats.csv", index=False)
    if not prompt.empty:
        prompt.to_csv(out_dir / "prompt_token_paired_stats.csv", index=False)
    gen_curve.to_csv(out_dir / "generated_norm_by_position.csv", index=False)
    if not prompt_curve.empty:
        prompt_curve.to_csv(out_dir / "prompt_ratio_by_position.csv", index=False)

    cols = ["model", "factor", "mean_strength", "gen_mean", "gen_median", "gen_std",
            "gen_min", "gen_max"]
    cols += [c for c in ["gen_mean_ratio", "gen_std_ratio",
                         "gen_frac_outside_baseline_p5_p95"] if c in gen.columns]
    print("\n=== generated tokens ===")
    print(gen[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    if not prompt.empty:
        pcols = ["model", "factor", "mean_strength", "prompt_ratio_mean",
                 "prompt_ratio_median", "prompt_ratio_std", "prompt_ratio_max",
                 "prompt_delta_mean"]
        print("\n=== prompt tokens (paired against factor 0) ===")
        print(prompt[pcols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    if args.plots:
        make_plots(gen_curve, prompt_curve, gen, out_dir)
        print(f"\nfigures written to {out_dir}")
    print(f"\ntables written to {out_dir}")


if __name__ == "__main__":
    main()
