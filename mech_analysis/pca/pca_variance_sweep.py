"""
How much variance the first N components capture, across many concepts.

    uv run --no-sync mech_analysis/pca/pca_variance_sweep.py [--n_concepts 20]

pca.py is a deep dive on one concept; this is the same PCA (same coverage filter, same
centering, same scaling options) run across a range of concepts, reporting only the
explained-variance curve. The checkpoint is loaded once and reused for every concept.

Each concept keeps its own positions with r_k >= --coverage, so the number of points --
and therefore the maximum possible rank, n-1 -- differs per concept. That matters for
reading the table: a concept with 12 kept positions can only ever spread its variance over
11 components, so its cumulative curve rises faster than one with 25 positions for reasons
that have nothing to do with the representation. The n column is there to be read
alongside the percentages, and the "chance" line in the summary is the flat 1/(n-1) curve
you would get from points in general position with no structure at all.

Outputs:
    pca_figures/{dump}_first{N}_cov{C}_variance.png   cumulative curves, one per concept
    pca_results/{dump}_first{N}_cov{C}_variance.csv   the per-concept table
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

DEFAULT_DUMP = "axbench/results/prod_9b_l20_concept500_diffmean_pos_steer_data"
POS_MODEL = "DiffMeanPositionalWeighted"


def short_name(dump_dir):
    name = Path(dump_dir).resolve().name
    return re.sub(r"^prod_.*?_diffmean_", "", name) or name


def load_weight(train_dir, model_name):
    merged = train_dir / f"{model_name}_weight.pt"
    if merged.exists():
        return torch.load(merged, map_location="cpu", weights_only=True)
    rank_files = sorted(train_dir.glob(f"rank_*_{model_name}_weight.pt"),
                        key=lambda p: int(p.name.split("_")[1]))
    if not rank_files:
        raise FileNotFoundError(f"no {model_name} weights in {train_dir}")
    parts = [torch.load(f, map_location="cpu", weights_only=True) for f in rank_files]
    if isinstance(parts[0], dict):
        return {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}
    return torch.cat(parts, dim=0)


def concept_rows(train_dir, n):
    """(row, concept_id, concept) for the first n concepts, in metadata order."""
    meta = train_dir / "metadata.jsonl"
    rows = [json.loads(line) for line in meta.read_text().splitlines() if line.strip()]
    return [(i, rows[i].get("concept_id", i), rows[i].get("concept"))
            for i in range(min(n, len(rows)))]


def evr_for(x, real_frac, coverage, unit_norm, no_center, n_comp):
    """Explained-variance ratios for one concept, padded to n_comp. None if too few."""
    keep = (x.norm(dim=1).numpy() > 0) & (real_frac >= coverage)
    if keep.sum() < 3:
        return None, int(keep.sum())
    v = x[torch.from_numpy(keep)]
    if unit_norm:
        v = v / v.norm(dim=1, keepdim=True)
    if not no_center:
        v = v - v.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(v)
    evr = (s ** 2 / (s ** 2).sum()).numpy()
    out = np.zeros(n_comp)
    out[:min(n_comp, len(evr))] = evr[:n_comp]
    return out, int(keep.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump_dir", default=DEFAULT_DUMP)
    p.add_argument("--n_concepts", type=int, default=20)
    p.add_argument("--coverage", type=float, default=0.9)
    p.add_argument("--n_components", type=int, default=10)
    p.add_argument("--unit_norm", action="store_true")
    p.add_argument("--no_center", action="store_true")
    p.add_argument("--prefix", default=None)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    train_dir = Path(args.dump_dir) / "train"
    w = load_weight(train_dir, POS_MODEL)          # loaded once for every concept
    pos, rf = w["positional_col"], w["real_frac"]
    nc = args.n_components

    rows, curves, labels = [], [], []
    for row, cid, name in concept_rows(train_dir, args.n_concepts):
        evr, n_keep = evr_for(pos[row].double(), rf[row].double().numpy(),
                              args.coverage, args.unit_norm, args.no_center, nc)
        if evr is None:
            print(f"[warn] concept {cid}: only {n_keep} position(s) at "
                  f"r_k >= {args.coverage:.0%}; skipped")
            continue
        cum = np.cumsum(evr)
        rows.append([cid, n_keep, *evr, *cum])
        curves.append(cum)
        labels.append(cid)

    if not curves:
        raise SystemExit("no concept had enough positions; lower --coverage")
    curves = np.array(curves)
    mean_cum = curves.mean(axis=0)
    ks = np.arange(1, nc + 1)
    n_med = int(np.median([r[1] for r in rows]))
    chance = np.cumsum(np.full(nc, 1.0 / max(n_med - 1, 1)))

    print(f"\n{POS_MODEL}, {len(rows)} concepts, r_k >= {args.coverage:.0%}, "
          f"{'unit-normalized' if args.unit_norm else 'as stored'}, "
          f"{'uncentered' if args.no_center else 'centered'}")
    head = f"{'concept':>7} {'n':>4} " + " ".join(f"{'PC1-' + str(k):>7}" for k in (1, 3, 5, nc))
    print(head)
    print("-" * len(head))
    for r in rows:
        cum = r[2 + nc:]
        print(f"{int(r[0]):>7} {int(r[1]):>4} " +
              " ".join(f"{cum[k - 1]:>6.1%} " for k in (1, 3, 5, nc)))
    print("-" * len(head))
    print(f"{'mean':>7} {np.mean([r[1] for r in rows]):>4.1f} " +
          " ".join(f"{mean_cum[k - 1]:>6.1%} " for k in (1, 3, 5, nc)))
    print(f"{'chance':>7} {n_med:>4} " +
          " ".join(f"{min(chance[k - 1], 1.0):>6.1%} " for k in (1, 3, 5, nc))
          + "  (flat 1/(n-1) at the median n)")

    print(f"\nmean per-component share across concepts:")
    print(f"    {'PC':>3}  {'individual':>10}  {'cumulative':>10}")
    mean_ind = np.diff(np.concatenate([[0.0], mean_cum]))
    for i in range(nc):
        print(f"    {i + 1:>3}  {mean_ind[i]:>9.1%}  {mean_cum[i]:>10.1%}")

    here = Path(__file__).resolve().parent
    out_dir, fig_dir = here / "pca_results", here / "pca_figures"
    out_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    dump_name = args.prefix or short_name(args.dump_dir)
    tag = (f"first{args.n_concepts}_cov{int(round(args.coverage * 100))}"
           + ("_unitnorm" if args.unit_norm else "") + ("_uncentered" if args.no_center else ""))
    stem = f"{dump_name}_{tag}_variance"

    header = (["concept_id", "n_positions"] + [f"evr_pc{k}" for k in ks]
              + [f"cum_pc1_{k}" for k in ks])
    with open(out_dir / f"{stem}.csv", "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join([str(int(r[0])), str(int(r[1]))]
                             + [f"{v:.6f}" for v in r[2:]]) + "\n")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for cum, cid in zip(curves, labels):
        ax.plot(ks, cum, color="grey", alpha=0.35, lw=1)
    ax.plot(ks, mean_cum, color="#1f77b4", lw=2.6, marker="o", ms=5,
            label=f"mean of {len(rows)} concepts")
    ax.plot(ks, np.minimum(chance, 1.0), color="#d62728", lw=1.6, ls="--",
            label=f"no structure: flat 1/(n-1), median n={n_med}")
    ax.set(xlabel="components", ylabel="cumulative explained variance",
           xticks=ks, ylim=(0, 1.02),
           title=f"{POS_MODEL}: variance captured across first {len(rows)} concepts\n"
                 f"r_k >= {args.coverage:.0%}, "
                 f"{'unit-normalized' if args.unit_norm else 'as stored'}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{stem}.png", dpi=200)
    plt.close(fig)
    print(f"\nwrote {fig_dir}/{stem}.png and {out_dir}/{stem}.csv")


if __name__ == "__main__":
    main()
