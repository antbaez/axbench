"""
Cosine similarity between one concept's DiffMeanPositionalWeighted per-position vectors
and its two single-vector counterparts.

    uv run mech_analysis/cosine_sim/cosine_sim.py [--dump_dir DIR] [--concept_id N]

Three directions exist per concept:

    MeanTokenDiffMean           one vector, pooled over every real token
    LastTokenDiffMean           one vector, the final real token only
    DiffMeanPositionalWeighted  num_positions vectors (r_k * v_k), one per end-aligned slot

Positions are plotted in sequence order, exactly as "positional_col" stores them: index 0
is the deepest slot (num_positions tokens back from the end) and the last index is the
final token. Nothing is flipped anywhere in this script.

Cosine is scale-invariant, so the r_k coverage weighting cancels everywhere here: every
number is identical for DiffMeanPositional and its Weighted subclass. The one exception is
a slot with r_k == 0 -- a slot no training row was ever long enough to reach, so always a
run of *leading* slots in sequence order. Its stored vector is exactly zero and its
direction is unrecoverable, so it is reported as NaN rather than as 0.

Cosine is signed rather than squared: a steering vector's sign is meaningful (+v steers
toward the concept, -v away), so anti-aligned positions should read as negative.

Figures go to cosine_sim_figures/ next to this file, raw arrays to cosine_sim_results/:
    {dump}_concept{N}_vs_mean_last.png   per-position cosine against the mean/final
                                         vectors, with the mean-vs-final scalar in the key
    {dump}_concept{N}_pairwise.png       P x P cosine between every pair of positions
    {dump}_concept{N}.npz                the raw arrays

Positions whose coverage r_k is below --coverage (default 90%) are left out of both
figures, since below that a position's vector is a diff-of-means over a handful of long
examples padded out with template/padding activations. The npz still holds every position,
so nothing is lost -- only the figures are trimmed. The surviving r_k curve is drawn
alongside the cosines.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch

DEFAULT_DUMP = "axbench/results/prod_9b_l20_concept500_diffmean_pos_steer_data_v11"
POS_MODEL = "DiffMeanPositionalWeighted"
MEAN_MODEL = "MeanTokenDiffMean"
LAST_MODEL = "LastTokenDiffMean"
POS_LABEL = "position (sequence order; last = final token)"

# Gemma-2's chat template ends every training example on its fixed
# add_generation_prompt=True suffix -- "<end_of_turn>\n<start_of_turn>model\n" -- not on
# response content (see prepare_df in scripts/train.py), so positions -4..0 relative to
# the last token are always these five literal tokens regardless of concept or
# instruction (verified against the tokenizer). Anything further back is
# instruction-dependent and intentionally left unlabeled.
CHAT_SUFFIX_TOKENS = {0: "\\n", -1: "model", -2: "<start_of_turn>", -3: "\\n",
                       -4: "<end_of_turn>"}


def short_name(dump_dir):
    """Filename stem for a dump dir: the run-specific tail, with the shared
    prod_<size>_<layer>_concept<N>_diffmean_ prefix stripped --
    prod_9b_l20_concept500_diffmean_pos_steer_data_v2 -> pos_steer_data_v2.

    Dumps that differ only inside the stripped prefix (a 2b and a 9b run of the same
    data) therefore produce the same filenames; pass --prefix to disambiguate.
    """
    name = Path(dump_dir).resolve().name
    return re.sub(r"^prod_.*?_diffmean_", "", name) or name


def load_weight(train_dir, model_name):
    """Merged {model}_weight.pt if present, else rank_*_ files concatenated in rank
    order (what train.py's merge would produce)."""
    merged = train_dir / f"{model_name}_weight.pt"
    if merged.exists():
        return torch.load(merged, map_location="cpu", weights_only=True)
    rank_files = sorted(train_dir.glob(f"rank_*_{model_name}_weight.pt"),
                        key=lambda p: int(p.name.split("_")[1]))
    if not rank_files:
        raise FileNotFoundError(f"no {model_name} weights in {train_dir}")
    print(f"[warn] {merged.name} missing; concatenating {len(rank_files)} rank file(s)")
    parts = [torch.load(f, map_location="cpu", weights_only=True) for f in rank_files]
    if isinstance(parts[0], dict):
        return {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}
    return torch.cat(parts, dim=0)


def concept_row(train_dir, concept_id):
    """Row of the merged tensors holding concept_id (metadata.jsonl is in row order)."""
    meta = train_dir / "metadata.jsonl"
    if not meta.exists():
        return concept_id, None
    rows = [json.loads(line) for line in meta.read_text().splitlines() if line.strip()]
    for i, row in enumerate(rows):
        if row.get("concept_id") == concept_id:
            return i, row.get("concept")
    raise ValueError(f"concept_id {concept_id} not in {meta}")


def unit(x):
    """Unit-normalize along the last dim; rows that are exactly zero stay zero."""
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def position_coverage(pos_w, row):
    """positional_col row, its r_k (real_frac), and its dead (r_k == 0) mask -- the
    coverage bookkeeping shared by every per-concept computation in this file."""
    x = pos_w["positional_col"][row].double()
    norms = x.norm(dim=1).numpy()
    real_frac = pos_w["real_frac"][row].double().numpy() if "real_frac" in pos_w else norms
    dead = (x.norm(dim=1) <= 0).numpy()
    return x, real_frac, dead


def position_cosines(pos_w, mean_w, last_w, row):
    """Per-position real_frac / cos-vs-mean / cos-vs-last / dead mask for one concept
    row, i.e. the same quantities the single-concept plot uses, factored out so the
    all-concepts aggregate can reuse them without duplicating the coverage/dead logic."""
    x, real_frac, dead = position_coverage(pos_w, row)
    xn = unit(x)
    mean_v = unit(mean_w[row].double())
    last_v = unit(last_w[row].double())
    cos_mean = (xn @ mean_v).numpy()
    cos_last = (xn @ last_v).numpy()
    cos_mean[dead] = np.nan
    cos_last[dead] = np.nan
    return real_frac, cos_mean, cos_last, dead


def plot_all_concepts_mean_cosine(train_dir, pos_w, mean_w, last_w, coverage, dump_name,
                                   out_dir, fig_dir):
    """Average the per-position cosine-vs-MeanTokenDiffMean / cosine-vs-LastTokenDiffMean
    curves across every concept in the dump. A concept contributes to position k's average
    only if that concept clears the same r_k >= coverage threshold there (and the slot
    isn't dead) that the single-concept figure uses to decide what to keep -- so this is
    the same 'keep' mask, just applied concept-by-concept and averaged instead of drawn
    per concept. Positions are never dropped from the x-axis: where fewer concepts (or
    none) qualify, the point is still plotted (as NaN if the count is zero) and the
    per-position concept count is drawn alongside so a thin average is visible rather than
    silently blended in with the well-supported positions."""
    meta = train_dir / "metadata.jsonl"
    rows = [json.loads(line) for line in meta.read_text().splitlines() if line.strip()]
    n_concepts = len(rows)
    P = pos_w["positional_col"].shape[1]

    sum_mean = np.zeros(P)
    sum_last = np.zeros(P)
    count = np.zeros(P, dtype=int)
    for i in range(n_concepts):
        real_frac, cos_mean, cos_last, dead = position_cosines(pos_w, mean_w, last_w, i)
        keep = (~dead) & (real_frac >= coverage)
        sum_mean[keep] += cos_mean[keep]
        sum_last[keep] += cos_last[keep]
        count[keep] += 1

    with np.errstate(invalid="ignore"):
        avg_mean = np.where(count > 0, sum_mean / np.maximum(count, 1), np.nan)
        avg_last = np.where(count > 0, sum_last / np.maximum(count, 1), np.nan)

    idx = np.arange(P)
    cov_label = f"r_k >= {coverage:.0%}"
    stem = f"{dump_name}_all_concepts"

    np.savez(out_dir / f"{stem}.npz", position=idx, mean_cos_vs_mean=avg_mean,
             mean_cos_vs_last=avg_last, n_concepts_at_position=count,
             n_concepts_total=np.array(n_concepts))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(idx, avg_mean, marker="o", ms=3, lw=1.4, color="#1f77b4",
            label=f"mean cos(positional, {MEAN_MODEL})")
    ax.plot(idx, avg_last, marker="o", ms=3, lw=1.4, color="#d62728",
            label=f"mean cos(positional, {LAST_MODEL})")
    ax.axhline(0, color="grey", lw=0.8, zorder=0)
    ax.set(xlabel=POS_LABEL, ylabel="mean cosine similarity", ylim=(-1.02, 1.02),
           xlim=(idx[0] - 1, idx[-1] + 1),
           title=f"{POS_MODEL} positions vs single-vector methods, "
                 f"averaged over {n_concepts} concepts ({cov_label})\n{dump_name}")
    ax.legend(fontsize=8, loc="lower left")

    ax2 = ax.twinx()
    ax2.plot(idx, count, color="seagreen", lw=1.0, alpha=0.55, zorder=0,
             label="# concepts contributing")
    ax2.set_ylabel("# concepts at this position", color="seagreen")
    ax2.tick_params(axis="y", labelcolor="seagreen")
    ax2.set_ylim(0, n_concepts * 1.05)

    fig.tight_layout()
    fig.savefig(fig_dir / f"{stem}_vs_mean_last.png", dpi=400)
    plt.close(fig)

    # --- same curves, trimmed to well-covered positions only ---
    # a copy of the figure above, restricted to positions where at least half the dump's
    # concepts contributed (matching the coverage bar used for the pairwise plot), with
    # the per-position concept-count line/axis dropped since every plotted point is by
    # construction backed by a comparable share of the dataset
    well_covered = count >= 0.5 * n_concepts
    kidx2 = idx[well_covered]
    if len(kidx2) >= 2:
        # relative to the final (last) token: 0 at the last token, negative counting
        # back from it, instead of the absolute sequence-order index used elsewhere
        rel2 = kidx2 - (P - 1)
        fig, ax = plt.subplots(figsize=(8, 6.8))
        ax.plot(rel2, avg_mean[well_covered], marker="o", ms=5, lw=1.8, color="#1f77b4",
                label="Mean cos(Mean Token, Positional)")
        ax.plot(rel2, avg_last[well_covered], marker="o", ms=5, lw=1.8, color="#d62728",
                label="Mean cos(Last Token, Positional)")
        ax.axhline(0, color="grey", lw=0.8, zorder=0)
        ax.set(ylim=(0, 1.02), xlim=(rel2[0] - 0.5, rel2[-1] + 0.5))
        ax.set_xlabel("Position (From Last Token)", fontsize=15)
        ax.set_ylabel("Mean Cosine Similarity", fontsize=15)
        ax.set_title("Mean Cosine Similarity Between Steering Vectors\n"
                     f"(r_k >= {coverage:.1f}, concept coverage >= 0.5)", fontsize=17)
        ax.tick_params(labelsize=12)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        # drawn as rotated annotations below the numeric ticks rather than baked into
        # the tick labels themselves, since the token text is wider than the 1-unit
        # tick spacing and would otherwise overlap its neighbors; rotation_mode="anchor"
        # keeps the (unrotated) anchor point -- not the rotated bounding-box center --
        # pinned under the tick, which is what actually lines the text up with it
        for xi, tok in CHAT_SUFFIX_TOKENS.items():
            if rel2[0] <= xi <= rel2[-1]:
                ax.annotate(tok, xy=(xi, 0), xycoords=("data", "axes fraction"),
                            xytext=(0, -22), textcoords="offset points",
                            rotation=45, rotation_mode="anchor", ha="right", va="top",
                            fontsize=12, annotation_clip=False)
        fig.subplots_adjust(bottom=0.3)
        ax.legend(fontsize=13, loc="upper left")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{stem}_vs_mean_last_min50.png", dpi=400)
        plt.close(fig)
        print(f"wrote {fig_dir}/{stem}_vs_mean_last_min50.png "
              f"({len(kidx2)}/{P} positions, pos {kidx2[0]}-{kidx2[-1]}, "
              f"rel {rel2[0]}-{rel2[-1]})")
    else:
        print(f"[warn] fewer than 2 positions reach 50% concept coverage; skipping "
              f"{stem}_vs_mean_last_min50.png")

    thin = np.flatnonzero((count > 0) & (count < n_concepts))
    print(f"all-concepts average ({n_concepts} concepts, {cov_label}): "
          f"positions with full coverage {int((count == n_concepts).sum())}/{P}, "
          f"zero coverage {int((count == 0).sum())}/{P}")
    if len(thin):
        print(f"  partial coverage at {len(thin)} position(s), count range "
              f"{count[thin].min()}-{count[thin].max()} (see {stem}_vs_mean_last.png)")
    print(f"wrote {fig_dir}/{stem}_vs_mean_last.png and {out_dir}/{stem}.npz")


def plot_all_concepts_pairwise(train_dir, pos_w, coverage, dump_name, out_dir, fig_dir,
                                annotate_max):
    """P x P mean pairwise cosine similarity between positions, averaged over every
    concept in the dump. A concept contributes to cell (a, b) only if *both* position a
    and position b clear the same r_k >= coverage threshold used everywhere else in this
    file (and neither is dead) -- so, unlike the per-position average above, coverage is
    a joint condition on the pair, not each position independently. Positions that never
    clear the threshold for any concept (diagonal count == 0) are dropped from the axes,
    same as the single-concept pairwise plot drops its own dead/low-coverage positions;
    everything else is kept even where only a handful of concepts back a given cell, and
    the per-cell contributing-concept count is saved alongside for that reason."""
    meta = train_dir / "metadata.jsonl"
    rows = [json.loads(line) for line in meta.read_text().splitlines() if line.strip()]
    n_concepts = len(rows)
    P = pos_w["positional_col"].shape[1]

    sum_mat = np.zeros((P, P))
    count_mat = np.zeros((P, P), dtype=int)
    for i in range(n_concepts):
        x, real_frac, dead = position_coverage(pos_w, i)
        keep = (~dead) & (real_frac >= coverage)
        cmat = (unit(x) @ unit(x).T).numpy()
        mask = keep[:, None] & keep[None, :]
        sum_mat[mask] += cmat[mask]
        count_mat += mask

    avg_mat = np.full((P, P), np.nan)
    nz = count_mat > 0
    avg_mat[nz] = sum_mat[nz] / count_mat[nz]

    # keep only positions where at least half the dump's concepts clear the r_k
    # threshold there -- a thin diagonal count (a handful of concepts) makes that
    # position's whole row/column noise, so it's dropped rather than plotted
    well_covered = (count_mat.diagonal() / n_concepts) >= 0.5
    kidx = np.arange(P)[well_covered]
    if len(kidx) < 2:
        print(f"[warn] fewer than 2 positions reach 50% concept coverage at "
              f"r_k >= {coverage:.0%} across all {n_concepts} concepts; skipping "
              f"all-concepts pairwise plot")
        return
    sub = avg_mat[np.ix_(kidx, kidx)]
    csub = count_mat[np.ix_(kidx, kidx)]
    n = len(kidx)
    cov_label = f"r_k >= {coverage:.0%}"
    stem = f"{dump_name}_all_concepts"

    np.savez(out_dir / f"{stem}_pairwise.npz", position=kidx, mean_cos_matrix=sub,
             count_matrix=csub, n_concepts_total=np.array(n_concepts))

    # viridis (purple -> blue -> green -> yellow) instead of the diverging RdBu_r --
    # these means are all positive, so a sequential colormap uses its full range instead
    # of wasting the half meant for negative values
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("lightgrey")
    fig, ax = plt.subplots(figsize=(12, 11))
    # scale to the data's own range rather than the full [-1, 1] -- these means are all
    # positive (no anti-aligned pairs at this coverage), so a fixed [-1, 1] scale wastes
    # the whole bottom half of the colormap and washes the real range into one shade
    vlo = float(np.nanmin(sub)) if np.isfinite(sub).any() else 0.0
    vhi = float(np.nanmax(sub)) if np.isfinite(sub).any() else 1.0
    vlo = min(vlo, 0.0)
    im = ax.imshow(sub, vmin=vlo, vmax=vhi, cmap=cmap, origin="upper")
    frac = csub / n_concepts
    # CELL-ANNOTATIONS: cosine value + concept-coverage fraction printed in each cell.
    # Disabled for a cleaner heatmap; uncomment to bring back the per-cell text.
    # if n <= annotate_max:
    #     # small enough to show both the cosine and the concept-coverage fraction behind it
    #     fs = max(4, min(9, 200 // n))
    #     for a in range(n):
    #         for b in range(n):
    #             v = sub[a, b]
    #             if np.isnan(v):
    #                 continue
    #             ax.text(b, a, f"{v:.2f}\n({frac[a, b]:.2f})", ha="center", va="center",
    #                     fontsize=fs, linespacing=1.3,
    #                     color="white" if abs(v) > 0.6 else "black")
    # else:
    #     # too dense for the cosine values too, but the coverage fraction -- what varies
    #     # most sharply near the low-coverage corner -- is still worth showing on its own
    #     fs = max(3, min(7, 220 // n))
    #     for a in range(n):
    #         for b in range(n):
    #             if np.isnan(sub[a, b]):
    #                 continue
    #             ax.text(b, a, f"{frac[a, b]:.2f}", ha="center", va="center", fontsize=fs,
    #                     color="white" if abs(sub[a, b]) > 0.6 else "black")
    # END CELL-ANNOTATIONS
    # relative to the final (last) token: 0 at the last token, negative counting back
    # from it, matching the transform used in plot_all_concepts_mean_cosine's min50 figure
    rel_kidx = kidx - (P - 1)
    step = max(1, n // 32)
    ax.set_xticks(np.arange(n)[::step], labels=rel_kidx[::step])
    ax.set_yticks(np.arange(n)[::step], labels=rel_kidx[::step])
    ax.tick_params(labelsize=12)
    ax.set_xlabel("Position (From Last Token)", fontsize=15)
    ax.set_ylabel("Position (From Last Token)", fontsize=15)
    ax.set_title("Mean Cosine Similarity Between Steering Vectors\n"
                 f"(r_k >= {coverage:.1f}, concept coverage >= 0.5)", fontsize=17)
    # shrink < 1.0 makes the horizontal colorbar shorter than the heatmap axes it sits
    # under, and a large aspect (length / thickness) makes it thinner
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", location="bottom",
                         shrink=0.7, aspect=50, pad=0.1)
    cbar.set_label("Mean Cosine Similarity", fontsize=15)
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()
    fig.savefig(fig_dir / f"{stem}_pairwise.png", dpi=400, bbox_inches="tight",
                pad_inches=0.1)
    plt.close(fig)

    off = csub[~np.eye(n, dtype=bool)]
    print(f"all-concepts pairwise ({n_concepts} concepts, {cov_label}): "
          f"{n}/{P} positions plotted (pos {kidx[0]}-{kidx[-1]}), "
          f"cell concept count min {off.min()} max {off.max()}")
    print(f"wrote {fig_dir}/{stem}_pairwise.png and {out_dir}/{stem}_pairwise.npz")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump_dir", default=DEFAULT_DUMP)
    p.add_argument("--concept_id", type=int, default=0)
    p.add_argument("--prefix", default=None,
                   help="filename prefix for the outputs; defaults to the dump dir name "
                        "with its shared prod_..._diffmean_ prefix stripped")
    p.add_argument("--coverage", type=float, default=0.90,
                   help="drop positions whose r_k is below this from the figures (default "
                        "0.90, i.e. 90%% of the training rows have a real token there); "
                        "the npz still holds every position")
    p.add_argument("--annotate_max", type=int, default=32,
                   help="annotate the pairwise matrix with numbers when P <= this; "
                        "above it the cells are too small to read")
    p.add_argument("--threads", type=int, default=4,
                   help="torch CPU threads; its default (one per core) thrashes on "
                        "shared many-core nodes, and these matrices are small")
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    train_dir = Path(args.dump_dir) / "train"
    pos_w = load_weight(train_dir, POS_MODEL)
    if "positional_col" not in pos_w:
        raise KeyError(f"{POS_MODEL} checkpoint in {train_dir} has no 'positional_col' "
                       f"key (keys: {sorted(pos_w)})")
    row, concept_name = concept_row(train_dir, args.concept_id)

    # [P, h] in the stored column order: index 0 is the deepest slot, index -1 the final
    # token. Left as-is; nothing here reverses the sequence.
    x = pos_w["positional_col"][row].double()
    P, h = x.shape
    # r_k: the share of training rows with a real token at this slot -- not left-padding
    # and past the chat-template prefix. Saved explicitly as "real_frac", and also exactly
    # the norm of the stored vector, since _scale_positions saves r_k * v_k with v_k
    # already unit-norm; the norm is the fallback for a checkpoint without the key.
    norms = x.norm(dim=1).numpy()
    real_frac = pos_w["real_frac"][row].double().numpy() if "real_frac" in pos_w else norms
    reached = np.flatnonzero(real_frac >= args.coverage)
    r1 = int(reached[0]) if len(reached) else None
    cov_label = f"r_k >= {args.coverage:.0%}"

    # a slot no training row ever reached (r_k == 0) is stored as exactly zero and has no
    # direction; mark it rather than letting 0/0 read as orthogonality
    dead = (x.norm(dim=1) <= 0).numpy()
    xn = unit(x)

    mean_w = load_weight(train_dir, MEAN_MODEL)
    last_w = load_weight(train_dir, LAST_MODEL)
    mean_v = unit(mean_w[row].double())
    last_v = unit(last_w[row].double())
    mean_vs_last = float(mean_v @ last_v)

    cos_mean = (xn @ mean_v).numpy()
    cos_last = (xn @ last_v).numpy()
    cos_mat = (xn @ xn.T).numpy()
    for arr in (cos_mean, cos_last):
        arr[dead] = np.nan
    cos_mat[dead, :] = np.nan
    cos_mat[:, dead] = np.nan

    # Figures show only positions with at least --coverage of the training rows behind
    # them; below that a position's vector is a diff-of-means over a handful of long
    # examples padded out with template/padding activations. The npz keeps everything.
    keep = (~dead) & (real_frac >= args.coverage)
    if keep.sum() < 2:
        raise ValueError(f"only {int(keep.sum())} position(s) at coverage >= "
                         f"{args.coverage:.0%}; lower --coverage")
    kidx = np.arange(P)[keep]
    dropped = int((~keep).sum())

    here = Path(__file__).resolve().parent
    out_dir, fig_dir = here / "cosine_sim_results", here / "cosine_sim_figures"
    out_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    dump_name = args.prefix or short_name(args.dump_dir)
    stem = f"{dump_name}_concept{args.concept_id}"
    idx = np.arange(P)
    title_tail = f"concept {args.concept_id}: {concept_name or ''}"[:120]

    np.savez(out_dir / f"{stem}.npz", position=idx, cos_vs_mean=cos_mean,
             cos_vs_last=cos_last, cos_matrix=cos_mat,
             mean_vs_last=np.array(mean_vs_last), real_frac=real_frac)

    # --- per-position cosine against the two single-vector methods ---
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(kidx, cos_mean[keep], marker="o", ms=3, lw=1.4, color="#1f77b4",
            label=f"positional vs {MEAN_MODEL}")
    ax.plot(kidx, cos_last[keep], marker="o", ms=3, lw=1.4, color="#d62728",
            label=f"positional vs {LAST_MODEL}")
    ax.axhline(mean_vs_last, color="k", ls=":", lw=1.2,
               label=f"{MEAN_MODEL} vs {LAST_MODEL} = {mean_vs_last:+.3f}")
    ax.axhline(0, color="grey", lw=0.8, zorder=0)
    ax.plot(kidx, real_frac[keep], color="seagreen", lw=1.0, alpha=0.55, zorder=0,
            label="r_k (coverage)")
    ax.plot([], [], " ", label=f"{dropped} position(s) below {args.coverage:.0%} removed")
    ax.set(xlabel=POS_LABEL, ylabel="cosine similarity", ylim=(-1.02, 1.02),
           xlim=(kidx[0] - 1, kidx[-1] + 1),
           title=f"{POS_MODEL} positions vs single-vector methods ({cov_label})"
                 f"\n{title_tail}")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{stem}_vs_mean_last.png", dpi=400)
    plt.close(fig)

    # --- P x P cosine between every pair of positions ---
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("lightgrey")
    fig, ax = plt.subplots(figsize=(10, 9))
    sub = cos_mat[np.ix_(keep, keep)]
    n = len(kidx)
    im = ax.imshow(sub, vmin=-1, vmax=1, cmap=cmap, origin="upper")
    if n <= args.annotate_max:
        fs = max(4, min(9, 200 // n))
        for a in range(n):
            for b in range(n):
                v = sub[a, b]
                if np.isnan(v):
                    continue
                ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=fs,
                        color="white" if abs(v) > 0.6 else "black")
    step = max(1, n // 32)
    ax.set_xticks(np.arange(n)[::step], labels=kidx[::step])
    ax.set_yticks(np.arange(n)[::step], labels=kidx[::step])
    ax.tick_params(labelsize=7)
    ax.set(xlabel=POS_LABEL, ylabel=POS_LABEL,
           title=f"{POS_MODEL}: cosine similarity between position pairs ({cov_label})"
                 f"\n{title_tail}")
    fig.colorbar(im, ax=ax, shrink=0.8, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{stem}_pairwise.png", dpi=400)
    plt.close(fig)

    # --- summary ---
    off = sub[~np.eye(n, dtype=bool)]
    print(f"concept {args.concept_id} ({concept_name}): P={P} positions, h={h}, "
          f"{int(dead.sum())} empty, {n} plotted at {cov_label} "
          f"(pos {kidx[0]}-{kidx[-1]})")
    print(f"  cos({MEAN_MODEL}, {LAST_MODEL}) = {mean_vs_last:+.4f}")
    for name, c in ((MEAN_MODEL, cos_mean), (LAST_MODEL, cos_last)):
        print(f"  positional vs {name}: final token (pos {P-1}) {c[-1]:+.3f}  "
              f"max {np.nanmax(c):+.3f} at pos {int(np.nanargmax(c))}  "
              f"min {np.nanmin(c):+.3f} at pos {int(np.nanargmin(c))}")
    print(f"  pairwise cosine (off-diagonal, plotted set): mean {off.mean():+.3f}  "
          f"min {off.min():+.3f}  max {off.max():+.3f}  "
          f"negative pairs {int((off < 0).sum()) // 2}/{len(off) // 2}")
    print(f"  coverage r_k: pos 0 {real_frac[0]:.3f}  "
          f"final token (pos {P-1}) {real_frac[-1]:.3f}  "
          + (f"reaches {args.coverage:.0%} at pos {r1} ({P - r1} slots)"
             if r1 is not None else f"never reaches {args.coverage:.0%}"))
    print(f"  max |norm - real_frac| = {np.abs(norms - real_frac).max():.2e}")
    print(f"wrote {fig_dir}/{stem}_vs_mean_last.png and {stem}_pairwise.png")
    print(f"wrote {out_dir}/{stem}.npz")

    # --- aggregate over every concept in the dump ---
    plot_all_concepts_mean_cosine(train_dir, pos_w, mean_w, last_w, args.coverage,
                                   dump_name, out_dir, fig_dir)
    plot_all_concepts_pairwise(train_dir, pos_w, args.coverage, dump_name, out_dir,
                                fig_dir, args.annotate_max)


if __name__ == "__main__":
    main()
