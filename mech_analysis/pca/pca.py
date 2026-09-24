"""
PCA across one concept's DiffMeanPositionalWeighted per-position steering vectors,
projected to 3D (CPU only, no model loading).

    uv run --no-sync mech_analysis/pca/pca.py [--dump_dir DIR] [--concept_id N]

DiffMeanPositionalWeighted stores num_positions vectors per concept under
"positional_col" ([C, P, h]). This script takes one concept's [P, h] stack, runs a PCA
across its P positions, and shows the positions in PC space: a static PC1-PC2 scatter and
an interactive PC1-PC2-PC3 plot you can rotate. Consecutive positions are joined by a
line, so a direction that drifts smoothly along the sequence traces a smooth path and one
that jumps does not. In the html that line carries arrowheads at each segment midpoint
giving the sequence direction, since the colour gradient alone is ambiguous once the
camera has been rotated (--no_arrows turns them off).

Positions stay in the stored sequence order: index 0 is the deepest slot (num_positions
tokens back from the end) and the last index is the final token. Nothing is flipped.

Positions whose coverage r_k is below --coverage (default 67%) are dropped before the PCA:
below that, a position's vector is a diff-of-means over a handful of long examples padded
out with template/padding activations. r_k is the norm of the stored vector, since the
checkpoint holds r_k * v_k with v_k unit-norm.

The stored vectors are r_k * v_k and are NOT renormalized, so by default the PCA sees the
coverage weighting. --unit_norm rescales every position to unit length first, which
answers the different question of how the *directions* alone are arranged.

Centering is across the kept positions, so PC1..PC3 describe how positions deviate from
this concept's own mean direction. --no_center keeps that centroid, in which case PC1 is
dominated by the shared direction itself.

MeanTokenDiffMean and LastTokenDiffMean are overlaid as out-of-sample points: each is put
through the same scaling as the positional stack, has the same centroid subtracted, and is
then projected onto the PCs. Because the PCs span at most n-1 of h=3584 dimensions and
were fitted *without* these vectors, a projected marker always looks closer to the cloud
than the vector really is -- so each one's legend entry carries the fraction of its
centered length the plotted axes actually capture. Read the marker together with that
number, never alone. Note also that r_k at the final position is 1 by construction, so the
last positional vector *is* LastTokenDiffMean: its marker landing on the final-token point
is a wiring check, not a finding.

Caveat worth keeping in mind: n points in h=3584 dimensions is an exact, unregularized fit
spanning at most n-1 dimensions. The explained-variance ratios describe these points; they
do not estimate a population.

Outputs, next to this file:
    pca_figures_3d/{dump}_concept{N}_cov{C}_{scale}_pca3d.html   always
    pca_figures_2d/{dump}_concept{N}_cov{C}_{scale}_pca2d.png    only with --plot_2d
    pca_results/{dump}_concept{N}_cov{C}_{scale}_pca.npz         always

where {C} is --coverage as a percentage, so runs at different thresholds sit side by side
instead of overwriting one another.
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
POS_MODEL = "DiffMeanPositionalWeighted"      # checkpoint name, must match the class
METHOD_LABEL = "Weighted Positional Token Steering"   # how the method is named in titles
OVERLAY_MODELS = ("MeanTokenDiffMean", "LastTokenDiffMean")
OVERLAY_STYLE = {"MeanTokenDiffMean": ("D", "#1f77b4"), "LastTokenDiffMean": ("^", "#d62728")}
# plot labels: what each direction is, rather than the class that produced it
OVERLAY_LABEL = {"MeanTokenDiffMean": "Mean Token Activation",
                 "LastTokenDiffMean": "Final Token Activation"}
POS_LABEL = "position (sequence order; last = final token)"


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


def write_html_3d(path, scores, idx, real_frac_kept, evr, overlays, title, plotlyjs,
                  arrows=True, arrow_scale=0.03, line_width=3):
    """Interactive PC1-PC2-PC3 view. Returns False (with a warning) if plotly is absent,
    so the rest of the script still runs."""
    try:
        import plotly.graph_objects as go
        from plotly.colors import sample_colorscale
    except ImportError:
        print("[warn] plotly not installed; skipping the interactive 3D figure. "
              "Install it with: uv pip install -c <constraints> 'plotly<7'")
        return False

    hover = [f"Position {p}<br>r_k {r:.3f}<br>PC1 {a:.3f}<br>PC2 {b:.3f}<br>PC3 {c:.3f}"
             for p, r, (a, b, c) in zip(idx, real_frac_kept, scores)]
    fig = go.Figure()
    # The path is drawn as one short trace per segment, before the markers so the points
    # draw on top. Each segment takes the colour of the position it leaves, sampled from
    # the same Viridis scale plotly applies to the markers (sample_colorscale reproduces
    # that mapping exactly, so line and point colours cannot drift apart). A single
    # lines+markers trace would allow only one colour for the whole path, and would also
    # put the line at the same depth as the points, where they largely hide it.
    #
    # The arrowhead for a segment is emitted in the same iteration and given that same
    # colour. A Cone trace colours its cones from their vector norms through a colorscale
    # and has no per-cone colour field, and every arrow here is deliberately the same
    # length -- so one trace could only ever produce one colour. One single-cone trace per
    # segment, each with a constant colorscale, is what lets the head match its line.
    #
    # sizemode="raw" draws the cone at its actual vector length. The other two modes
    # multiply the norm by an internal factor plotly derives from the *minimum* spacing
    # between successive points; several consecutive positions here nearly coincide, so
    # that factor collapses to ~0 and the arrowheads come out invisible.
    span_idx = max(float(idx[-1] - idx[0]), 1.0)
    seg_colors = sample_colorscale(
        "Viridis", [(float(i) - idx[0]) / span_idx for i in idx], colortype="rgb")
    head_len = float(np.ptp(scores, axis=0).max()) * arrow_scale
    for i in range(len(scores) - 1):
        colour = seg_colors[i]
        fig.add_trace(go.Scatter3d(
            x=scores[i:i + 2, 0], y=scores[i:i + 2, 1], z=scores[i:i + 2, 2],
            mode="lines", line=dict(color=colour, width=line_width),
            showlegend=False, hoverinfo="skip"))
        step = scores[i + 1] - scores[i]
        length = float(np.linalg.norm(step))
        if not arrows or length <= 0:
            continue
        d = step / length * head_len
        mid = (scores[i] + scores[i + 1]) / 2.0
        fig.add_trace(go.Cone(
            x=[mid[0]], y=[mid[1]], z=[mid[2]], u=[d[0]], v=[d[1]], w=[d[2]],
            sizemode="raw", sizeref=1, anchor="center", showscale=False,
            colorscale=[[0, colour], [1, colour]],
            # no hover: the arrowheads sit on top of the points, and their tooltip was
            # firing instead of the position's
            hoverinfo="skip",
            showlegend=False))
    fig.add_trace(go.Scatter3d(
        x=scores[:, 0], y=scores[:, 1], z=scores[:, 2], mode="markers",
        # the three axes on screen only hold part of the spread; say how much up front
        name=f"Token Position (PC1-3 = {evr[:3].sum():.1%} of variance)", legendrank=1,
        marker=dict(size=6, symbol="circle", color=idx, colorscale="Viridis",
                    showscale=True,
                    # pinned right of the plotting area and kept narrow; plotly's default
                    # places the colorbar and the legend in the same corner, where they
                    # overlap (the legend ends up behind the gradient)
                    colorbar=dict(title=dict(text="Position", side="right"),
                                  x=1.02, xanchor="left", y=0.5, yanchor="middle",
                                  len=0.72, thickness=14),
                    line=dict(width=0.5, color="white")),
        text=hover, hoverinfo="text"))
    for rank, (nm, sc, cap3, _) in enumerate(overlays, start=4):
        fig.add_trace(go.Scatter3d(
            x=[sc[0]], y=[sc[1]], z=[sc[2]], mode="markers",
            name=f"{OVERLAY_LABEL.get(nm, nm)} ({cap3:.0%} in PC1-3)", legendrank=rank,
            marker=dict(size=6, symbol="x", line=dict(width=0),
                        color=OVERLAY_STYLE.get(nm, ("", "black"))[1]),
            hovertext=[f"{OVERLAY_LABEL.get(nm, nm)}<br>PC1-3 Captures {cap3:.1%} "
                       f"of Its Centered Length"],
            hoverinfo="text"))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=f"PC1 ({evr[0]:.1%} of Variance)",
                   yaxis_title=f"PC2 ({evr[1]:.1%} of Variance)",
                   zaxis_title=f"PC3 ({evr[2]:.1%} of Variance)", aspectmode="data"),
        # legend moved to the free top-left corner of the scene, away from the
        # colorbar, with a translucent backing so the path stays visible underneath
        legend=dict(itemsizing="constant", x=0.01, y=0.99, xanchor="left",
                    yanchor="top", bgcolor="rgba(255,255,255,0.78)",
                    bordercolor="rgba(0,0,0,0.25)", borderwidth=1,
                    font=dict(size=14)),
        margin=dict(l=0, r=90, t=60, b=0))
    fig.write_html(str(path), include_plotlyjs=plotlyjs)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump_dir", default=DEFAULT_DUMP)
    p.add_argument("--concept_id", type=int, default=0)
    p.add_argument("--prefix", default=None,
                   help="filename prefix for the outputs; defaults to the dump dir name "
                        "with its shared prod_..._diffmean_ prefix stripped")
    p.add_argument("--coverage", type=float, default=0.95,
                   help="drop positions whose r_k is below this before the PCA (default "
                        "0.95, i.e. almost every training row has a real token there)")
    p.add_argument("--as_stored", action="store_true",
                   help="keep the stored r_k-weighted vector lengths instead of "
                        "unit-normalizing each position first; unit-norm is the default, "
                        "so only the directions are compared")
    p.add_argument("--plot_2d", action="store_true",
                   help="also write the static PC1-PC2 png; only the interactive 3D html "
                        "is written by default")
    p.add_argument("--no_center", action="store_true",
                   help="skip centering across positions; PC1 then captures the shared "
                        "direction rather than deviation from it")
    p.add_argument("--no_arrows", action="store_true",
                   help="omit the arrowheads marking sequence direction in the html")
    p.add_argument("--arrow_scale", type=float, default=0.03,
                   help="arrowhead length as a fraction of the largest PC axis range "
                        "(default 0.03); raise it if the heads are hard to see, lower it "
                        "if they crowd the points")
    p.add_argument("--n_report", type=int, default=10,
                   help="how many components to print the variance table for (default 10)")
    p.add_argument("--line_width", type=float, default=3,
                   help="width of the path joining consecutive positions in the html "
                        "(default 3); raise it if the line is hard to see")
    p.add_argument("--no_overlay", action="store_true",
                   help="omit the MeanTokenDiffMean / LastTokenDiffMean markers")
    p.add_argument("--offline_html", action="store_true",
                   help="inline plotly.js in the html (~3MB, works with no internet) "
                        "instead of loading it from the CDN")
    p.add_argument("--threads", type=int, default=4,
                   help="torch CPU threads; its default (one per core) thrashes on "
                        "shared many-core nodes, and these matrices are small")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    unit_norm = not args.as_stored

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

    # Keep only positions with at least --coverage of the training rows behind them. A
    # slot below that is a diff-of-means over a handful of long examples padded out with
    # template/padding activations, so it is mostly noise; r_k == 0 slots are the extreme
    # case, stored as exactly zero, and would otherwise put the origin in the fit.
    keep = (norms > 0) & (real_frac >= args.coverage)
    idx = np.arange(P)[keep]
    x = x[torch.from_numpy(keep)]
    if x.shape[0] < 4:
        raise ValueError(f"only {x.shape[0]} position(s) for concept {args.concept_id} at "
                         f"coverage >= {args.coverage:.0%}; need 4+ for three PCs, so "
                         f"lower --coverage")
    print(f"[warn] dropping {int((~keep).sum())} position(s) below {args.coverage:.0%} "
          f"coverage: {np.arange(P)[~keep].tolist()}" if not keep.all() else
          f"keeping all {P} positions")

    if unit_norm:
        x = x / x.norm(dim=1, keepdim=True)
    centroid = torch.zeros(h, dtype=x.dtype) if args.no_center else x.mean(dim=0)
    centered = x - centroid

    _, s, vh = torch.linalg.svd(centered, full_matrices=False)
    evr = (s ** 2 / (s ** 2).sum()).numpy()
    scores = (centered @ vh[:3].T).numpy()          # [n_keep, 3]

    # Out-of-sample overlays: same scaling, same centroid, then projected onto the PCs.
    # capture2/capture3 say how much of each vector the plotted axes actually hold -- the
    # rest lies in the discarded dimensions, so a marker alone is not evidence of anything.
    overlays = []
    if not args.no_overlay:
        for name in OVERLAY_MODELS:
            v = load_weight(train_dir, name)[row].double()
            if unit_norm:
                v = v / v.norm().clamp(min=1e-12)
            d = v - centroid
            proj = (d @ vh[:3].T).numpy()
            denom = float(d @ d)
            cap3 = float((proj ** 2).sum() / denom) if denom > 0 else 0.0
            cap2 = float((proj[:2] ** 2).sum() / denom) if denom > 0 else 0.0
            overlays.append((name, proj, cap3, cap2))

    here = Path(__file__).resolve().parent
    # 2D and 3D live in separate folders; only the 3D html is written unless --plot_2d
    out_dir = here / "pca_results"
    fig2d_dir, fig3d_dir = here / "pca_figures_2d", here / "pca_figures_3d"
    out_dir.mkdir(exist_ok=True)
    fig3d_dir.mkdir(exist_ok=True)
    if args.plot_2d:
        fig2d_dir.mkdir(exist_ok=True)
    dump_name = args.prefix or short_name(args.dump_dir)
    # coverage is in the filename: two runs at different thresholds are different
    # analyses, and without it the second silently overwrites the first
    suffix = (f"_cov{int(round(args.coverage * 100))}"
              + ("_unitnorm" if unit_norm else "_asstored")
              + ("_uncentered" if args.no_center else ""))
    stem = f"{dump_name}_concept{args.concept_id}{suffix}"
    name = (concept_name or "")
    title = (f"{METHOD_LABEL}: PCA Across {len(idx)} Positions ({cov_label})<br>"
             f"Concept {args.concept_id}: {name[:1].upper() + name[1:]}")[:200]

    np.savez(out_dir / f"{stem}_pca.npz", position=idx, scores=scores,
             components=vh[:3].numpy(), explained_variance_ratio=evr,
             real_frac=real_frac[keep], coverage=np.array(args.coverage),
             overlay_names=np.array([o[0] for o in overlays]),
             overlay_scores=np.array([o[1] for o in overlays]).reshape(-1, 3),
             overlay_capture_pc123=np.array([o[2] for o in overlays]),
             overlay_capture_pc12=np.array([o[3] for o in overlays]))

    if args.plot_2d:
        # --- static PC1-PC2 ---
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        ax.plot(scores[:, 0], scores[:, 1], lw=0.7, color="grey", zorder=0, alpha=0.8)
        sc = ax.scatter(scores[:, 0], scores[:, 1], c=idx, cmap="viridis", s=45,
                        edgecolor="white", linewidth=0.4, zorder=2)
        for i, marker, label in ((0, "s", f"pos {idx[0]} (deepest kept, {cov_label})"),
                                 (len(idx) - 1, "*", f"pos {idx[-1]} (final token)")):
            ax.scatter(*scores[i, :2], marker=marker, s=160, facecolor="none",
                       edgecolor="black", linewidth=1.2, zorder=3, label=label)
        for name, proj, _, cap2 in overlays:
            marker, color = OVERLAY_STYLE.get(name, ("X", "black"))
            ax.scatter(proj[0], proj[1], marker=marker, s=95, color=color, zorder=4,
                       edgecolor="black", linewidth=0.6,
                       label=f"{name} ({cap2:.0%} in PC1-2)")
        ax.axhline(0, color="lightgrey", lw=0.8, zorder=0)
        ax.axvline(0, color="lightgrey", lw=0.8, zorder=0)
        ax.set(xlabel=f"PC1 ({evr[0]:.1%} of Variance)",
               ylabel=f"PC2 ({evr[1]:.1%} of Variance)",
               title=title.replace("<br>", "\n"))
        ax.legend(fontsize=8, loc="best")
        fig.colorbar(sc, ax=ax, shrink=0.85, label=POS_LABEL)
        fig.tight_layout()
        fig.savefig(fig2d_dir / f"{stem}_pca2d.png", dpi=200)
        plt.close(fig)

    # --- interactive PC1-PC2-PC3 ---
    html = fig3d_dir / f"{stem}_pca3d.html"
    wrote_html = write_html_3d(html, scores, idx, real_frac[keep], evr, overlays, title,
                               "inline" if args.offline_html else "cdn",
                               arrows=not args.no_arrows, arrow_scale=args.arrow_scale,
                               line_width=args.line_width)

    print(f"concept {args.concept_id} ({concept_name}): {len(idx)}/{P} positions, h={h}")
    print(f"  {'unit-normalized' if unit_norm else 'as stored (r_k weighted)'}, "
          f"{'uncentered' if args.no_center else 'centered across positions'}")
    print(f"  r_k reaches {args.coverage:.0%} at pos {r1}" if r1 is not None
          else f"  r_k never reaches {args.coverage:.0%}")
    n_rep = min(args.n_report, len(evr))
    print(f"  explained variance ({len(evr)} components exist; "
          f"{len(idx)} points span at most {len(idx) - 1})")
    print(f"    {'PC':>3}  {'individual':>10}  {'cumulative':>10}")
    cum = np.cumsum(evr)
    for i in range(n_rep):
        print(f"    {i + 1:>3}  {evr[i]:>9.1%}  {cum[i]:>10.1%}")
    for name, _, cap3, cap2 in overlays:
        print(f"  {name:<18} captured: PC1-2 {cap2:6.1%}   PC1-3 {cap3:6.1%}")
    if args.plot_2d:
        print(f"wrote {fig2d_dir}/{stem}_pca2d.png")
    if wrote_html:
        print(f"wrote {fig3d_dir}/{stem}_pca3d.html")
    print(f"wrote {out_dir}/{stem}_pca.npz")


if __name__ == "__main__":
    main()
