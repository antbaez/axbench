"""
PCA of trained DiffMean-variant steering vectors (CPU only, no model loading).

    uv run axbench/pca.py [--dump_dir DIR] [--ranks 1 2 3 4 5] [--n_components 50]

MeanTokenDiffMean / LastTokenDiffMean store one vector per concept ([C, h]); each gets
an ordinary PCA across concepts. DiffMeanPositional stores num_positions vectors per
concept ([C, P, h] under "positional_col"); it gets one PCA *per position*, and the
positions are compared by subspace overlap of their top-r components:

    M = U_a^T U_b             # U_* = [h, r] orthonormal top-r PCs
    overlap(a, b) = ||M||_F^2 / r   # = mean cos^2 of the principal angles, in [0, 1]

which is invariant to PC sign, order, and rotation within the subspace -- individual
PCs are not comparable by index across separate PCAs. Chance level for two random
r-dim subspaces of R^h is ~r/h.

Positions are reported as k = tokens back from the end (k=0 is the final token, i.e.
column slot -1 of "positional_col").

Before each PCA, every vector is unit-normalized (a no-op for the stored vectors,
which training already normalizes) and centered across concepts, so PC1 is not just
the shared "any concept" direction.

Outputs under {dump_dir}/pca/: pca_results.json, overlaps.npz, and PNG plots. Large
annotated per-r position x position overlap matrices go to pca_results/ next to this
file, as {dump_name}_overlap_r{r}.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

DEFAULT_DUMP = "axbench/results/prod_9b_l20_concept500_diffmean_pos_steer_data"
SINGLE_MODELS = ("MeanTokenDiffMean", "LastTokenDiffMean")
POS_MODEL = "DiffMeanPositional"
POS_LABEL = "position (sequence order; last = final token)"


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


def prep(x):
    """[C, h] -> float64, unit-normalized rows, centered across concepts."""
    x = x.double()
    norms = x.norm(dim=1, keepdim=True)
    if (norms == 0).any():
        print(f"[warn] {(norms == 0).sum().item()} zero vector(s) left as zero")
    x = x / norms.clamp(min=1e-12)
    return x - x.mean(dim=0, keepdim=True)


def pca(x, n_components):
    """Returns (components [h, n] orthonormal, explained-variance ratio [n])."""
    _, s, vh = torch.linalg.svd(prep(x), full_matrices=False)
    var = s ** 2
    evr = var / var.sum()
    n = min(n_components, vh.shape[0])
    return vh[:n].T, evr[:n]


def overlap(ua, ub):
    """Mean cos^2 of the principal angles between span(ua) and span(ub) (same r)."""
    return (torch.linalg.matrix_norm(ua.T @ ub) ** 2 / ua.shape[1]).item()


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


def single_concept(pos_c, mean_c, last_c, ranks, n_keep):
    """One concept's [P, h] positional stack (rows in k order): cos^2 between
    positions, PCA across positions, and how much of the mean/last-token vector lies in
    the top-r positional PCs (uncentered projection ||U_r^T v||^2 of the unit vector).

    Also reports PC1 and PC2 individually: their explained variance, and their own
    signed cosine similarity with the mean/last-token vectors -- distinct from
    captured_{name}, which is the *aggregate* squared overlap of an r-dim subspace
    and says nothing about either component on its own."""
    x = pos_c.double()
    x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)
    cos2 = (x @ x.T) ** 2
    u, evr = pca(pos_c, n_keep)
    top2 = u[:, :2]  # [h, 2] orthonormal PC1, PC2 -- sign is arbitrary (SVD convention)
    out = {"explained_variance_ratio": evr.tolist(),
           "top2_explained_variance_ratio": evr[:2].tolist(),
           "cos2_matrix": cos2.tolist(),
           "cos2_vs_final_token": cos2[0].tolist(),
           "cos2_adjacent": [cos2[k, k + 1].item() for k in range(len(x) - 1)]}
    for name, v in (("MeanTokenDiffMean", mean_c), ("LastTokenDiffMean", last_c)):
        v = v.double() / v.double().norm().clamp(min=1e-12)
        out[f"cos_vs_{name}"] = (x @ v).tolist()
        out[f"captured_{name}"] = {r: ((u[:, :r].T @ v) ** 2).sum().item() for r in ranks}
        out[f"cos_pc1_vs_{name}"] = (top2[:, 0] @ v).item()
        out[f"cos_pc2_vs_{name}"] = (top2[:, 1] @ v).item()
    return out


def load_positional(train_dir):
    """DiffMeanPositional's [C, P, h] stack in column order (slot -1 = final token)."""
    pos_w = load_weight(train_dir, POS_MODEL)
    if "positional_col" not in pos_w:
        raise KeyError(
            f"{POS_MODEL} checkpoint in {train_dir} has no 'positional_col' key (keys: "
            f"{sorted(pos_w)}); only sequence-ordered checkpoints are supported")
    return pos_w["positional_col"]


def main():
    """Signed cosine similarity between every pair of one concept's positional vectors.

    Signed, not cos^2: a steering vector's sign is meaningful (+v steers toward the
    concept, -v away), so anti-aligned positions should read as such.
    """
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--dump_dir", default=DEFAULT_DUMP)
    p.add_argument("--concept_id", type=int, default=0)
    p.add_argument("--threads", type=int, default=4,
                   help="torch CPU threads; its default (one per core) thrashes on "
                        "shared many-core nodes, and these matrices are small")
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    train_dir = Path(args.dump_dir) / "train"
    pos = load_positional(train_dir)
    row, concept_name = concept_row(train_dir, args.concept_id)
    x = pos[row].double()                            # [P, h], sequence order: P-1 = final token
    x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)
    cos = (x @ x.T).numpy()
    P = cos.shape[0]
    ks = np.arange(P)

    fig_dir = Path(__file__).resolve().parent / "pca_results"
    fig_dir.mkdir(exist_ok=True)
    stem = f"{Path(args.dump_dir).resolve().name}_cosine_concept{args.concept_id}"
    np.save(fig_dir / f"{stem}.npy", cos)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cos, vmin=-1, vmax=1, cmap="RdBu_r", origin="upper")
    fs = max(3, min(8, 160 // P))
    for a in range(P):
        for b in range(P):
            ax.text(b, a, f"{cos[a, b]:.2f}", ha="center", va="center", fontsize=fs,
                    color="white" if abs(cos[a, b]) > 0.6 else "black")
    ax.set_xticks(ks)
    ax.set_yticks(ks)
    ax.tick_params(labelsize=max(5, fs + 1))
    ax.set(xlabel=POS_LABEL, ylabel=POS_LABEL,
           title=f"{POS_MODEL}: cosine similarity between positions\n"
                 f"concept {args.concept_id}: {concept_name or ''}"[:140])
    fig.colorbar(im, ax=ax, shrink=0.8, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{stem}.png", dpi=200)
    plt.close(fig)

    # cosine of each position with the concept's MeanTokenDiffMean / LastTokenDiffMean
    vs = {}
    for m in SINGLE_MODELS:
        v = load_weight(train_dir, m)[row].double()
        vs[m] = (x @ (v / v.norm().clamp(min=1e-12))).numpy()
    np.savez(fig_dir / f"{stem}_vs_mean_last.npz", **vs)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m, c in vs.items():
        ax.plot(ks, c, marker="o", ms=3, label=m)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(ks)
    ax.tick_params(axis="x", labelsize=7)
    ax.set(xlabel=POS_LABEL, ylabel="cosine similarity", ylim=(-1.02, 1.02),
           title=f"{POS_MODEL} positions vs single-vector methods\n"
                 f"concept {args.concept_id}: {concept_name or ''}"[:140])
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{stem}_vs_mean_last.png", dpi=200)
    plt.close(fig)

    off = cos[~np.eye(P, dtype=bool)]
    print(f"concept {args.concept_id} ({concept_name}): {P} positions")
    print(f"  off-diagonal cosine: mean={off.mean():.3f} min={off.min():.3f} "
          f"max={off.max():.3f}  negative pairs={(off < 0).sum() // 2}/{P * (P - 1) // 2}")
    for m, c in vs.items():
        print(f"  cosine vs {m}: final token={c[-1]:.3f}  max={c.max():.3f} at pos "
              f"{c.argmax()}  min={c.min():.3f} at pos {c.argmin()}")
    print(f"wrote {fig_dir}/{stem}.png/.npy and {stem}_vs_mean_last.png/.npz")


def run_pca():
    """The PCA analyses: per-position PCA across concepts, subspace overlaps, and
    single-concept PCA across positions (top-2 directions reported against the
    concept's MeanTokenDiffMean/LastTokenDiffMean vectors)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump_dir", default=DEFAULT_DUMP)
    p.add_argument("--out_dir", default=None, help="default: {dump_dir}/pca")
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--n_components", type=int, default=50,
                   help="components kept for explained-variance curves")
    p.add_argument("--concept_id", type=int, default=0,
                   help="concept for the single-concept (PCA across positions) analysis")
    p.add_argument("--threads", type=int, default=4,
                   help="torch CPU threads; its default (one per core) thrashes on "
                        "shared many-core nodes, and these matrices are small")
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    train_dir = Path(args.dump_dir) / "train"
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.dump_dir) / "pca"
    out_dir.mkdir(parents=True, exist_ok=True)
    ranks = sorted(args.ranks)
    n_keep = max(args.n_components, max(ranks))

    # --- single-vector methods: ordinary PCA across concepts ---
    single = {m: load_weight(train_dir, m) for m in SINGLE_MODELS}
    pos = load_positional(train_dir)                                    # [C, P, h]
    C, P, h = pos.shape
    for m, w in single.items():
        if w.shape != (C, h):
            raise ValueError(f"{m} shape {tuple(w.shape)} != expected {(C, h)}")
    print(f"concepts={C}  positions={P}  hidden={h}  ranks={ranks}")
    if C <= max(ranks):
        raise ValueError(f"need more than {max(ranks)} concepts, got {C}")

    single_pcs = {m: pca(w, n_keep) for m, w in single.items()}

    # --- positional: one PCA per position, k = tokens back from the end ---
    pos_pcs = [pca(pos[:, P - 1 - k], n_keep) for k in range(P)]

    results = {"concepts": C, "positions": P, "hidden": h, "ranks": ranks,
               "chance_overlap": {r: r / h for r in ranks},
               "explained_variance_ratio": {
                   m: evr.tolist() for m, (_, evr) in single_pcs.items()},
               "positional_explained_variance_ratio": [evr.tolist() for _, evr in pos_pcs],
               "mean_vs_last_overlap": {}, "overlap_vs_final_token": {},
               "adjacent_overlap": {}, "overlap_vs_single": {m: {} for m in SINGLE_MODELS},
               "last_token_sanity_cosine": None}
    pos_matrices = {}

    # Sanity check: slot -1 and LastTokenDiffMean are built from the same token of the
    # same left-padded batches, so should agree per concept (cosine ~ 1 up to bf16).
    cos = torch.nn.functional.cosine_similarity(
        pos[:, -1].double(), single["LastTokenDiffMean"].double(), dim=1)
    results["last_token_sanity_cosine"] = {
        "mean": cos.mean().item(), "min": cos.min().item()}

    for r in ranks:
        U = [u[:, :r] for u, _ in pos_pcs]
        mat = np.array([[overlap(U[a], U[b]) for b in range(P)] for a in range(P)])
        pos_matrices[r] = mat
        results["overlap_vs_final_token"][r] = mat[0].tolist()
        results["adjacent_overlap"][r] = [mat[k, k + 1] for k in range(P - 1)]
        for m, (u, _) in single_pcs.items():
            results["overlap_vs_single"][m][r] = [overlap(U[k], u[:, :r]) for k in range(P)]
        results["mean_vs_last_overlap"][r] = overlap(
            single_pcs["MeanTokenDiffMean"][0][:, :r], single_pcs["LastTokenDiffMean"][0][:, :r])

    # --- single concept: PCA across its own positions ---
    row, concept_name = concept_row(train_dir, args.concept_id)
    sc = single_concept(pos[row].flip(0), single["MeanTokenDiffMean"][row],
                        single["LastTokenDiffMean"][row], ranks, n_keep)
    sc.update({"concept_id": args.concept_id, "concept": concept_name})
    results["single_concept"] = sc

    (out_dir / "pca_results.json").write_text(json.dumps(results, indent=1))
    np.savez(out_dir / "overlaps.npz", **{f"r{r}": m for r, m in pos_matrices.items()})

    # --- plots ---
    ks = np.arange(P)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, (_, evr) in single_pcs.items():
        ax.plot(np.arange(1, len(evr) + 1), np.cumsum(evr.numpy()), lw=2, label=m)
    for k in sorted({0, P // 4, P // 2, P - 1}):
        evr = pos_pcs[k][1].numpy()
        ax.plot(np.arange(1, len(evr) + 1), np.cumsum(evr), ls="--", label=f"{POS_MODEL} k={k}")
    ax.set(xlabel="components", ylabel="cumulative explained variance",
           title="PCA across concepts")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "explained_variance.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(ranks), figsize=(3.6 * len(ranks), 3.6), squeeze=False)
    for ax, r in zip(axes[0], ranks):
        im = ax.imshow(pos_matrices[r], vmin=0, vmax=1, cmap="viridis", origin="upper")
        ax.set(title=f"r={r}", xlabel="k (tokens back)")
    axes[0][0].set_ylabel("k (tokens back)")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="subspace overlap")
    fig.suptitle(f"{POS_MODEL}: top-r subspace overlap between positions")
    fig.savefig(out_dir / "overlap_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # one large, annotated P x P figure per r, next to this script
    fig_dir = Path(__file__).resolve().parent / "pca_results"
    fig_dir.mkdir(exist_ok=True)
    dump_name = Path(args.dump_dir).resolve().name
    for r in ranks:
        mat = pos_matrices[r]
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", origin="upper")
        fs = max(3, min(8, 160 // P))
        for a in range(P):
            for b in range(P):
                ax.text(b, a, f"{mat[a, b]:.2f}", ha="center", va="center", fontsize=fs,
                        color="black" if mat[a, b] > 0.6 else "white")
        ax.set_xticks(ks)
        ax.set_yticks(ks)
        ax.tick_params(labelsize=max(5, fs + 1))
        ax.set(xlabel="k (tokens back from end)", ylabel="k (tokens back from end)",
               title=f"{POS_MODEL}: top-{r} subspace overlap between positions\n"
                     f"{dump_name}  ({C} concepts, chance ≈ {r / h:.4f})")
        fig.colorbar(im, ax=ax, shrink=0.8, label="subspace overlap (mean cos² of principal angles)")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{dump_name}_overlap_r{r}.png", dpi=200)
        plt.close(fig)

    panels = [("vs final token (k=0)", lambda r: results["overlap_vs_final_token"][r], ks),
              ("adjacent (k vs k+1)", lambda r: results["adjacent_overlap"][r], ks[:-1])] + \
             [(f"vs {m}", lambda r, m=m: results["overlap_vs_single"][m][r], ks)
              for m in SINGLE_MODELS]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), sharey=True)
    for ax, (title, get, x) in zip(axes, panels):
        for r in ranks:
            ax.plot(x, get(r), label=f"r={r}")
        ax.set(title=title, xlabel="k (tokens back)", ylim=(0, 1.02))
    axes[0].set_ylabel("subspace overlap")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "overlap_by_position.png", dpi=150)
    plt.close(fig)

    cid = args.concept_id
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.6]})
    im = axes[0].imshow(np.array(sc["cos2_matrix"]), vmin=0, vmax=1, cmap="viridis")
    axes[0].set(title="cos² between positions", xlabel="k (tokens back)",
                ylabel="k (tokens back)")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    axes[1].plot(ks, sc["cos2_vs_final_token"], label="cos² vs k=0")
    axes[1].plot(ks[:-1], sc["cos2_adjacent"], label="cos² vs k+1")
    for m in SINGLE_MODELS:
        axes[1].plot(ks, np.array(sc[f"cos_vs_{m}"]) ** 2, label=f"cos² vs {m}")
    axes[1].set(xlabel="k (tokens back)", ylim=(0, 1.02))
    axes[1].legend(fontsize=8)
    fig.suptitle(f"concept {cid}: {concept_name or ''}"[:110])
    fig.tight_layout()
    fig.savefig(out_dir / f"concept{cid}_position_similarity.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    evr = np.array(sc["explained_variance_ratio"])
    ax.bar(np.arange(1, len(evr) + 1), evr, alpha=0.5, label="per component")
    ax.plot(np.arange(1, len(evr) + 1), np.cumsum(evr), color="k", label="cumulative")
    ax.set(xlabel="component", ylabel="explained variance ratio", ylim=(0, 1.02),
           title=f"concept {cid}: PCA across its {P} positions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"concept{cid}_explained_variance.png", dpi=150)
    plt.close(fig)

    # --- summary ---
    print(f"\nLastToken sanity cosine (slot -1 vs LastTokenDiffMean): "
          f"mean={cos.mean():.4f} min={cos.min():.4f}")
    print(f"\n{'r':>2} {'chance':>7} {'Mean~Last':>9} {'k0~Last':>8} {'k0~Mean':>8} "
          f"{'adj(mean)':>9} {'k0~k' + str(P - 1):>8}")
    for r in ranks:
        print(f"{r:>2} {r / h:>7.4f} {results['mean_vs_last_overlap'][r]:>9.3f} "
              f"{results['overlap_vs_single']['LastTokenDiffMean'][r][0]:>8.3f} "
              f"{results['overlap_vs_single']['MeanTokenDiffMean'][r][0]:>8.3f} "
              f"{np.mean(results['adjacent_overlap'][r]):>9.3f} "
              f"{pos_matrices[r][0, -1]:>8.3f}")
    evr = sc["explained_variance_ratio"]
    print(f"\nconcept {cid} ({concept_name}): PCA across {P} positions")
    print(f"  EVR PC1..5: {' '.join(f'{v:.3f}' for v in evr[:5])}   "
          f"mean cos² between positions: "
          f"{(np.sum(sc['cos2_matrix']) - P) / (P * (P - 1)):.3f}")
    print(f"  top-2 PCA directions: PC1 explains {sc['top2_explained_variance_ratio'][0]:.1%}, "
          f"PC2 explains {sc['top2_explained_variance_ratio'][1]:.1%}")
    print(f"    cos(PC1, MeanTokenDiffMean)={sc['cos_pc1_vs_MeanTokenDiffMean']:+.3f}   "
          f"cos(PC1, LastTokenDiffMean)={sc['cos_pc1_vs_LastTokenDiffMean']:+.3f}")
    print(f"    cos(PC2, MeanTokenDiffMean)={sc['cos_pc2_vs_MeanTokenDiffMean']:+.3f}   "
          f"cos(PC2, LastTokenDiffMean)={sc['cos_pc2_vs_LastTokenDiffMean']:+.3f}")
    print(f"  {'r':>2} {'Mean captured':>14} {'Last captured':>14}")
    for r in ranks:
        print(f"  {r:>2} {sc['captured_MeanTokenDiffMean'][r]:>14.3f} "
              f"{sc['captured_LastTokenDiffMean'][r]:>14.3f}")
    print(f"\nwrote {out_dir}/pca_results.json, overlaps.npz, and 5 plots")
    print(f"wrote {len(ranks)} overlap matrices to {fig_dir}/{dump_name}_overlap_r*.png")


if __name__ == "__main__":
    run_pca()
