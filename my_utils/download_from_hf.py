"""Download the antbaez/axbench-positional-steering HF dataset repo into axbench/results/.

Auth: uses the HF_TOKEN already set in the environment (or a prior `huggingface-cli login`),
picked up automatically by huggingface_hub. Only needed if the repo is private.

Usage:
    uv run my_utils/download_from_hf.py
    uv run my_utils/download_from_hf.py --repo-id antbaez/axbench-positional-steering --path-in-repo results
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "antbaez/axbench-positional-steering"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "axbench" / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--path-in-repo", default="results")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    local_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        allow_patterns=[f"{args.path_in_repo}/*"],
        local_dir=str(results_dir.parent),
    )

    print(f"Downloaded {args.repo_id} ({args.repo_type}) /{args.path_in_repo} -> {local_dir}/{args.path_in_repo}")


if __name__ == "__main__":
    main()
