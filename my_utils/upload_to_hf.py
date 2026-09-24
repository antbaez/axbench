"""Upload axbench/results/ to the antbaez/axbench-positional-steering HF dataset repo.

Auth: uses the HF_TOKEN already set in the environment (or a prior `huggingface-cli login`),
picked up automatically by huggingface_hub.

Usage:
    uv run my_utils/upload_to_hf.py
    uv run my_utils/upload_to_hf.py --repo-id antbaez/axbench-positional-steering --path-in-repo results
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi
from tqdm import tqdm

REPO_ID = "antbaez/axbench-positional-steering"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "axbench" / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--path-in-repo", default="results")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="Upload results/")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"No such directory: {results_dir}")

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )

    files = sorted(f for f in results_dir.rglob("*") if f.is_file())
    for f in tqdm(files, desc="Uploading files", unit="file"):
        api.upload_file(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            path_or_fileobj=str(f),
            path_in_repo=f"{args.path_in_repo}/{f.relative_to(results_dir)}",
            commit_message=args.commit_message,
        )

    print(f"Uploaded {len(files)} files: {results_dir} -> {args.repo_id} ({args.repo_type}) at /{args.path_in_repo}")


if __name__ == "__main__":
    main()
