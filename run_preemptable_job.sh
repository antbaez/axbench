#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err

# Generic single-GPU sbatch worker template. Submit as:
#   sbatch run_preemptable_job.sh <command> [args...]
# Runs the given command under the resources/directives above. Called by
# run_preemptable.sh, which owns CFG/DUMP/NPROC/etc and does the actual
# `sbatch run_preemptable_job.sh ...` submission per pipeline stage -- this
# file itself is just the resource template and shouldn't need per-run edits.

set -e

# Hardcoded, not derived from ${BASH_SOURCE[0]} -- matches the other
# run_preemptable_*.sh scripts' handling of nested/job-spool invocation.
if [ -f ~/axbench/.env ]; then
  set -a
  source ~/axbench/.env
  set +a
fi

cd ~/axbench

exec "$@"
