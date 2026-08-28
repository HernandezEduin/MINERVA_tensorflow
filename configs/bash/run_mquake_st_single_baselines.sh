#!/usr/bin/env bash
set -euo pipefail

# Run MQuAKE-ST single-answer path-fidelity baselines from the repo root.
# Usage:
#   bash configs/bash/run_mquake_st_single_answer_baselines.sh
#   bash configs/bash/run_mquake_st_single_answer_baselines.sh 0 42 100

seeds=("$@")
if [[ $# -eq 0 ]]; then
  seeds=(0 42 100)
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

output_dir="output/mquake_st/baselines"
mkdir -p "$output_dir"

common_args=(
  --data-input-dir ./datasets/nlq/mquake_st/
  --question-path ./datasets/nlq/mquake_st/mquake_sa_qa_nhop.csv
  --cached-qa-metadata-path ./.cache/itl/mquake_sa_qa_nhop.json
  --test-only
  --use-self-loops
  --use-full-graph
  --num-rollout-steps 4
)

for seed in "${seeds[@]}"; do
  conda run -n minerva_tf2 python -m code.baselines.random_walk_stats \
    "${common_args[@]}" \
    --num-walks 100 \
    --seed "$seed" \
    --output "$output_dir/random_walk_stats_mquake_sa_seed${seed}.json"

  conda run -n minerva_tf2 python -m code.baselines.shortcut_oracle_stats \
    "${common_args[@]}" \
    --seed "$seed" \
    --output "$output_dir/shortcut_oracle_stats_mquake_sa_seed${seed}.json"
done
