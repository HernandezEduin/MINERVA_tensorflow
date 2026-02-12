# Run all .yaml configs in a folder in parallel, up to N workers.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config_folder> <max_parallel_jobs>"
  echo "Example: $0 configs/nlq_signals 4"
  exit 1
fi

CFG_DIR="$1"
MAX_JOBS="$2"

if [[ ! -d "$CFG_DIR" ]]; then
  echo "Error: config_folder '$CFG_DIR' is not a directory (or does not exist)."
  exit 1
fi

if ! [[ "$MAX_JOBS" =~ ^[0-9]+$ ]] || (( MAX_JOBS <= 0 )); then
  echo "Error: max_parallel_jobs must be a positive integer."
  exit 1
fi

running=0
echo "Config folder: $CFG_DIR"
echo "Running with MAX_JOBS=$MAX_JOBS"

shopt -s nullglob

# Collect YAMLs into an array
cfgs=("$CFG_DIR"/*.yaml)

# Error if none found
if (( ${#cfgs[@]} == 0 )); then
  echo "Error: no .yaml files found under '$CFG_DIR'"
  exit 1
fi

# Sort deterministically
IFS=$'\n' cfgs=($(printf "%s\n" "${cfgs[@]}" | sort))
unset IFS

for cfg in "${cfgs[@]}"; do
  echo "[LAUNCH] $cfg"
  bash run_nlq.sh "$cfg" &
  ((++running))

  if (( running >= MAX_JOBS )); then
    wait -n
    ((--running))
  fi
done

wait
echo "All experiments completed."
