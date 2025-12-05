#!/usr/bin/env bash
# Full-batch RLHF runs (J-only reward) for all supervised checkpoints.
# Usage: bash scripts/run_full_rlhf_J.sh

set -uo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

models=(
  "bart-large"
  "pure-t5-base"
  "bart-paraphrase"
  "flan-t5-base"
  "mvp"
  "prophetnet-large-uncased"
  "bart-base"
  "t5-small"
  "blenderbot"
  "distilbart"
)

total=${#models[@]}
start_time=$(date +%s)
log_dir="logs/rl_full_run_jonly"
mkdir -p "$log_dir"
output_root="RL/full_run_jonly"
mkdir -p "$output_root"

format_time() {
  local seconds=$1
  if (( seconds < 0 )); then
    seconds=0
  fi
  date -u -r "$seconds" +"%H:%M:%S"
}

for idx in "${!models[@]}"; do
  model_key="${models[$idx]}"
  input_dir="TRAIN/${model_key}"
  output_dir="${output_root}/${model_key}"

  if [[ ! -d "${input_dir}/final_model" ]]; then
    echo "[WARN] ${model_key}: missing ${input_dir}/final_model, skipping."
    continue
  fi

  hf_name=$(python - <<'PY'
import json, os, sys
input_dir = sys.argv[1]
fallback = sys.argv[2]
config_path = os.path.join(input_dir, "final_model", "config.json")
try:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(cfg.get("_name_or_path", fallback))
except Exception:
    print(fallback)
PY
"$input_dir" "$model_key")

  timestamp=$(date +%Y%m%d_%H%M%S)
  log_file="${log_dir}/${model_key}_${timestamp}.log"
  printf '\n[%d/%d] Starting %s (J-only) -> %s\n' "$((idx + 1))" "$total" "$model_key" "$output_dir"

  cmd=(
    python RL/train.py
      --model_name "$hf_name"
      --data_path TRAIN/paradetox.tsv
      --input_dir "$input_dir"
      --output_dir "$output_dir"
      --ppo_epochs 10
      --ppo_steps_per_epoch 50
      --ppo_batch_size 16
      --ppo_lr 6e-6
      --rl_train_size 9000
      --rl_eval_size 1500
      --eval_sample_size 300
      --reward_alpha 0.4
      --reward_beta 0.3
      --metric_reward_weight 0.8
      --bleu_reward_weight 0.0
  )

  if ( set -o pipefail; "${cmd[@]}" 2>&1 | tee "$log_file" ); then
    echo "[OK] ${model_key} finished. Logs: ${log_file}"
  else
    status=$?
    echo "[FAIL] ${model_key} exited with status ${status}. See ${log_file}. Continuing..."
  fi

  elapsed=$(( $(date +%s) - start_time ))
  avg=$(( elapsed / (idx + 1) ))
  remaining=$(( avg * (total - idx - 1) ))
  printf 'Elapsed: %s | ETA: %s\n' "$(format_time "$elapsed")" "$(format_time "$remaining")"
done

echo "\nAll J-only runs dispatched. Check ${log_dir} for per-model logs and ${output_root} for checkpoints."
