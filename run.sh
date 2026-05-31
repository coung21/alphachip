#!/usr/bin/env bash
# run.sh — convenience runner for training, evaluation, and sanity checks
# Usage:
#   ./run.sh train [--mode debug|demo|colab_fast|colab_full] [--n_envs N] [--n_steps S] [--total_steps T] [--device cpu|cuda] [--wandb]
#   ./run.sh eval [--ckpt PATH] [--episodes N] [--device cpu|cuda]
#   ./run.sh sanity_train
#   ./run.sh sanity_eval

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtualenv if present
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

COMMAND=${1:-help}
shift || true

# defaults
MODE=debug
N_ENVS=1
N_STEPS=16
TOTAL_STEPS=2000
DEVICE=cpu
WANDB=0
CKPT="checkpoints/final.pt"
EPISODES=5

# parse generic args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --n_envs) N_ENVS="$2"; shift 2;;
    --n_steps) N_STEPS="$2"; shift 2;;
    --total_steps) TOTAL_STEPS="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --wandb) WANDB=1; shift 1;;
    --ckpt) CKPT="$2"; shift 2;;
    --episodes) EPISODES="$2"; shift 2;;
    --help) echo "Options: --mode --n_envs --n_steps --total_steps --device --wandb --ckpt --episodes"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

run_train() {
  CMD=(python train.py --mode "$MODE" --n_envs "$N_ENVS" --n_steps "$N_STEPS" --total_steps "$TOTAL_STEPS" --device "$DEVICE")
  if [ "$WANDB" -eq 1 ]; then
    CMD+=(--wandb)
  fi
  echo "+ ${CMD[*]}"
  "${CMD[@]}"
}

run_eval() {
  CMD=(python evaluate.py)
  echo "+ ${CMD[*]}"
  "${CMD[@]}"
}

sanity_train() {
  echo "Running short sanity training: debug mode, 1 env, 2k steps, CPU"
  python train.py --mode debug --n_envs 1 --n_steps 16 --total_steps 2000 --device cpu
}

sanity_eval() {
  echo "Running evaluation (5 episodes)"
  python evaluate.py
}

case "$COMMAND" in
  train)
    run_train
    ;;
  eval|evaluate)
    run_eval
    ;;
  sanity_train)
    sanity_train
    ;;
  sanity_eval)
    sanity_eval
    ;;
  help|*)
    cat <<EOF
Usage: ./run.sh <command> [options]

Commands:
  train          Run training (see options below)
  eval|evaluate  Run evaluation demo
  sanity_train   Quick local sanity training run (debug, CPU)
  sanity_eval    Quick evaluation run (uses evaluate.py)

Options:
  --mode MODE           Netlist preset: debug|demo|colab_fast|colab_full (default: debug)
  --n_envs N            Number of parallel envs (default: 1)
  --n_steps S           Rollout steps per env (default: 16 for sanity)
  --total_steps T       Total env steps (default: 2000 for sanity)
  --device DEVICE       Device: cpu or cuda (default: cpu)
  --wandb               Enable WandB logging (train only)
  --ckpt PATH           Checkpoint path for evaluation
  --episodes N          Number of eval episodes (evaluate.py uses 5 by default)

Examples:
  ./run.sh train --mode colab_fast --n_envs 4 --n_steps 128 --total_steps 200000 --device cuda --wandb
  ./run.sh eval
  ./run.sh sanity_train
EOF
    ;;
esac
