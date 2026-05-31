# RL Placement Project

This repository contains code for training and evaluating an RL-based placement model.

Quick overview
- Training: `train.py` / `ppo_trainer.py`
- Evaluation: `evaluate.py`
- Model: `model.py`
- Environment: `environment.py`
- Utilities: `placement_util.py`, `ariane_parser.py`
- Checkpoints: `checkpoints/` (contains saved `.pt` files)

Prerequisites

1. Python 3.8+ recommended
2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

Running

- Train a model (example):

```bash
python train.py
```

- Evaluate a checkpoint:

```bash
python evaluate.py --checkpoint checkpoints/final.pt
```

Run script (`run.sh`) — Detailed usage
------------------------------------

This repository includes a convenience runner `run.sh` for training, evaluation, and quick sanity checks. The script wraps `train.py` and `evaluate.py` and accepts the common options described below.

Basic steps

1. Activate the virtual environment:

```bash
source .venv/bin/activate
```

2. (Optional) Make the script executable:

```bash
chmod +x run.sh
```

3. Examples:

```bash
# Full training (GPU, WandB enabled)
./run.sh train --mode colab_fast --n_envs 4 --n_steps 128 --total_steps 200000 --device cuda --wandb

# Evaluate (uses default checkpoint unless --ckpt provided)
./run.sh eval --ckpt checkpoints/final.pt --episodes 5 --device cpu

# Quick local sanity runs
./run.sh sanity_train
./run.sh sanity_eval
```

Options (supported by `run.sh`)

- `--mode MODE` : preset netlist/config (debug|demo|colab_fast|colab_full). Default: `debug`.
- `--n_envs N` : number of parallel environments. Default: `1`.
- `--n_steps S` : rollout steps per env. Default: `16`.
- `--total_steps T` : total environment steps for training. Default: `2000`.
- `--device DEVICE` : `cpu` or `cuda`. Default: `cpu`.
- `--wandb` : enable WandB logging for training.
- `--ckpt PATH` : checkpoint path for evaluation (e.g., `checkpoints/final.pt`).
- `--episodes N` : number of evaluation episodes (default: `5`).

Tips

- To log to Weights & Biases, run `wandb login` first or set `WANDB_API_KEY` in the environment, then add `--wandb` to the `train` command.
- Use `--device cuda` only if your machine has a compatible GPU and CUDA drivers.
- Checkpoints are saved to the `checkpoints/` directory; logs and run artifacts are placed under `wandb/` by default when WandB is enabled.

Hướng dẫn ngắn bằng tiếng Việt

1. Bật môi trường ảo:

```bash
source .venv/bin/activate
```

2. Cấp quyền thực thi (nếu muốn):

```bash
chmod +x run.sh
```

3. Chạy ví dụ nhanh:

```bash
# Chạy huấn luyện nhanh (debug)
./run.sh train --mode debug --n_envs 1 --n_steps 16 --total_steps 2000 --device cpu

# Huấn luyện thực tế với WandB và GPU
./run.sh train --mode colab_fast --n_envs 4 --n_steps 128 --total_steps 200000 --device cuda --wandb

# Đánh giá checkpoint
./run.sh eval --ckpt checkpoints/final.pt --episodes 5 --device cpu
```

Nếu bạn muốn mình thêm ví dụ cấu hình cụ thể (ví dụ `colab_fast` vs `colab_full`) hoặc tự động upload logs, mình sẽ bổ sung.


WandB

The project uses Weights & Biases for logging. To export charts locally, use the provided helper:

```bash
python export_wandb_charts.py --entity YOUR_ENTITY --project YOUR_PROJECT --outdir wandb_charts
```

This will create PNG charts per run in `wandb_charts/` (see `README_WANDB_EXPORT.md` for details).

Files of interest

- `run.sh` — quick runner script
- `checkpoints/` — saved training checkpoints
- `data/` — dataset inputs (e.g., `data/ariane/netlist.pb.txt`)

Notes

- If you use WandB, make sure to run `wandb login` or set the `WANDB_API_KEY` environment variable.
- Adjust training hyperparameters in the training scripts or configuration sections as needed.

Contact

If you want additional helpers (CSV export of histories, upload of PNGs to S3/Drive, or CI integration), open an issue or ask me to add them.
