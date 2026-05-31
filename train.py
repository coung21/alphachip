"""
train.py
────────
Entry point: train PPO agent trên Ariane RISC-V netlist.

Quick start (Colab):
    python train.py --mode colab_fast          # 133 hard macros, grid 16x16, ~1.5h
    python train.py --mode colab_full          # 133+799 macros, grid 32x32, ~5h
    python train.py --mode debug               # 20 macros, sanity check

Dùng demo netlist (không cần download):
    python train.py --mode demo
"""

from __future__ import annotations
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from placement_util import Block, Net, get_placement_order
from environment import ChipFloorplanEnv
from model import ChipPlacementModel
from ppo_trainer import PPOTrainer, PPOConfig


# ══════════════════════════════════════════════════════════════════════
#  Netlist loaders
# ══════════════════════════════════════════════════════════════════════

def load_netlist(mode: str, data_dir: str):
    """Load netlist theo mode. Returns (blocks, nets, grid_cols, grid_rows)."""

    if mode == "demo":
        # Netlist demo nhỏ — không cần download
        blocks = [
            Block("cpu_core",  0.18, 0.15),
            Block("l2_cache",  0.22, 0.20),
            Block("dma",       0.10, 0.08),
            Block("mem_ctrl0", 0.12, 0.10),
            Block("mem_ctrl1", 0.12, 0.10),
            Block("pcie",      0.08, 0.06),
            Block("usb",       0.06, 0.05),
            Block("gpio",      0.05, 0.04),
        ]
        nets = [
            Net("n0",  [0, 1],    weight=3.0),
            Net("n1",  [0, 2],    weight=2.0),
            Net("n2",  [0, 3],    weight=2.5),
            Net("n3",  [0, 4],    weight=2.5),
            Net("n4",  [1, 3],    weight=1.5),
            Net("n5",  [1, 4],    weight=1.5),
            Net("n6",  [0, 5],    weight=1.0),
            Net("n7",  [0, 6],    weight=0.5),
            Net("n8",  [0, 7],    weight=0.5),
            Net("n9",  [2, 3, 4], weight=1.0),
            Net("n10", [5, 6, 7], weight=0.3),
        ]
        return blocks, nets, 16, 16

    elif mode == "debug":
        from ariane_parser import load_ariane
        blocks, nets = load_ariane(mode="debug", data_dir=data_dir)
        return blocks, nets, 16, 16

    elif mode == "colab_fast":
        # 133 hard SRAM macros, grid nhỏ → train ~1.5h trên Colab T4
        from ariane_parser import load_ariane
        blocks, nets = load_ariane(mode="colab_fast", data_dir=data_dir)
        return blocks, nets, 16, 16

    elif mode == "colab_full":
        # 133 hard + soft macros lớn, grid lớn hơn → ~5h
        from ariane_parser import load_ariane
        blocks, nets = load_ariane(mode="colab_full", data_dir=data_dir)
        return blocks, nets, 32, 32

    else:
        raise ValueError(f"Unknown mode: {mode}")


def make_demo_netlist():
    """Backward-compatible helper for `evaluate.py` to get demo blocks and nets."""
    blocks, nets, cols, rows = load_netlist("demo", data_dir="")
    return blocks, nets


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Chip Floorplanning PPO — Ariane RISC-V")
    parser.add_argument("--mode",         type=str,   default="colab_fast",
                        choices=["demo", "debug", "colab_fast", "colab_full"],
                        help="Netlist preset (default: colab_fast)")
    parser.add_argument("--data_dir",     type=str,   default="data/ariane")
    parser.add_argument("--grid_cols",    type=int,   default=None,
                        help="Override grid cols (default: từ mode preset)")
    parser.add_argument("--grid_rows",    type=int,   default=None,
                        help="Override grid rows")
    parser.add_argument("--n_envs",       type=int,   default=4)
    parser.add_argument("--n_steps",      type=int,   default=None,
                        help="Rollout steps per env (default: n_blocks * 2)")
    parser.add_argument("--total_steps",  type=int,   default=None,
                        help="Total env steps (default: tự chọn theo mode)")
    parser.add_argument("--hidden_dim",   type=int,   default=128)
    parser.add_argument("--n_gcn_layers", type=int,   default=3)
    parser.add_argument("--lr",           type=float, default=4e-4)
    parser.add_argument("--ckpt_dir",     type=str,   default="checkpoints")
    parser.add_argument("--device",       type=str,   default="auto")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # ── Netlist ───────────────────────────────────────────────────────
    blocks, nets, default_cols, default_rows = load_netlist(args.mode, args.data_dir)
    n_blocks = len(blocks)

    grid_cols = args.grid_cols or default_cols
    grid_rows = args.grid_rows or default_rows

    # Total steps mặc định theo mode
    default_total = {
        "demo":       50_000,
        "debug":      20_000,
        "colab_fast": 200_000,
        "colab_full": 500_000,
    }
    total_steps = args.total_steps or default_total[args.mode]

    # n_steps = số bước rollout per env — nên >= 1 episode length
    n_steps = args.n_steps or max(64, n_blocks * 2)

    order = get_placement_order(blocks)
    print(f"\nNetlist : {n_blocks} blocks, {len(nets)} nets")
    print(f"Grid    : {grid_cols}×{grid_rows} = {grid_cols*grid_rows} actions")
    print(f"Training: {total_steps:,} steps, {args.n_envs} envs, {n_steps} steps/rollout")
    print(f"Placement order (first 5): {[blocks[i].name.split('/')[-1] for i in order[:5]]}")

    # ── Environments ──────────────────────────────────────────────────
    def make_env():
        return ChipFloorplanEnv(
            blocks=blocks,
            nets=nets,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            congestion_weight=0.5,
        )

    envs = [make_env() for _ in range(args.n_envs)]

    # ── Model ─────────────────────────────────────────────────────────
    model = ChipPlacementModel(
        n_blocks=n_blocks,
        n_actions=grid_cols * grid_rows,
        node_dim=ChipFloorplanEnv.NODE_DIM,
        edge_dim=1,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_gcn_layers,
        canvas_emb_dim=64,
        metadata_dim=4,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model   : {total_params:,} parameters")

    # ── PPO Config ────────────────────────────────────────────────────
    config = PPOConfig(
        n_envs=args.n_envs,
        n_steps=n_steps,
        total_timesteps=total_steps,
        gamma=1.0,            # sparse terminal reward → no discount
        learning_rate=args.lr,
        batch_size=min(64, args.n_envs * n_steps // 4),
        n_epochs=4,
        entropy_coef=0.01,
        log_interval=10,
        save_interval=50,
        save_path=args.ckpt_dir,
        use_wandb=args.wandb,
        wandb_project="chip-placement",
        device=device,
    )

    # Initialize WandB if requested
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project or "chip-placement",
                config={
                    **vars(args),
                    "total_params": total_params,
                },
                reinit=True,
            )
            print(f"WandB run: {wandb.run.url}")
        except Exception as e:
            print(f"WandB init failed: {e}")

    # ── Train ─────────────────────────────────────────────────────────
    trainer = PPOTrainer(envs, model, config)
    trainer.train()


if __name__ == "__main__":
    main()