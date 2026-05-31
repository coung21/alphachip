"""
evaluate.py
───────────
Evaluate model đã train: chạy greedy rollout, tính metrics, visualize layout.
"""

from __future__ import annotations
import argparse
import math
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm.auto import tqdm

from placement_util import Block, Net, compute_hpwl, compute_congestion
from environment import ChipFloorplanEnv
from model import ChipPlacementModel

# Optional WandB
try:
    import wandb
except Exception:
    wandb = None


def evaluate_episode(
    env: ChipFloorplanEnv,
    model: ChipPlacementModel,
    device: str = "cpu",
    greedy: bool = True,
) -> dict:
    """
    Chạy 1 episode với model đã train.
    greedy=True: argmax action (no sampling).
    """
    model.eval()
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    actions_taken = []

    with torch.no_grad():
        while not done:
            obs_t = {
                k: torch.from_numpy(v).unsqueeze(0).to(device)
                for k, v in obs.items()
            }
            logits, _ = model(obs_t)
            mask = obs_t["action_mask"]
            logits = logits.masked_fill(~mask, float("-inf"))

            if greedy:
                action = logits.argmax(dim=-1).item()
            else:
                action = torch.distributions.Categorical(logits=logits).sample().item()

            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_reward += reward
            actions_taken.append(action)

    hpwl = compute_hpwl(env.nets, env.blocks)
    cong = compute_congestion(env.nets, env.blocks)

    return {
        "reward": total_reward,
        "hpwl": hpwl,
        "congestion": cong,
        "actions": actions_taken,
        "blocks": env.blocks,
        "nets": env.nets,
    }


def visualize_layout(
    blocks: list,
    nets: list,
    title: str = "Chip Floorplan",
    save_path: str | None = None,
):
    """Visualize chip layout với matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X (normalized)")
    ax.set_ylabel("Y (normalized)")

    # Canvas border
    ax.add_patch(patches.Rectangle(
        (0, 0), 1, 1,
        linewidth=2, edgecolor="black", facecolor="#f8f8f8", zorder=0
    ))

    # Màu cho mỗi block
    cmap = plt.cm.get_cmap("tab20", len(blocks))

    for i, b in enumerate(blocks):
        if b.x is None:
            continue
        color = cmap(i)
        rect = patches.Rectangle(
            (b.x, b.y), b.width, b.height,
            linewidth=1.5,
            edgecolor="black",
            facecolor=color,
            alpha=0.75,
            zorder=2,
        )
        ax.add_patch(rect)

    # Metrics
    hpwl = compute_hpwl(nets, blocks)
    ax.text(
        0.02, 0.98, f"HPWL = {hpwl:.4f}",
        transform=ax.transAxes,
        va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Layout saved → {save_path}")
    else:
        plt.show()

    return fig


def _infer_square_grid(n_actions: int) -> tuple[int, int]:
    side = int(round(math.sqrt(n_actions)))
    if side * side != n_actions:
        raise ValueError(f"Checkpoint action count {n_actions} is not a square grid.")
    return side, side


# ══════════════════════════════════════════════════════════════════════
#  Demo: evaluate với random weights (trước training)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained chip placement checkpoint")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "demo", "debug", "colab_fast", "colab_full"],
                        help="Netlist preset to evaluate. auto infers from checkpoint shape.")
    parser.add_argument("--data_dir", type=str, default="data/ariane")
    parser.add_argument("--ckpt", type=str, default="checkpoints/final.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--grid_cols", type=int, default=None)
    parser.add_argument("--grid_rows", type=int, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    ckpt = None
    ckpt_n_actions = None
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        state_dict = ckpt.get("model_state", ckpt)
        actor_weight = state_dict.get("actor.weight")
        if actor_weight is not None:
            ckpt_n_actions = int(actor_weight.shape[0])

    if args.mode == "auto":
        if ckpt_n_actions == 1024:
            mode = "colab_full"
        elif ckpt_n_actions == 256:
            mode = "colab_fast"
        else:
            mode = "demo"
    else:
        mode = args.mode

    from train import load_netlist
    blocks, nets, default_cols, default_rows = load_netlist(mode, args.data_dir)

    if args.grid_cols is not None and args.grid_rows is not None:
        grid_cols, grid_rows = args.grid_cols, args.grid_rows
    elif ckpt_n_actions is not None:
        grid_cols, grid_rows = _infer_square_grid(ckpt_n_actions)
    else:
        grid_cols, grid_rows = default_cols, default_rows

    env = ChipFloorplanEnv(
        blocks=blocks,
        nets=nets,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
    )

    model = ChipPlacementModel(
        n_blocks=len(blocks),
        n_actions=grid_cols * grid_rows,
        node_dim=ChipFloorplanEnv.NODE_DIM,
        edge_dim=1,
        hidden_dim=128,
        n_layers=3,
    )

    if ckpt is not None:
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        print("No checkpoint found — using random weights.")

    model = model.to(device)

    print(f"Eval mode: {mode} | grid: {grid_cols}x{grid_rows} | device: {device}")

    # Chạy episodes
    results = []
    if wandb is not None:
        try:
            wandb.init(project="chip-placement-eval", reinit=True)
        except Exception:
            wandb = None
    for i in tqdm(range(args.episodes), desc="Evaluating", dynamic_ncols=True):
        r = evaluate_episode(env, model, device=device, greedy=(i == 0))
        results.append(r)
        print(f"Episode {i+1}: reward={r['reward']:+.4f}, HPWL={r['hpwl']:.4f}, "
              f"cong={r['congestion']:.2f}, mode={'greedy' if i==0 else 'sample'}")
        if wandb is not None:
            wandb.log({
                "eval/episode": i + 1,
                "eval/reward": r["reward"],
                "eval/hpwl": r["hpwl"],
                "eval/congestion": r["congestion"],
            })

    # Visualize best layout
    best = min(results, key=lambda x: x["hpwl"])
    print(f"\nBest HPWL: {best['hpwl']:.4f}")
    out_path = "eval_best_layout.png"
    fig = visualize_layout(best["blocks"], best["nets"], title="Best Floorplan Layout", save_path=out_path)
    if wandb is not None:
        try:
            wandb.log({"eval/best_hpwl": best["hpwl"]})
            wandb.log({"eval/best_layout": wandb.Image(out_path)})
            art = wandb.Artifact("eval_layout", type="evaluation")
            art.add_file(out_path)
            run = getattr(wandb, "run", None)
            if run is not None:
                run.log_artifact(art)
        except Exception as e:
            print(f"WandB logging failed: {e}")