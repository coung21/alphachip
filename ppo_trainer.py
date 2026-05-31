"""
PPO training loop
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

# Optional WandB
try:
    import wandb
except Exception:
    wandb = None

@dataclass
class PPOConfig:
    n_envs: int = 4

    n_steps: int = 128
    total_timesteps: int = 500_000

    # PPO hyperparameters
    gamma: float = 1.0 # Sparse reward only at episode end
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Optimization
    learning_rate: float = 4e-4
    batch_size: int = 64
    n_epochs: int = 4
    anneal_lr: bool = True

    # Logging
    log_interval: int = 10
    save_interval: int = 100
    save_path: str = "./checkpoints"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # WandB
    use_wandb: bool = False
    wandb_project: Optional[str] = None



class RolloutBuffer:
    """
    Buffer để lưu trữ rollout từ nhiều environment trong quá trình thu thập dữ liệu cho PPO update.
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_space_sample: Dict[str, Any],
        n_actions: int,
        device: str
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device
        self.n_actions = n_actions
        self.full = False

        self.ptr = 0
        
        T, E = n_steps, n_envs

        # Pre-allocate obs buffers theo shape của mỗi key
        self.obs_bufs: Dict[str, np.ndarray] = {}
        for k, v in obs_space_sample.items():
            shape = (T, E) + v.shape
            self.obs_bufs[k] = np.zeros(shape, dtype=np.float32)

        self.actions = np.zeros((T, E), dtype=np.int64)
        self.rewards = np.zeros((T, E), dtype=np.float32)
        self.dones = np.zeros((T, E), dtype=bool)
        self.values = np.zeros((T, E), dtype=np.float32)
        self.logprobs = np.zeros((T, E), dtype=np.float32)

    def store(
        self,
        step: int,
        obs: List[Dict[str, np.ndarray]],
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        logprobs: np.ndarray
    ):
        for k in self.obs_bufs:
            self.obs_bufs[k][step] = np.stack([o[k] for o in obs])
        self.actions[step] = actions
        self.rewards[step] = rewards
        self.dones[step] = dones
        self.values[step] = values
        self.logprobs[step] = logprobs

    def compute_returns_and_advantages(
            self,
            last_values: np.ndarray, # (n_envs,)
            last_dones: np.ndarray,  # (n_envs,)
            gamma: float,
            gae_lambda: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GAE advantage estimation"""
        T, E = self.n_steps, self.n_envs
        advantages = np.zeros((T, E), dtype=np.float32)
        lastgaelam = np.zeros(E, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_none_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_none_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_values * next_none_terminal - self.values[t]
            lastgaelam = delta + gamma * gae_lambda * next_none_terminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + self.values
        return returns, advantages
    
    def get_minibatches(
            self,
            returns: np.ndarray,
            advantages: np.ndarray,
            batch_size: int,
    ):
        """Yield minibatches of data for PPO update"""
        T, E = self.n_steps, self.n_envs
        total_size = T * E
        indices = np.random.permutation(total_size)

        # Flatten obs buffers
        flat_obs = {
            k: v.reshape((total_size, *v.shape[2:])) for k, v in self.obs_bufs.items()
        }
        flat_actions = self.actions.reshape(total_size)
        flat_logprobs = self.logprobs.reshape(total_size)
        flat_returns = returns.reshape(total_size)
        flat_advantages = advantages.reshape(total_size)
        flat_values = self.values.reshape(total_size)

        for start in range(0, total_size, batch_size):
            idx = indices[start: start + batch_size]
            yield {
                "obs": {
                    k: (torch.from_numpy(v[idx]).to(self.device).bool()
                        if k == "action_mask" else torch.from_numpy(v[idx]).to(self.device))
                    for k, v in flat_obs.items()
                },
                "actions": torch.from_numpy(flat_actions[idx]).to(self.device),
                "old_log_probs": torch.from_numpy(flat_logprobs[idx]).to(self.device),
                "returns": torch.from_numpy(flat_returns[idx]).to(self.device),
                "advantages": torch.from_numpy(flat_advantages[idx]).to(self.device),
                "old_values": torch.from_numpy(flat_values[idx]).to(self.device),
            }
                

class PPOTrainer:

    def __init__(self, envs, model, config: PPOConfig):
        self.envs = envs
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.device = config.device
        self.use_wandb = bool(config.use_wandb and wandb is not None)


        # Rollout buffer
        obs_sample = envs[0].reset()[0] # obs, info
        self.buffer = RolloutBuffer(
            n_steps=config.n_steps,
            n_envs=config.n_envs,
            obs_space_sample=obs_sample,
            n_actions=envs[0].n_actions,
            device=config.device
        )

        os.makedirs(config.save_path, exist_ok=True)
        
        # Metrics
        self.ep_rewards = deque(maxlen=100)
        self.ep_hpwls = deque(maxlen=100)
        self.global_step = 0
        self.update_count = 0

    def train(self):
        cfg = self.config
        n_updates = cfg.total_timesteps // (cfg.n_steps * cfg.n_envs)

        #reset all envs
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)

        done = np.zeros(cfg.n_envs, dtype=bool)

        print(f"Training on {self.device}")
        print(f"Total updates: {n_updates}, steps/update: {cfg.n_steps * cfg.n_envs}")

        start_time = time.time()
        progress = tqdm(range(1, n_updates + 1), desc="Training", dynamic_ncols=True)
        for update in progress:
            # Anneal learning rate
            if cfg.anneal_lr:
                frac = 1.0 - (update - 1) / n_updates
                self.optimizer.param_groups[0]['lr'] = cfg.learning_rate * frac

            # Collect rollout
            for step in range(cfg.n_steps):
                self.global_step += cfg.n_envs

                # Forward pass
                with torch.no_grad():
                    obs_tensor = self._obs_to_tensor(obs_list)
                    actions, logprobs, _,values = self.model.get_action_and_value(obs_tensor)
                    actions_np = actions.cpu().numpy()
                    logprobs_np = logprobs.cpu().numpy()
                    values_np = values.cpu().numpy()

                # Step envs
                next_obs_list, rewards, terminateds, truncateds = [], [], [], []
                for i, (env, a) in enumerate(zip(self.envs, actions_np)):
                    obs, r, term, trunc, info = env.step(int(a))
                    rewards.append(r)
                    terminateds.append(term)
                    truncateds.append(trunc)
 
                    if term or trunc:
                        self.ep_rewards.append(info["episode_reward"])
                        self.ep_hpwls.append(info["hpwl"])
                        obs, _ = env.reset()
 
                    next_obs_list.append(obs)
 
                rewards_np    = np.array(rewards, dtype=np.float32)
                dones_np      = np.array(
                    [t or tr for t, tr in zip(terminateds, truncateds)],
                    dtype=np.float32
                )
 
                self.buffer.store(
                    step, obs_list, actions_np, rewards_np,
                    dones_np, values_np, logprobs_np,
                )
                obs_list = next_obs_list
 
            # ── Bootstrap value ───────────────────────────────────────
            with torch.no_grad():
                obs_tensor = self._obs_to_tensor(obs_list)
                _, _, _, last_values = self.model.get_action_and_value(obs_tensor)
                last_values_np = last_values.cpu().numpy()
 
            returns, advantages = self.buffer.compute_returns_and_advantages(
                last_values_np, dones_np, cfg.gamma, cfg.gae_lambda,
            )
 
            # ── PPO update ────────────────────────────────────────────
            stats = self._ppo_update(returns, advantages)
            self.update_count += 1
 
            # ── Logging ───────────────────────────────────────────────
            if update % cfg.log_interval == 0:
                elapsed = time.time() - start_time
                sps = self.global_step / elapsed
                mean_r  = np.mean(self.ep_rewards) if self.ep_rewards else 0.0
                mean_wl = np.mean(self.ep_hpwls)   if self.ep_hpwls   else 0.0
                print(
                    f"Update {update:4d}/{n_updates} | "
                    f"Step {self.global_step:7d} | "
                    f"SPS {sps:.0f} | "
                    f"Reward {mean_r:+.4f} | "
                    f"HPWL {mean_wl:.4f} | "
                    f"Loss {stats['loss']:.4f} | "
                    f"Entropy {stats['entropy']:.4f} | "
                    f"LR {self.optimizer.param_groups[0]['lr']:.2e}"
                )
                progress.set_postfix(
                    step=self.global_step,
                    sps=f"{sps:.0f}",
                    reward=f"{mean_r:+.4f}",
                    hpwl=f"{mean_wl:.4f}",
                    loss=f"{stats['loss']:.4f}",
                )
                if self.use_wandb:
                    wandb.log({
                        "update": update,
                        "step": self.global_step,
                        "sps": sps,
                        "reward": mean_r,
                        "hpwl": mean_wl,
                        "loss": stats["loss"],
                        "entropy": stats["entropy"],
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }, step=self.global_step)
 
            if update % cfg.save_interval == 0:
                path = os.path.join(cfg.save_path, f"ckpt_{update:05d}.pt")
                self.save(path)
 
        self.save(os.path.join(cfg.save_path, "final.pt"))
        progress.close()
        print("Training complete.")
 
    # ── PPO update step ───────────────────────────────────────────────
 
    def _ppo_update(self, returns: np.ndarray, advantages: np.ndarray) -> Dict:
        cfg = self.config
        all_losses, all_pg, all_vf, all_ent = [], [], [], []
        all_clipfrac = []
 
        for _ in range(cfg.n_epochs):
            for batch in self.buffer.get_minibatches(returns, advantages, cfg.batch_size):
                obs       = batch["obs"]
                actions   = batch["actions"]
                old_lp    = batch["old_log_probs"]
                ret       = batch["returns"]
                adv       = batch["advantages"]
                old_vals  = batch["old_values"]
 
                # Normalize advantages (per-minibatch)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
 
                _, new_lp, entropy, new_vals = (
                    self.model.get_action_and_value(obs, actions)
                )
 
                # Policy loss (clipped)
                ratio = torch.exp(new_lp - old_lp)
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()
 
                # Value loss (clipped)
                v_clipped = old_vals + (new_vals - old_vals).clamp(
                    -cfg.value_clip_eps, cfg.value_clip_eps
                )
                vf_loss1 = (new_vals - ret).pow(2)
                vf_loss2 = (v_clipped - ret).pow(2)
                vf_loss  = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
 
                # Entropy bonus
                ent_loss = entropy.mean()
 
                # Total loss
                loss = (
                    pg_loss
                    + cfg.value_coef * vf_loss
                    - cfg.entropy_coef * ent_loss
                )
 
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
 
                # Stats
                clipfrac = ((ratio - 1).abs() > cfg.clip_eps).float().mean()
                all_losses.append(loss.item())
                all_pg.append(pg_loss.item())
                all_vf.append(vf_loss.item())
                all_ent.append(ent_loss.item())
                all_clipfrac.append(clipfrac.item())
 
        return {
            "loss":      np.mean(all_losses),
            "pg_loss":   np.mean(all_pg),
            "vf_loss":   np.mean(all_vf),
            "entropy":   np.mean(all_ent),
            "clipfrac":  np.mean(all_clipfrac),
        }
 
    # ── Helpers ───────────────────────────────────────────────────────
 
    def _obs_to_tensor(
        self, obs_list: List[Dict[str, np.ndarray]]
    ) -> Dict[str, torch.Tensor]:
        """Stack list of obs dicts → dict of batched tensors."""
        keys = obs_list[0].keys()
        return {
            k: torch.from_numpy(
                np.stack([o[k] for o in obs_list])
            ).to(self.device)
            for k in keys
        }
 
    def save(self, path: str):
        torch.save({
            "model_state":  self.model.state_dict(),
            "optim_state":  self.optimizer.state_dict(),
            "global_step":  self.global_step,
            "update_count": self.update_count,
        }, path)
        print(f"  Saved checkpoint → {path}")
        # Upload to WandB as artifact if enabled
        if self.use_wandb:
            try:
                artifact = wandb.Artifact("checkpoint", type="model")
                artifact.add_file(path)
                wandb.run.log_artifact(artifact)
                print(f"  Uploaded checkpoint to WandB → {path}")
            except Exception as e:
                print(f"  WandB upload failed: {e}")
 
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.global_step  = ckpt["global_step"]
        self.update_count = ckpt["update_count"]
        print(f"  Loaded checkpoint ← {path}")