"""
Chip floorplanning Gymnasium environment referenced theo circuit_training.

Design desisions từ paper:
    1. Sparse reward: chỉ có reward khi hoàn thành một episode, không có intermediate reward.
    2. Reward = -(norm_wirelength + λ_c * norm_congestion)
    3. Placement order = sắp xếp các block theo thứ tự giảm dần về kích thước (largest first).
    4. Action space = grid cell rời rạc (x, y)
    5. Observation = node_features + canvas + metadata + action_mask 
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Dict
import copy

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from placement_util import (
    Block,
    Net,
    PlacementGrid,
    get_placement_order,
    compute_hpwl,
    compute_congestion,
)

class ChipFloorplanEnv(gym.Env):
    """
    Chip floorplanning RL environment.

    Observation space:
        - node_features: (n_blocks, NODE_DIM) per block features cho GNN
        - edge_index: (2, max_edges) adjacency từ netlist 
        - edge_features: (max_edges, EDGE_DIM) per edge net weight
        - canvas: (1, rows, cols) occupancy map
        - current_node: (NODE_DIM,) feature của block hiện tại
        - metadata: (4,): step_ratio, cur_w, cur_h, n_placed/n

    Action space:
        - Discrete (rows * cols): chọn grid cell index (row-major: a = row * cols + col)

    Reward:
        - 0 cho mọi step trừ khi episode kết thúc
        - Terminal = -(norm_wirelength + λ_c * norm_congestion) nếu hoàn thành episode
"""

    metadata = {"render_modes": ["human", "rgb_array"]}
 
    # Số chiều feature
    NODE_DIM = 8   # [w, h, area, cx, cy, is_placed, is_port, is_current]
    EDGE_DIM = 1   # [net_weight (normalized)]
 
    def __init__(
        self,
        blocks: List[Block],
        nets: List[Net],
        grid_cols: int = 32,
        grid_rows: int = 32,
        congestion_weight: float = 0.5,
        max_edges: int = 2048,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
 
        # Lưu bản gốc để reset
        self._orig_blocks = copy.deepcopy(blocks)
        self._orig_nets = copy.deepcopy(nets)
 
        self.blocks: List[Block] = copy.deepcopy(blocks)
        self.nets: List[Net] = nets
        self.n_blocks = len(blocks)
 
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.n_actions = grid_cols * grid_rows
        self.congestion_weight = congestion_weight
        self.max_edges = max_edges
        self.render_mode = render_mode
 
        # Thứ tự đặt (tính 1 lần, không đổi)
        self.placement_order: List[int] = get_placement_order(self.blocks)
        self.n_movable = len(self.placement_order)
 
        # Precompute edge index + features (static per episode)
        self._static_edge_index, self._static_edge_feats = (
            self._build_edges()
        )
 
        # ── Spaces ───────────────────────────────────────────────────
        self.action_space = spaces.Discrete(self.n_actions)
 
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(
                0.0, 1.0, (self.n_blocks, self.NODE_DIM), np.float32
            ),
            "edge_index": spaces.Box(
                0, self.n_blocks, (2, max_edges), np.int64
            ),
            "edge_features": spaces.Box(
                0.0, 1.0, (max_edges, self.EDGE_DIM), np.float32
            ),
            "canvas": spaces.Box(
                0.0, 1.0, (1, grid_rows, grid_cols), np.float32
            ),
            "current_node": spaces.Box(
                0.0, 1.0, (self.NODE_DIM,), np.float32
            ),
            "metadata": spaces.Box(0.0, 1.0, (4,), np.float32),
            # Action mask: True = valid placement
            "action_mask": spaces.Box(
                0, 1, (self.n_actions,), np.bool_
            ),
        })
 
        # Runtime state (khởi tạo bởi reset)
        self.pg: Optional[PlacementGrid] = None
        self.step_idx: int = 0
        self._done: bool = False
 
    # ══════════════════════════════════════════════════════════════════
    #  Gymnasium API
    # ══════════════════════════════════════════════════════════════════
 
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
 
        # Restore block positions
        self.blocks = copy.deepcopy(self._orig_blocks)
 
        # Fresh grid
        self.pg = PlacementGrid(self.grid_cols, self.grid_rows)
 
        # Place fixed blocks
        for i, b in enumerate(self.blocks):
            if (b.is_fixed or b.is_port) and b.x is not None:
                col, row = self.pg.xy_to_cell(b.x, b.y)
                self.pg.place(i + 1, col, row, b)
 
        self.step_idx = 0
        self._done = False
        self._episode_reward = 0.0
 
        return self._obs(), {}
 
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        assert not self._done, "Call reset() before stepping."
 
        block_idx = self.placement_order[self.step_idx]
        block = self.blocks[block_idx]
 
        # Decode action → grid cell → normalized coords
        col, row = self.pg.action_to_cell(action)
        col, row, _, _ = self.pg.block_cells(col, row, block)  # clamped
        x, y = self.pg.cell_to_xy(col, row)
        # Offset về bottom-left corner
        x -= block.width / 2
        y -= block.height / 2
        x = float(np.clip(x, 0.0, 1.0 - block.width))
        y = float(np.clip(y, 0.0, 1.0 - block.height))
 
        block.x, block.y = x, y
        self.pg.place(block_idx + 1, col, row, block)
 
        self.step_idx += 1
 
        # ── Reward (sparse) ──────────────────────────────────────────
        terminated = self.step_idx == self.n_movable
        reward = self._terminal_reward() if terminated else 0.0
        if terminated:
            self._done = True
 
        self._episode_reward += reward
        return self._obs(), reward, terminated, False, self._info()
 
    # ══════════════════════════════════════════════════════════════════
    #  Reward
    # ══════════════════════════════════════════════════════════════════
 
    def _terminal_reward(self) -> float:
        """
        R = -(norm_wirelength + λ_c × norm_congestion)
 
        Normalize:
          wirelength  / (n_nets × canvas_diagonal)  → scale-free
          congestion via tanh                        → bounded [0, 1)
        """
        hpwl = compute_hpwl(self.nets, self.blocks)
        canvas_diag = np.sqrt(2.0)  # diagonal của [0,1]² canvas
        n_nets = max(len(self.nets), 1)
        norm_wl = hpwl / (n_nets * canvas_diag)
 
        cong = compute_congestion(
            self.nets, self.blocks,
            self.grid_rows // 4,
            self.grid_cols // 4,
        )
        norm_cong = float(np.tanh(cong / 5.0))  # squash → [0,1)
 
        return -(norm_wl + self.congestion_weight * norm_cong)
 
    # ══════════════════════════════════════════════════════════════════
    #  Observation
    # ══════════════════════════════════════════════════════════════════
 
    def _obs(self) -> Dict[str, np.ndarray]:
        cur_idx = (
            self.placement_order[self.step_idx]
            if self.step_idx < self.n_movable
            else -1
        )
 
        # ── Node features ─────────────────────────────────────────────
        # [w, h, area, cx, cy, is_placed, is_port, is_current]
        nf = np.zeros((self.n_blocks, self.NODE_DIM), dtype=np.float32)
        for i, b in enumerate(self.blocks):
            nf[i, 0] = b.width
            nf[i, 1] = b.height
            nf[i, 2] = b.area
            nf[i, 3] = b.cx if b.cx is not None else -1.0
            nf[i, 4] = b.cy if b.cy is not None else -1.0
            nf[i, 5] = 1.0 if b.is_placed else 0.0
            nf[i, 6] = 1.0 if b.is_port else 0.0
            nf[i, 7] = 1.0 if i == cur_idx else 0.0
 
        # ── Current node feature ──────────────────────────────────────
        cur_node = nf[cur_idx] if cur_idx >= 0 else np.zeros(self.NODE_DIM, np.float32)
 
        # ── Canvas ────────────────────────────────────────────────────
        canvas = self.pg.oppupancy_map()[np.newaxis, :, :]  # (1, H, W)
 
        # ── Metadata ──────────────────────────────────────────────────
        step_ratio = self.step_idx / max(self.n_movable, 1)
        cur_w = self.blocks[cur_idx].width if cur_idx >= 0 else 0.0
        cur_h = self.blocks[cur_idx].height if cur_idx >= 0 else 0.0
        n_placed_ratio = sum(b.is_placed for b in self.blocks) / self.n_blocks
        metadata = np.array(
            [step_ratio, cur_w, cur_h, n_placed_ratio], dtype=np.float32
        )
 
        # ── Action mask ───────────────────────────────────────────────
        if cur_idx >= 0:
            mask = self.pg.get_action_mask(self.blocks[cur_idx])
        else:
            mask = np.zeros(self.n_actions, dtype=np.bool_)
 
        return {
            "node_features":  nf,
            "edge_index":     self._static_edge_index,
            "edge_features":  self._static_edge_feats,
            "canvas":         canvas,
            "current_node":   cur_node,
            "metadata":       metadata,
            "action_mask":    mask,
        }
 
    def _info(self) -> Dict[str, Any]:
        return {
            "step": self.step_idx,
            "n_movable": self.n_movable,
            "hpwl": compute_hpwl(self.nets, self.blocks),
            "episode_reward": self._episode_reward,
        }
 
    # ══════════════════════════════════════════════════════════════════
    #  Edge Index (static — computed once at init)
    # ══════════════════════════════════════════════════════════════════
 
    def _build_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build edge_index (2, E) và edge_features (E, 1) từ nets.
        Mỗi net (a, b, c, ...) → undirected edges: a↔b, a↔c, b↔c, ...
 
        Normalize net weight về [0, 1] theo max weight.
        """
        src, dst, weights = [], [], []
        for net in self.nets:
            idxs = net.block_indices
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    src.extend([idxs[i], idxs[j]])
                    dst.extend([idxs[j], idxs[i]])
                    weights.extend([net.weight, net.weight])
 
        max_w = max(weights) if weights else 1.0
 
        # Pad hoặc truncate đến max_edges
        def pad(arr, fill=0):
            arr = np.array(arr)
            if len(arr) >= self.max_edges:
                return arr[:self.max_edges]
            return np.concatenate([arr, np.full(self.max_edges - len(arr), fill)])
 
        ei = np.stack([
            pad(src, 0).astype(np.int64),
            pad(dst, 0).astype(np.int64),
        ])  # (2, max_edges)
 
        ef_raw = pad(weights, 0.0).astype(np.float32) / max_w
        ef = ef_raw[:, np.newaxis]  # (max_edges, 1)
 
        return ei, ef
 
    # ══════════════════════════════════════════════════════════════════
    #  Render
    # ══════════════════════════════════════════════════════════════════
 
    def render(self):
        if self.render_mode != "human":
            return
        syms = "·" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        print("┌" + "─" * self.grid_cols + "┐")
        for r in range(self.grid_rows - 1, -1, -1):  # y=0 ở bottom
            row_str = "│"
            for c in range(self.grid_cols):
                cell = self.pg.grid[r, c]
                row_str += syms[min(cell, len(syms) - 1)]
            print(row_str + "│")
        print("└" + "─" * self.grid_cols + "┘")
 
    def close(self):
        pass