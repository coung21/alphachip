"""
GNN-based Actor-Critic theo kiến trúc AlphaChip

References:
  - Nature 2021: A graph placement methodology for fast chip design
"""

from __future__ import annotations
from typing import Any, List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

class CanvasEncoder(nn.Module):
    """CNN encoder cho canvas occupancy map (B, 1, H, W) -> (B, out_dim)"""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            # (B, 1, H, W)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),# H / 2
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # H / 4
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # H / 8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # (B, 64, 4, 4)
        )

        self.proj = nn.Linear(64 * 4 * 4, out_dim)

    def forward(self, canvas: torch.Tensor) -> torch.Tensor:
        x = self.cnn(canvas).view(canvas.size(0), -1) # (B, 64*4*4)
        return F.relu(self.proj(x)) # (B, out_dim)
    

class NetlistGNN(nn.Module):
    """GCN encoder cho netlist graph
        GCNConv:
            H' = σ( D̃^{-1/2} · Ã · D̃^{-1/2} · H · W )
            Ã = A + I  (self-loop)
            D̃_ii = Σ_j Ã_ij  (degree matrix)

        
        Args:
            node_in_dim: input feature dimension per node (= NODE_DIM)
            edge_in_dim: input feature dimension per edge (= NET_DIM)
            hidden_dim: hidden dimension sau mỗi GCN layer
            n_layers: số lượng GCN layer chồng lên nhau
            dropout: dropout rate giữa các GCN layer
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Node input projection
        self.node_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Edge feature fusion (GCN chỉ dung edge weights, nên cần map edge features về scalar weight)
        self.edge_fusion = nn.Sequential(
            nn.Linear(edge_in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1), # output edge weight scalar
            nn.Softplus() # ensure non-negative weights
        )

        # GCN layers
        self.convs = nn.ModuleList([
            GCNConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                improved=False,
                add_self_loops=True,
                normalize=True # D̃^{-1/2} · Ã · D̃^{-1/2
            ) for _ in range(n_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def build_pyg_batch(
            self,
            node_feats: torch.Tensor, # (B, N, node_in_dim)
            edge_index: torch.Tensor, # (B, 2, max_edges)
            edge_feats: torch.Tensor, # (B, max_edges, edge_in_dim)
            n_valid_edges: torch.Tensor # (B,)

    ) -> Batch:
        """
        Convert batched obs tensors thành PyG Batch object:
            - node_feats: (B, N, NODE_DIM
            - edge_index: (B, 2, max_edges) - padded
            - edge_feats: (B, max_edges, EDGE_DIM) - padded
            - n_valid_edges: (B,) - number of valid edges per graph (để unpad edge_index và edge_feats)
        """
        B = node_feats.size(0)
        max_e = edge_index.size(2)

        data_list = []
        for b in range(B):
            ve = n_valid_edges[b] if n_valid_edges is not None else max_e
            ei = edge_index[b, :, :ve].long() # (2, ve)
            ea = edge_feats[b, :ve, :] # (ve, edge_in_dim)
            x = node_feats[b] # (N, node_in_dim)
            data_list.append(Data(x=x, edge_index=ei, edge_attr=ea))
        return Batch.from_data_list(data_list)

    def forward(
        self,
        node_feats: torch.Tensor, # (B, N, node_in_dim)
        edge_index: torch.Tensor, # (B, 2, max_edges)
        edge_feats: torch.Tensor, # (B, max_edges, edge_in_dim)
        n_valid_edges: torch.Tensor # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            - graph_emb: (B, hidden_dim) - global mean pooling over node embeddings
            - node_emb: (B, N, hidden_dim)
        """
        B, N, _ = node_feats.shape

        # Build PyG Batch
        pyg_batch = self.build_pyg_batch(
            node_feats, edge_index, edge_feats, n_valid_edges
        )

        # Node feature projection
        h = self.node_proj(pyg_batch.x) # (B*N, hidden_dim)

        # Edge weight fusion
        edge_weight = self.edge_fusion(pyg_batch.edge_attr).squeeze(-1) # (B*ve,)
        ei = pyg_batch.edge_index # (2, B*ve)

        # GCN layers
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, ei, edge_weight=edge_weight) # (B*N, hidden_dim)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)

            h = h + h_new # Residual connection

        # Graph-level embedding bằng global mean pooling
        graph_emb = global_mean_pool(h, pyg_batch.batch) # (B, hidden_dim)

        # Node embeddings reshape về (B, N, hidden_dim)
        node_emb = h.view(B, N, self.hidden_dim) # (B, N, hidden_dim)

        return graph_emb, node_emb
    

class ChipPlacementModel(nn.Module):
    """
    Actor-Critic model cho chip placement
        - Actor head: dựa trên graph_emb và node_emb để tính action logits (B, N) cho việc chọn cell đặt block hiện tại
        - Critic head: dựa trên graph_emb để dự đoán state value (B, 1)

    Forward:
        logits : (B, n_actions) - logit cho việc chọn cell đặt block hiện tại (trước khi apply action mask)
        value : (B, 1) - giá trị của state hiện tại
    """

    def __init__(
            self,
            n_blocks: int,
            n_actions: int,
            node_dim: int = 8,
            edge_dim: int = 1,
            hidden_dim: int = 128,
            n_layers: int = 3,
            canvas_emb_dim: int = 64,
            metadata_dim: int = 4,
            dropout: float = 0.1      
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        # Netlist GNN encoder
        self.gnn = NetlistGNN(
            node_in_dim=node_dim,
            edge_in_dim=edge_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )

        # Canvas encoder
        self.canvas_encoder = CanvasEncoder(out_dim=canvas_emb_dim)

        # Current node projection (để kết hợp với graph_emb và canvas_emb)
        self.cur_node_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Fusion (graph_emb, canvas_emb, cur_node_emb, metadata) -> hidden
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + canvas_emb_dim + hidden_dim + metadata_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU()
        )

        # Actor head: hidden -> action logits
        self.actor = nn.Linear(hidden_dim, n_actions)
        # Critic head: hidden -> state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        """Orthogonal init cho tất cả Linear layers (PPO best practice)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Actor head: gain nhỏ → uniform initial policy → better exploration
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

    
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        n_valid_edges: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: dict chứa các thành phần sau:
                - node_features: (B, N, node_dim)
                - edge_index: (B, 2, max_edges)
                - edge_features: (B, max_edges, edge_dim)
                - canvas: (B, 1, H, W)
                - current_node: (B, node_dim)
                - metadata: (B, metadata_dim)
            n_valid_edges: (B,) - số lượng edge hợp lệ trong edge_index và edge_features (để unpad)

        Returns:
            - logits: (B, n_actions) - logit cho việc chọn cell đặt block hiện tại
            - value: (B, 1) - giá trị của state hiện tại
        """
        nf = obs["node_features"].float() # (B, N, node_dim)
        ei = obs["edge_index"].long() # (B, 2, max_edges)
        ef = obs["edge_features"].float() # (B, max_edges, edge_dim)
        canvas = obs["canvas"].float() # (B, 1, H, W)
        cur_node = obs["current_node"].float() # (B, node_dim)
        metadata = obs["metadata"].float() # (B, metadata_dim)

        B, N, _ = nf.shape

        # netlist graph encoding
        graph_emb, node_emb = self.gnn(nf, ei, ef, n_valid_edges) # graph_emb: (B, hidden_dim), node_emb: (B, N, hidden_dim)

        # current node
        is_current = obs["node_features"][:, :, 7].bool() # (B, N) - node_dim=8, bit 7 là is_current
        # Nếu không có block nào được đánh dấu (episode done), dùng zeros
        cur_mask = is_current.any(dim=1) # (B,) - có block nào được đánh dấu không
        # Lấy index của current node mỗi sample
        cur_idx = is_current.long().argmax(dim=1) # (B,) - index của node hiện tại, nếu không có node nào được đánh dấu thì trả về 0 (sẽ bị mask sau)
        cur_node_emb = node_emb[torch.arange(B), cur_idx] # (B, hidden_dim)
        # Zero out nếu không có node nào được đánh dấu
        cur_node_emb = cur_node_emb * cur_mask.float().unsqueeze(1)
        cur_node_emb = self.cur_node_proj(cur_node_emb) # (B, hidden_dim)

        # Canvas encoding
        canvas_emb = self.canvas_encoder(canvas) # (B, canvas_emb_dim)

        # Fusion
        fused = torch.cat([graph_emb, canvas_emb, cur_node_emb, metadata], dim=1) # (B, hidden_dim + canvas_emb_dim + hidden_dim + metadata_dim)
        state_emb = self.fusion(fused) # (B, hidden_dim)

        # Actor head
        logits = self.actor(state_emb) # (B, n_actions)
        # Critic head
        value = self.critic(state_emb) # (B, 1)

        return logits, value

    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        action: Optional[torch.Tensor] = None,
        n_valid_edges: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dùng trong rollout
        Args:
            - obs: batched obs dict
            - actions: (B,) nếu evaluate existing actions, None nếu cần model chọn action

        Returns:
            - action: (B,)
            - logprob: (B,)
            - entropy: (B,)
            - value: (B,)
        """

        logits, value = self.forward(obs, n_valid_edges) # (B, n_actions), (B, 1)

        # Action masking
        mask = obs["action_mask"] # (B, n_actions), bool
        masked_logits = logits.masked_fill(~mask, -1e9) # (B, n_actions)

        dist = torch.distributions.Categorical(logits=masked_logits)

        if action is None:
            action = dist.sample() # (B,)

        logprob = dist.log_prob(action) # (B,)
        entropy = dist.entropy() # (B,)
        return action, logprob, entropy, value.squeeze(-1) # (B,), (B,), (B,), (B,)