"""
placement_util.py
─────────────────
Core data structures và metrics theo circuit_training / AlphaChip.
 
References:
  - Nature 2021: A graph placement methodology for fast chip design
  - circuit_training/environment/placement_util.py (Google Research)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict
import numpy as np

@dataclass
class Block:
    name: str
    width: float
    height: float
    is_port: bool = False
    is_fixed: bool = False

    # Sau khi placement
    x: Optional[float] = None # x gốc bottom-left
    y: Optional[float] = None  # y gốc bottom-left

    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def cx(self) -> Optional[float]:
        return self.x + self.width / 2 if self.x is not None else None
    
    @property
    def cy(self) -> Optional[float]:
        return self.y + self.height / 2 if self.y is not None else None
    
    @property
    def is_placed(self) -> bool:
        return self.x is not None and self.y is not None

@dataclass
class Net:
    """
    Hyperedge trong netlist, kết nối nhiều block qua wire.
    """
    name: str
    block_indices: List[int] 
    weight: float = 1.0

class PlacementGrid:
    
    def __init__(self, cols: int, rows: int):
        self.cols = cols
        self.rows = rows

        self.cell_w = 1.0 / cols
        self.cell_h = 1.0 / rows

        self.grid = np.zeros((rows, cols), dtype=np.float32)  # occupancy map

    def reset(self):
        self.grid[:] = 0

    # Coordinate Conversions
    def xy_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Normalized (x, y) to grid cell (row, col)"""
        col = int(np.clip(x / self.cell_w, 0, self.cols - 1))
        row = int(np.clip(y / self.cell_h, 0, self.rows - 1))
        return row, col
    
    def cell_to_xy(self, row: int, col: int) -> Tuple[float, float]:
        """Grid cell (row, col) to normalized (x, y) of cell center"""
        x = (col + 0.5) * self.cell_w
        y = (row + 0.5) * self.cell_h
        return x, y
    
    def action_to_cell(self, action: int) -> Tuple[int, int]:
        """Action index to grid cell (row, col)"""
        row = action // self.cols
        col = action % self.cols
        return row, col
    
    def cell_to_action(self, row: int, col: int) -> int:
        """Grid cell (row, col) to action index"""
        return row * self.cols + col

    # Placement check

    def block_cells(self, col: int, row: int, block: Block) -> Tuple[int, int, int, int]:
        """Tính (col_end, row_end) của block nếu đặt tại (col, row)
           Clamp để block không vượt quá canvas        
        """
        bw = max(1, int(np.ceil(block.width / self.cell_w)))
        bh = max(1, int(np.ceil(block.height / self.cell_h)))

        col = min(col, self.cols - bw)
        row = min(row, self.rows - bh)

        col = max(col, 0)
        row = max(row, 0)

        return col, row, col + bw, row + bh

    def candidate_xy(self, col: int, row: int, block: Block) -> Tuple[float, float]:
        """Bottom-left coordinates for placing block at the given cell."""
        c0, r0, _, _ = self.block_cells(col, row, block)
        x = (c0 + 0.5) * self.cell_w - block.width / 2
        y = (r0 + 0.5) * self.cell_h - block.height / 2
        x = float(np.clip(x, 0.0, 1.0 - block.width))
        y = float(np.clip(y, 0.0, 1.0 - block.height))
        return x, y

    @staticmethod
    def _rects_overlap(a: Block, b: Block) -> bool:
        if a.x is None or a.y is None or b.x is None or b.y is None:
            return False
        return not (
            a.x + a.width <= b.x or
            b.x + b.width <= a.x or
            a.y + a.height <= b.y or
            b.y + b.height <= a.y
        )
    
    def is_valid(
        self,
        col: int,
        row: int,
        block: Block,
        placed_blocks: Optional[List[Block]] = None,
    ) -> bool:
        """Kiểm tra nếu block có thể đặt tại (col, row) mà không chồng lấn hay OOB"""
        c0, r0, c1, r1 = self.block_cells(col, row, block)
        if c1 > self.cols or r1 > self.rows:
            return False

        if not bool(np.all(self.grid[r0:r1, c0:c1] == 0)):
            return False

        if placed_blocks:
            candidate = Block(block.name, block.width, block.height)
            candidate.x, candidate.y = self.candidate_xy(col, row, block)
            for other in placed_blocks:
                if other is block:
                    continue
                if other.is_placed and self._rects_overlap(candidate, other):
                    return False

        return True
                    
    def place(self, block_id: int, col: int, row: int, block: Block):
        """Đặt block tại (col, row) và cập nhật grid occupancy"""
        c0, r0, c1, r1 = self.block_cells(col, row, block)
        self.grid[r0:r1, c0:c1] = block_id

    def oppupancy_map(self) -> np.ndarray:
        """Trả về occupancy map (1 nếu có block, 0 nếu trống)"""
        return (self.grid > 0).astype(np.float32)
    
    def get_action_mask(
        self,
        block: Block,
        placed_blocks: Optional[List[Block]] = None,
    ) -> np.ndarray:
        """Trả về boolen mask shape (rows*cols)"""
        bw = max(1, int(np.ceil(block.width / self.cell_w)))
        bh = max(1, int(np.ceil(block.height / self.cell_h)))

        mask = np.zeros((self.rows, self.cols), dtype=bool)

        for r in range(self.rows - bh + 1):
            for c in range(self.cols - bw + 1):
                if self.is_valid(c, r, block, placed_blocks=placed_blocks):
                    mask[r, c] = True
        return mask.flatten() # (rows*cols,)


def compute_hpwl(netlist: List[Net], blocks: List[Block]) -> float:
    """
    Tính HPWL (Half-Perimeter Wire Length) cho netlist và block placements.
    Công thức : HPWL = sum_over_nets( (max_x - min_x) + (max_y - min_y)) 
    """
    total = 0.0
    for net in netlist:
        xs, ys = [], []
        for idx in net.block_indices:
            b = blocks[idx]
            if b.cx is not None and b.cy is not None:
                xs.append(b.cx)
                ys.append(b.cy)
        if len(xs) >= 2:
            total += net.weight * ((max(xs) - min(xs)) + (max(ys) - min(ys)))
    return total

def compute_congestion(nets: List[Net], blocks: List[Block], grid_rows: int = 32, grid_cols: int = 32) -> float:
    """
     Proxy routing congestion (simplified Steiner tree model).
        - Chia canvas thành lưới 32x32, tính congestion score dựa trên số net crossing mỗi cell.
    """
    h_demand = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    v_demand = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    cw = 1.0 / grid_cols
    ch = 1.0 / grid_rows

    for net in nets:
        xs, ys = [], []
        for idx in net.block_indices:
            b = blocks[idx]
            if b.cx is not None and b.cy is not None:
                xs.append(b.cx)
                ys.append(b.cy)
        if len(xs) < 2:
            continue

        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        c0 = int(x0 / cw)
        c1 = min(int(x1 / cw), grid_cols - 1)
        r0 = int(y0 / ch)
        r1 = min(int(y1 / ch), grid_rows - 1)

        if c1 > c0:
            h_demand[r0:r1+1, c0:c1] += net.weight
        if r1 > r0:
            v_demand[r0:r1, c0:c1+1] += net.weight

    h_cap = max(h_demand.mean(), 1e-8)
    v_cap = max(v_demand.mean(), 1e-8)
    return float(max(h_demand.max() / h_cap, v_demand.max() / v_cap))

def compute_density(
        blocks: List[Block],
        grid_rows: int = 32,
        grid_cols: int = 32
    ) -> float:
    """
    Max cell density
    """
    density = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    cw = 1.0 / grid_cols
    ch = 1.0 / grid_rows

    for b in blocks:
        if not b.is_placed:
            continue
        assert b.x is not None and b.y is not None
        c0 = int(b.x / cw)
        c1 = min(int((b.x + b.width) / cw), grid_cols - 1)
        r0 = int(b.y / ch)
        r1 = min(int((b.y + b.height) / ch), grid_rows - 1)
        area_cell = cw * ch
        block_area_in_cell = (b.width / (c1 - c0 + 1)) * (b.height / (r1 - r0 + 1))
        density[r0:r1+1, c0:c1+1] += block_area_in_cell / area_cell

    return float(density.max())

def get_placement_order(blocks: List[Block]) -> List[int]:
    """
    Sắp xếp block theo thứ tự giảm dần về kích thước (largest first).
    - Rationale: Đặt các block lớn trước giúp tránh fragmentation canvas.
    - Fixed/port được bỏ qua
    """
    
    movable = [
        (i, b) for i, b in enumerate(blocks) if not b.is_fixed and not b.is_port
    ]

    movable.sort(key=lambda x: x[1].area, reverse=True)

    return [i for i, _ in movable]

