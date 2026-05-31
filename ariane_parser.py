"""
ariane_parser.py
────────────────
Parse Ariane RISC-V netlist (circuit_training .pb.txt format) thành
Block và Net objects dùng được với ChipFloorplanEnv.

Format: TensorFlow GraphDef proto (text format)
Source: circuit_training/environment/test_data/ariane/netlist.pb.txt

Node types trong file:
  MACRO      — hard macro (SRAM blocks), 133 nodes
  macro      — soft macro (clustered stdcells, Grp_X), 799 nodes
  PORT       — IO ports, fixed ở boundary, 1231 nodes
  MACRO_PIN  — pin của hard macro
  macro_pin  — pin của soft macro (Grp_X/Poutput_N, Grp_X/Pinput)

Net construction:
  Grp_X/Poutput_N.inputs = [MACRO_PIN hoặc PORT]
  → net kết nối Grp_X với parent của mỗi input

Canvas: bounding box của tất cả macros (micron)
  Ariane: ~371 × 337 micron
"""

from __future__ import annotations
import re
import os
import urllib.request
import collections
from typing import List, Tuple, Dict, Optional

from placement_util import Block, Net


# ── Constants ─────────────────────────────────────────────────────────

ARIANE_URL = (
    "https://raw.githubusercontent.com/google-research/circuit_training"
    "/main/circuit_training/environment/test_data/ariane/netlist.pb.txt"
)

# Canvas thực của Ariane (micron) — từ bounding box macros trong file
ARIANE_CANVAS_W = 356.592  # micron  (circuit_training header value)
ARIANE_CANVAS_H = 356.640  # micron


# ══════════════════════════════════════════════════════════════════════
#  Parser
# ══════════════════════════════════════════════════════════════════════

def _parse_all_nodes(content: str) -> List[Dict]:
    """Parse toàn bộ node blocks từ proto text."""
    raw_nodes = content.split("node {")[1:]
    parsed = []
    for raw in raw_nodes:
        name   = re.search(r'^\s*name:\s*"([^"]+)"', raw, re.MULTILINE)
        type_m = re.search(r'key:\s*"type".*?placeholder:\s*"([^"]+)"', raw, re.DOTALL)
        width  = re.search(r'key:\s*"width".*?f:\s*([0-9.e+\-]+)', raw, re.DOTALL)
        height = re.search(r'key:\s*"height".*?f:\s*([0-9.e+\-]+)', raw, re.DOTALL)
        x_m    = re.search(r'key:\s*"x".*?f:\s*([0-9.e+\-]+)', raw, re.DOTALL)
        y_m    = re.search(r'key:\s*"y".*?f:\s*([0-9.e+\-]+)', raw, re.DOTALL)
        side_m = re.search(r'key:\s*"side".*?placeholder:\s*"([^"]+)"', raw, re.DOTALL)
        inputs = re.findall(r'^\s*input:\s*"([^"]+)"', raw, re.MULTILINE)

        parsed.append({
            "name":   name.group(1) if name else "",
            "type":   type_m.group(1) if type_m else ("PORT" if side_m else "stdcell"),
            "width":  float(width.group(1)) if width else 0.0,
            "height": float(height.group(1)) if height else 0.0,
            "x":      float(x_m.group(1)) if x_m else None,
            "y":      float(y_m.group(1)) if y_m else None,
            "inputs": inputs,
        })
    return parsed


def parse_ariane_netlist(
    pb_path: str,
    canvas_w: float = ARIANE_CANVAS_W,
    canvas_h: float = ARIANE_CANVAS_H,
    include_ports: bool = False,
    min_macro_area: float = 0.0,
) -> Tuple[List[Block], List[Net], float, float]:
    """
    Parse Ariane .pb.txt → (blocks, nets, canvas_w, canvas_h).

    Args:
        pb_path        : đường dẫn tới netlist.pb.txt
        canvas_w/h     : kích thước canvas micron (normalize về [0,1])
        include_ports  : có include IO ports vào blocks không
                         (True = fixed blocks, False = bỏ qua)
        min_macro_area : bỏ qua soft macro (Grp_) có area < threshold
                         (micron²) để giảm số lượng blocks

    Returns:
        blocks   : List[Block] — tất cả movable macros (normalized)
        nets     : List[Net]   — connections giữa macros
        canvas_w : float (micron)
        canvas_h : float (micron)
    """
    with open(pb_path, "r") as f:
        content = f.read()

    all_nodes = _parse_all_nodes(content)
    name_to_node: Dict[str, Dict] = {n["name"]: n for n in all_nodes}

    # ── Phân loại nodes ───────────────────────────────────────────────
    hard_macros = [n for n in all_nodes if n["type"] == "MACRO"]       # SRAM
    soft_macros = [n for n in all_nodes if n["type"] == "macro"]       # Grp_
    ports       = [n for n in all_nodes if n["type"] == "PORT"]

    # Lọc soft macro theo diện tích tối thiểu
    soft_macros = [
        n for n in soft_macros
        if (n["width"] * n["height"]) >= min_macro_area
    ]

    movable_nodes = hard_macros + soft_macros
    if include_ports:
        movable_nodes += ports

    movable_names = {n["name"] for n in movable_nodes}
    hard_macro_names = {n["name"] for n in hard_macros}
    port_names = {n["name"] for n in ports}

    # ── Build Blocks ──────────────────────────────────────────────────
    blocks: List[Block] = []
    name_to_block_idx: Dict[str, int] = {}

    for node in movable_nodes:
        is_port  = node["type"] == "PORT"
        is_fixed = is_port  # ports fixed ở boundary
        w_norm   = node["width"]  / canvas_w
        h_norm   = node["height"] / canvas_h
        x_norm   = (node["x"] / canvas_w) if node["x"] is not None else None
        y_norm   = (node["y"] / canvas_h) if node["y"] is not None else None

        # Clamp kích thước tối thiểu 1 grid cell (sẽ được handle bởi env)
        w_norm = max(w_norm, 1e-4)
        h_norm = max(h_norm, 1e-4)

        idx = len(blocks)
        name_to_block_idx[node["name"]] = idx
        blocks.append(Block(
            name=node["name"],
            width=w_norm,
            height=h_norm,
            is_port=is_port,
            is_fixed=is_fixed,
            x=x_norm if is_fixed else None,
            y=y_norm if is_fixed else None,
        ))

    # ── Build Nets từ macro_pin (Grp_X/Poutput_N) ────────────────────
    # Grp_X/Poutput_N.inputs = list of MACRO_PIN or PORT names
    # → net kết nối Grp_X với parent macro của mỗi input
    macro_pin_nodes = [n for n in all_nodes if n["type"] == "macro_pin"
                       and "/Poutput" in n["name"]]

    nets: List[Net] = []
    net_idx = 0

    for pin_node in macro_pin_nodes:
        grp_name = pin_node["name"].rsplit("/", 1)[0]  # Grp_X
        if grp_name not in movable_names:
            continue

        connected: set = {grp_name}  # bắt đầu với Grp_X chính nó

        for inp in pin_node["inputs"]:
            # inp = 'MACRO_NAME/PIN_NAME' hoặc 'PORT_NAME'
            parts = inp.rsplit("/", 1)
            src_name = parts[0] if len(parts) > 1 else inp
            if src_name in movable_names and src_name != grp_name:
                connected.add(src_name)

        # Chỉ tạo net nếu có ít nhất 2 nodes khác nhau
        if len(connected) < 2:
            continue

        member_indices = [
            name_to_block_idx[n] for n in connected
            if n in name_to_block_idx
        ]
        if len(member_indices) < 2:
            continue

        nets.append(Net(
            name=f"net_{net_idx}",
            block_indices=member_indices,
            weight=float(len(connected) - 1),  # weight ~ fan-out
        ))
        net_idx += 1

    return blocks, nets, canvas_w, canvas_h


# ══════════════════════════════════════════════════════════════════════
#  Downloader
# ══════════════════════════════════════════════════════════════════════

def download_ariane(save_dir: str = "data/ariane") -> str:
    """
    Download Ariane netlist.pb.txt nếu chưa có.
    Returns đường dẫn tới file.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "netlist.pb.txt")

    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f"[Ariane] Already downloaded: {path} ({size_kb:.0f} KB)")
        return path

    print(f"[Ariane] Downloading from circuit_training repo...")
    try:
        urllib.request.urlretrieve(ARIANE_URL, path)
        size_kb = os.path.getsize(path) / 1024
        print(f"[Ariane] Downloaded: {size_kb:.0f} KB → {path}")
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}\nManually download from:\n{ARIANE_URL}")

    return path


# ══════════════════════════════════════════════════════════════════════
#  Preset configs
# ══════════════════════════════════════════════════════════════════════

def load_ariane(
    mode: str = "colab_fast",
    data_dir: str = "data/ariane",
) -> Tuple[List[Block], List[Net]]:
    """
    Load Ariane netlist với preset config cho các use-case khác nhau.

    mode:
        "colab_fast"   — chỉ 133 hard macros (SRAM), train ~1.5h trên Colab T4
                         Đây là config giống AlphaChip paper nhất
        "colab_full"   — 133 hard + 799 soft macros, train ~4-6h
        "debug"        — 20 hard macros đầu, test nhanh
    """
    pb_path = download_ariane(data_dir)

    if mode == "colab_fast":
        # Chỉ MACRO (hard SRAM macros) — 133 blocks
        # Đây là config chuẩn của circuit_training Ariane benchmark
        blocks, nets, cw, ch = parse_ariane_netlist(
            pb_path,
            include_ports=False,
            min_macro_area=999999.0,  # bỏ hết soft macro
        )
        # min_macro_area quá cao → không lấy soft macro
        # Parse lại chỉ hard macro
        blocks, nets, cw, ch = _load_hard_macros_only(pb_path)

    elif mode == "colab_full":
        # Hard + soft macros lớn (area > 5 micron²)
        blocks, nets, cw, ch = parse_ariane_netlist(
            pb_path,
            include_ports=False,
            min_macro_area=5.0,
        )

    elif mode == "debug":
        blocks, nets, cw, ch = _load_hard_macros_only(pb_path)
        # Chỉ giữ 20 blocks đầu và nets liên quan
        blocks = blocks[:20]
        kept = set(range(20))
        nets = [
            Net(n.name, [i for i in n.block_indices if i in kept], n.weight)
            for n in nets
            if sum(1 for i in n.block_indices if i in kept) >= 2
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'colab_fast', 'colab_full', 'debug'")

    print(f"[Ariane:{mode}] {len(blocks)} blocks, {len(nets)} nets")
    return blocks, nets


def _load_hard_macros_only(pb_path: str) -> Tuple[List[Block], List[Net], float, float]:
    """
    Load chỉ 133 hard SRAM macros và nets giữa chúng.
    Canvas normalize theo ARIANE_CANVAS_W/H.
    """
    with open(pb_path) as f:
        content = f.read()

    all_nodes = _parse_all_nodes(content)
    name_to_node = {n["name"]: n for n in all_nodes}

    hard_macros = [n for n in all_nodes if n["type"] == "MACRO"]
    soft_macros = [n for n in all_nodes if n["type"] == "macro"]

    cw, ch = ARIANE_CANVAS_W, ARIANE_CANVAS_H

    # Build blocks từ hard macros
    blocks: List[Block] = []
    name_to_idx: Dict[str, int] = {}
    for node in hard_macros:
        idx = len(blocks)
        name_to_idx[node["name"]] = idx
        blocks.append(Block(
            name=node["name"],
            width=max(node["width"] / cw, 1e-4),
            height=max(node["height"] / ch, 1e-4),
            is_port=False,
            is_fixed=False,
        ))

    hard_names = set(name_to_idx.keys())
    soft_names = {n["name"] for n in soft_macros}

    # Nets: Grp_ soft macros kết nối với hard macros
    # Grp_X/Poutput_N.inputs → [HARD_MACRO/PIN, ...]
    # → net: Grp_X và tất cả hard macros trong inputs
    # (Grp_ không có trong blocks, chỉ dùng như "wire hub")
    nets: List[Net] = []
    net_idx = 0

    macro_pin_nodes = [
        n for n in all_nodes
        if n["type"] == "macro_pin" and "/Poutput" in n["name"]
    ]

    seen_nets: set = set()  # deduplicate bằng frozenset of indices

    for pin_node in macro_pin_nodes:
        grp_name = pin_node["name"].rsplit("/", 1)[0]

        # Thu thập hard macros kết nối qua pin này
        connected_hard: set = set()

        # Check inputs của pin node → có hard macro nào không?
        for inp in pin_node["inputs"]:
            src = inp.rsplit("/", 1)[0]
            if src in hard_names:
                connected_hard.add(src)

        # Grp_ này có input từ MACRO_PIN node nào không?
        pinput_name = f"{grp_name}/Pinput"
        if pinput_name in name_to_node:
            pinput_node = name_to_node[pinput_name]
            for inp2 in pinput_node.get("inputs", []):
                src2 = inp2.rsplit("/", 1)[0]
                if src2 in hard_names:
                    connected_hard.add(src2)

        if len(connected_hard) < 2:
            continue

        member_indices = sorted([name_to_idx[n] for n in connected_hard])
        key = frozenset(member_indices)
        if key in seen_nets:
            continue
        seen_nets.add(key)

        nets.append(Net(
            name=f"net_{net_idx}",
            block_indices=member_indices,
            weight=float(len(connected_hard) - 1),
        ))
        net_idx += 1

    # Nếu quá ít nets, fallback: dùng spatial proximity
    if len(nets) < 10:
        nets = _build_proximity_nets(blocks, hard_macros, cw, ch)

    return blocks, nets, cw, ch


def _build_proximity_nets(
    blocks: List[Block],
    macro_nodes: List[Dict],
    canvas_w: float,
    canvas_h: float,
    k_nearest: int = 3,
) -> List[Net]:
    """
    Fallback: tạo nets dựa trên spatial proximity trong initial placement.
    Mỗi macro kết nối với k nearest neighbors.
    Dùng khi net extraction từ proto không tìm được đủ connections.
    """
    import numpy as np

    centers = []
    for node in macro_nodes:
        if node["x"] is not None:
            cx = (node["x"] + node["width"] / 2) / canvas_w
            cy = (node["y"] + node["height"] / 2) / canvas_h
        else:
            cx = cy = 0.5
        centers.append([cx, cy])
    centers = np.array(centers)

    nets = []
    seen = set()
    for i in range(len(centers)):
        dists = np.sum((centers - centers[i]) ** 2, axis=1)
        dists[i] = np.inf
        neighbors = np.argsort(dists)[:k_nearest]
        for j in neighbors:
            key = tuple(sorted([i, int(j)]))
            if key not in seen:
                seen.add(key)
                nets.append(Net(
                    name=f"prox_{i}_{j}",
                    block_indices=list(key),
                    weight=1.0 / (dists[j] + 1e-6),
                ))
    return nets


# ══════════════════════════════════════════════════════════════════════
#  CLI / sanity check
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "colab_fast"
    blocks, nets = load_ariane(mode=mode)

    print(f"\nBlocks ({len(blocks)}):")
    for b in blocks[:5]:
        print(f"  {b.name[:60]:60s}  w={b.width:.4f}  h={b.height:.4f}")
    print(f"  ...")

    print(f"\nNets ({len(nets)}):")
    for n in nets[:5]:
        print(f"  {n.name}: {n.block_indices[:4]}  w={n.weight:.2f}")
    print(f"  ...")

    areas = [b.width * b.height for b in blocks]
    print(f"\nBlock area: min={min(areas):.4f}, max={max(areas):.4f}, mean={sum(areas)/len(areas):.4f}")
    net_sizes = collections.Counter(len(n.block_indices) for n in nets)
    print(f"Net sizes: {dict(sorted(net_sizes.items()))}")