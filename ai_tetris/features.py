"""状态特征与动作 mask 构造。"""
from __future__ import annotations

import numpy as np

from . import constants as C
from . import engine as E

# 特征维度
BOARD_DIM = C.W * C.H * C.D          # 686
HEIGHT_DIM = C.W * C.D               # 49（每列高度）
HOLES_DIM = C.W * C.D                # 49（每列空洞数）
PIECE_DIM = len(C.PIECE_TYPES)       # 8
HOLD_DIM = len(C.PIECE_TYPES) + 1    # 9（无暂存 + 8 种）
NEXT_DIM = len(C.PIECE_TYPES) * 3    # 24
STATE_DIM = (BOARD_DIM + HEIGHT_DIM + HOLES_DIM + PIECE_DIM +
             HOLD_DIM + NEXT_DIM + 2)  # 827


def height_and_holes(board) -> tuple[list[float], list[float]]:
    """返回 (heights_normalized, holes_normalized)，长度均为 49。"""
    heights = []
    holes = []
    for x in range(C.W):
        for z in range(C.D):
            h = 0
            for y in range(C.H):
                if board[x][y][z]:
                    h = y + 1
            heights.append(h / C.H)
            hh = 0
            for y in range(h):
                if not board[x][y][z]:
                    hh += 1
            holes.append(hh / C.H)
    return heights, holes


def max_layer_fill(board) -> float:
    """所有水平层中填充格子的最大值（0..49）。"""
    best = 0
    for y in range(C.H):
        cnt = 0
        for x in range(C.W):
            for z in range(C.D):
                if board[x][y][z]:
                    cnt += 1
        if cnt > best:
            best = cnt
    return float(best)


def wells_from_heights(heights: list[float]) -> float:
    """计算井深总和（归一化）。heights 是 49 维 [0,1] 归一化高度。"""
    well = 0.0
    for x in range(C.W):
        for z in range(C.D):
            h = heights[x * C.D + z]
            for nx, nz in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
                if 0 <= nx < C.W and 0 <= nz < C.D:
                    nh = heights[nx * C.D + nz]
                    if nh > h + 0.05:
                        well += nh - h
    return well / C.H


def potential(board, done: bool) -> float:
    """势函数：- (最高列 + 0.6 * 总空洞 + 0.3 * 井深)。done 时返回 0。

    用于 potential-based reward shaping，不改变游戏得分的理论最优策略。
    """
    if done:
        return 0.0
    heights, holes = height_and_holes(board)
    max_h = max(heights)          # 已归一化到 [0,1]
    holes_sum = sum(holes)        # 已归一化
    well_sum = wells_from_heights(heights)
    return -(max_h + 0.6 * holes_sum + 0.3 * well_sum)


def _onehot(n: int, idx: int) -> list[float]:
    v = [0.0] * n
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def state_vector_from_browser(st: dict) -> np.ndarray:
    """从 browser_ctl.read_state() 返回的 dict 构造状态向量。"""
    from .planner import board_from_browser

    board = board_from_browser(st["board"])
    piece = st.get("piece")
    board_flat = [0.0] * BOARD_DIM
    i = 0
    for x in range(C.W):
        for y in range(C.H):
            for z in range(C.D):
                board_flat[i] = 1.0 if board[x][y][z] else 0.0
                i += 1

    heights, holes = height_and_holes(board)

    ptype = piece["type"] if piece else "1"
    piece_idx = C.PIECE_TYPES.index(ptype)
    hold = st.get("hold")
    if hold is None:
        hold_idx = len(C.PIECE_TYPES)
    else:
        hold_idx = C.PIECE_TYPES.index(hold)

    next_vec: list[float] = []
    queue = st.get("queue") or []
    for k in range(3):
        if k < len(queue):
            next_vec.extend(_onehot(len(C.PIECE_TYPES), C.PIECE_TYPES.index(queue[k])))
        else:
            next_vec.extend([0.0] * len(C.PIECE_TYPES))

    feats = (
        board_flat
        + heights
        + holes
        + _onehot(len(C.PIECE_TYPES), piece_idx)
        + _onehot(len(C.PIECE_TYPES) + 1, hold_idx)
        + next_vec
        + [min((st.get("level") or 1) / 20.0, 1.0)]
        + [1.0 if st.get("canHold") else 0.0]
    )
    return np.asarray(feats, dtype=np.float32)


def state_vector(game: "E.TetrisEngine") -> np.ndarray:
    """构造固定长度状态向量。"""
    st = game.state
    piece = game.piece

    board_flat = [0.0] * BOARD_DIM
    i = 0
    for x in range(C.W):
        for y in range(C.H):
            for z in range(C.D):
                board_flat[i] = 1.0 if game.board[x][y][z] else 0.0
                i += 1

    heights, holes = height_and_holes(game.board)

    ptype = piece["type"] if piece else "1"
    piece_idx = C.PIECE_TYPES.index(ptype)

    if game.hold is None:
        hold_idx = len(C.PIECE_TYPES)
    else:
        hold_idx = C.PIECE_TYPES.index(game.hold)

    next_vec: list[float] = []
    for k in range(3):
        if k < len(game.queue):
            next_vec.extend(_onehot(len(C.PIECE_TYPES), C.PIECE_TYPES.index(game.queue[k])))
        else:
            next_vec.extend([0.0] * len(C.PIECE_TYPES))

    feats = (
        board_flat
        + heights
        + holes
        + _onehot(len(C.PIECE_TYPES), piece_idx)
        + _onehot(len(C.PIECE_TYPES) + 1, hold_idx)
        + next_vec
        + [min(st["level"] / 20.0, 1.0)]
        + [1.0 if st["canHold"] else 0.0]
    )
    return np.asarray(feats, dtype=np.float32)


def action_mask(game: "E.TetrisEngine", horizontal_only: bool = False) -> np.ndarray:
    """返回 bool mask，长度 ACTION_SIZE。"""
    return np.asarray(game.legal_action_mask(horizontal_only=horizontal_only), dtype=bool)


def decode_action(action: int):
    return E.TetrisEngine.decode_action(None, action) if False else _decode(action)


def _decode(action: int):
    hold = action & 1
    a = action >> 1
    ori_slot = a % C.MAX_ORI
    a //= C.MAX_ORI
    z = a % C.D
    x = a // C.D
    return x, z, ori_slot, hold


def action_description(game: "E.TetrisEngine", action: int) -> dict:
    """人类可读的动作描述（调试用）。"""
    x, z, ori_slot, hold = _decode(action)
    ptype = game.piece["type"] if game.piece else "?"
    if hold:
        ptype = game.hold if game.hold is not None else (game.queue[0] if game.queue else "?")
    oris = E.ORIENTATIONS.get(ptype, [])
    cells = oris[ori_slot] if ori_slot < len(oris) else None
    return {"piece": ptype, "x": x, "z": z, "ori_slot": ori_slot, "hold": hold, "cells": cells}
