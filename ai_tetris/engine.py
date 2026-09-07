"""与 3d-tetris.html 规则一致的 3D 俄罗斯方块逻辑模拟器。

训练时使用离散 step：动作 = 目标落点 (x, z, orientation_slot, hold)，
执行时等价于：先用 C 暂存（可选），再在出生层把方块摆到目标姿态和水平位置，
然后空格硬降到目标层，锁定并结算分数。
"""
from __future__ import annotations

import random
from typing import Iterable, Optional

from . import constants as C


# ---------------------------------------------------------------- 旋转/归一化
def normalize_cells(cells: Iterable[Iterable[int]]) -> list[list[int]]:
    cells = [[int(v) for v in c] for c in cells]
    minx = min(c[0] for c in cells)
    miny = min(c[1] for c in cells)
    minz = min(c[2] for c in cells)
    return [[c[0] - minx, c[1] - miny, c[2] - minz] for c in cells]


def rot_y_ccw(cells):
    return normalize_cells([[c[2], c[1], -c[0]] for c in cells])


def rot_y_cw(cells):
    return normalize_cells([[-c[2], c[1], c[0]] for c in cells])


def rot_z_ccw(cells):
    return normalize_cells([[-c[1], c[0], c[2]] for c in cells])


def rot_z_cw(cells):
    return normalize_cells([[c[1], -c[0], c[2]] for c in cells])


ROT_FUNCS = {
    "rotYCCW": rot_y_ccw,
    "rotYCW": rot_y_cw,
    "rotZCCW": rot_z_ccw,
    "rotZCW": rot_z_cw,
}

# 旋转踢墙（与 JS 一致，顺序一致）
KICKS = [
    (0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0),
]


def cells_to_tuple(cells) -> tuple:
    return tuple(sorted((int(c[0]), int(c[1]), int(c[2])) for c in cells))


def orientations_for_shape(shape: str) -> list[list[list[int]]]:
    """返回该方块所有可达的唯一姿态（BFS 枚举 4 种旋转操作）。"""
    start = normalize_cells(C.SHAPES[shape])
    seen = {cells_to_tuple(start)}
    result = [start]
    q = [start]
    while q:
        cells = q.pop()
        for op in C.ROT_OPS:
            nc = normalize_cells(ROT_FUNCS[op](cells))
            t = cells_to_tuple(nc)
            if t not in seen:
                seen.add(t)
                result.append(nc)
                q.append(nc)
    # 固定顺序，便于训练可复现
    result.sort(key=lambda c: cells_to_tuple(c))
    return result


# 每个方块的姿态表（长度不一，最多 24）
ORIENTATIONS = {s: orientations_for_shape(s) for s in C.PIECE_TYPES}
ORI_INDEX = {s: {cells_to_tuple(c): i for i, c in enumerate(ORIENTATIONS[s])} for s in C.PIECE_TYPES}


# ---------------------------------------------------------------- 高度图/落点
def compute_height_map(board) -> list[list[int]]:
    """height[x][z] = 该列最高占用 y + 1；空列为 0。"""
    heights = [[0] * C.D for _ in range(C.W)]
    for x in range(C.W):
        for z in range(C.D):
            m = 0
            for y in range(C.H):
                if board[x][y][z]:
                    m = y + 1
            heights[x][z] = m
    return heights


def drop_y_from_heights(heights, x: int, z: int, cells) -> int:
    """从出生层 H 下落时最终的 base y。

    对每个小格 (dx,dy,dz)，其所在列要求 y >= height[col] - dy。
    最终 y = max(0, 这些值)。此公式与 JS computeGhost 从 y=H 开始下落的
    结果一致（只要 y + max_dy < H 且 x,z 在边界内）。
    """
    y = 0
    for dx, dy, dz in cells:
        col_h = heights[x + dx][z + dz]
        need = col_h - dy
        if need > y:
            y = need
    return y


def drop_y(board, x: int, z: int, cells, start_y: int = C.H) -> int:
    """按 JS computeGhost 逻辑，从 start_y 开始求 ghost y（调试/回退用）。"""
    y = start_y
    while y > 0 and can_place(board, x, z, y - 1, cells):
        y -= 1
    return y


# ---------------------------------------------------------------- 棋盘/占用
def new_board() -> list:
    b = []
    for x in range(C.W):
        b.append([])
        for y in range(C.H):
            b[x].append([False] * C.D)
    return b


def cell_occupied(board, x: int, y: int, z: int) -> bool:
    if y < 0:
        return True
    if y >= C.H:
        return False
    return bool(board[x][y][z])


def can_place(board, x: int, z: int, y: int, cells) -> bool:
    for dx, dy, dz in cells:
        X = x + dx
        Y = y + dy
        Z = z + dz
        if X < 0 or X >= C.W or Z < 0 or Z >= C.D:
            return False
        if cell_occupied(board, X, Y, Z):
            return False
    return True


def is_plane_full(board, y: int) -> bool:
    for x in range(C.W):
        for z in range(C.D):
            if not board[x][y][z]:
                return False
    return True


def is_board_empty(board) -> bool:
    for x in range(C.W):
        for y in range(C.H):
            for z in range(C.D):
                if board[x][y][z]:
                    return False
    return True


def clear_plane(board, y: int):
    for yy in range(y, C.H - 1):
        for x in range(C.W):
            for z in range(C.D):
                board[x][yy][z] = board[x][yy + 1][z]
    for x in range(C.W):
        for z in range(C.D):
            board[x][C.H - 1][z] = False


# ---------------------------------------------------------------- 主引擎
class TetrisEngine:
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.board = new_board()
        self.queue: list[str] = []
        self.hold: Optional[str] = None
        self.piece: Optional[dict] = None
        self.ghost_y = 0
        self.state = {
            "status": "idle",
            "score": 0,
            "level": 1,
            "lines": 0,
            "planes": 0,
            "combo": -1,
            "canHold": True,
            "resetCount": 0,
            "lockAcc": 0,
            "dropAcc": 0,
            "softHeld": False,
        }
        self.done = False

    # ---------------- 基础操作 ----------------
    def reset(self):
        self.rng = random.Random(self.rng.random())  # 保持种子序列但可重新开新局
        self.board = new_board()
        self.queue = []
        self.hold = None
        self.piece = None
        self.ghost_y = 0
        self.state = {
            "status": "playing",
            "score": 0,
            "level": 1,
            "lines": 0,
            "planes": 0,
            "combo": -1,
            "canHold": True,
            "resetCount": 0,
            "lockAcc": 0,
            "dropAcc": 0,
            "softHeld": False,
        }
        self.done = False
        while len(self.queue) < C.QUEUE_INITIAL:
            self._refill_bag()
        self._spawn()
        return self

    def _refill_bag(self):
        bag = C.BAG.copy()
        self.rng.shuffle(bag)
        self.queue.extend(bag)

    def _spawn(self, type_: Optional[str] = None):
        while len(self.queue) < C.QUEUE_INITIAL:
            self._refill_bag()
        t = type_ or self.queue.pop(0)
        cells = [c.copy() for c in C.SHAPES[t]]
        mx = max(c[0] for c in cells)
        mz = max(c[2] for c in cells)
        self.piece = {
            "type": t,
            "cells": cells,
            "x": (C.W - (mx + 1)) // 2,
            "z": (C.D - (mz + 1)) // 2,
            "y": C.H,
            "spawnT": 0,
        }
        self.state["canHold"] = True
        self.state["resetCount"] = 0
        self.state["lockAcc"] = 0
        self.state["dropAcc"] = 0
        if not can_place(self.board, self.piece["x"], self.piece["z"], self.piece["y"], cells):
            if can_place(self.board, self.piece["x"], self.piece["z"], C.H - 1, cells):
                self.piece["y"] = C.H - 1
            else:
                self.done = True
                self.state["status"] = "over"
                return
        self.ghost_y = drop_y(self.board, self.piece["x"], self.piece["z"], cells, start_y=self.piece["y"])

    def _compute_ghost(self):
        p = self.piece
        self.ghost_y = drop_y(self.board, p["x"], p["z"], p["cells"], start_y=p["y"])

    def _lock_piece(self) -> int:
        """锁定当前方块，结算消除/分数，并生成下一块。返回本步锁定时产生的分数。

        注意：不包含硬降距离分；硬降分由调用者先加入 state.score。
        """
        p = self.piece
        if p is None or self.done:
            return 0
        if p["y"] >= C.H:
            self.done = True
            self.state["status"] = "over"
            return 0
        score_before = self.state["score"]

        # 安全检查：不应越界（训练动作已保证）；若越界按游戏结束处理
        for dx, dy, dz in p["cells"]:
            Y = p["y"] + dy
            if Y < 0 or Y >= C.H:
                self.done = True
                self.state["status"] = "over"
                return 0

        for dx, dy, dz in p["cells"]:
            X = p["x"] + dx
            Y = p["y"] + dy
            Z = p["z"] + dz
            self.board[X][Y][Z] = True

        # 检查方块覆盖的所有高度层（与 JS 一致），从高到低消除
        max_dy = max(c[1] for c in p["cells"])
        full_planes = []
        for yy in range(p["y"], min(p["y"] + max_dy + 1, C.H)):
            if is_plane_full(self.board, yy):
                full_planes.append(yy)
        full_planes.sort(reverse=True)
        cleared = len(full_planes)
        for yy in full_planes:
            clear_plane(self.board, yy)

        if cleared > 0:
            self.state["combo"] += 1
            pts = (C.PLANE_PTS[cleared] if cleared < len(C.PLANE_PTS) else 800) * self.state["level"]
            if self.state["combo"] > 0:
                pts += 50 * self.state["level"] * self.state["combo"]
            if is_board_empty(self.board):
                pts += 2000 * self.state["level"]
            self.state["score"] += pts
            self.state["lines"] += cleared
            self.state["planes"] += cleared
            nl = 1 + self.state["planes"] // 6
            if nl > self.state["level"]:
                self.state["level"] = nl
        else:
            self.state["combo"] = -1

        self._spawn()
        return self.state["score"] - score_before

    def _do_hold(self):
        """执行暂存，与 JS holdPiece 一致。返回执行后当前方块类型。"""
        if self.state["status"] != "playing" or self.piece is None or not self.state["canHold"]:
            return None
        t = self.piece["type"]
        if self.hold is not None:
            h = self.hold
            self.hold = t
            self._spawn(h)
        else:
            self.hold = t
            self._spawn()
        self.state["canHold"] = False
        return self.piece["type"] if self.piece else None

    # ---------------- 训练用离散 step ----------------
    def legal_action_mask(self, horizontal_only: bool = False) -> list[bool]:
        """返回长度 ACTION_SIZE 的合法动作 mask。

        合法动作：在出生层可完成的目标落点 (x, z, 姿态, 是否 hold)。
        horizontal_only=True 时只保留水平姿态（dy=0），动作空间更小、
        训练更快，仍可正常消层。
        """
        mask = [False] * C.ACTION_SIZE
        if self.done or self.state["status"] != "playing" or self.piece is None:
            return mask

        heights = compute_height_map(self.board)

        # 不暂存：当前方块
        self._mask_for_piece(mask, self.piece["type"], heights, hold=0,
                             horizontal_only=horizontal_only)
        # 暂存：先 hold，再摆放 hold 后的方块
        if self.state["canHold"]:
            after_hold = self.hold if self.hold is not None else (self.queue[0] if self.queue else None)
            if after_hold is not None:
                self._mask_for_piece(mask, after_hold, heights, hold=1,
                                     horizontal_only=horizontal_only)
        return mask

    def _mask_for_piece(self, mask: list[bool], ptype: str, heights, hold: int,
                        horizontal_only: bool = False):
        oris = ORIENTATIONS[ptype]
        for ori_slot, cells in enumerate(oris):
            max_dy = max(c[1] for c in cells)
            if horizontal_only and max_dy != 0:
                continue
            for x in range(C.W):
                for z in range(C.D):
                    # 足迹必须在棋盘内
                    ok = True
                    for dx, dy, dz in cells:
                        if not (0 <= x + dx < C.W and 0 <= z + dz < C.D):
                            ok = False
                            break
                    if not ok:
                        continue
                    y = drop_y_from_heights(heights, x, z, cells)
                    # y 必须可锁定：y < H 且最高小格也在棋盘内
                    if y >= C.H:
                        continue
                    if y + max_dy >= C.H:
                        continue
                    idx = (((x * C.D + z) * C.MAX_ORI + ori_slot) * 2) + hold
                    mask[idx] = True

    def valid_action_indices(self) -> list[int]:
        return [i for i, m in enumerate(self.legal_action_mask()) if m]

    def decode_action(self, action: int):
        """动作 -> (x, z, ori_slot, hold)。"""
        hold = action & 1
        a = action >> 1
        ori_slot = a % C.MAX_ORI
        a //= C.MAX_ORI
        z = a % C.D
        x = a // C.D
        return x, z, ori_slot, hold

    @staticmethod
    def encode_action(x: int, z: int, ori_slot: int, hold: int) -> int:
        return (((x * C.D + z) * C.MAX_ORI + ori_slot) * 2) + hold

    def step(self, action: int):
        """执行一个目标落点动作。返回 (reward, done, info)。"""
        if self.done or self.state["status"] != "playing" or self.piece is None:
            return 0.0, True, {"invalid": True}

        x, z, ori_slot, hold = self.decode_action(action)
        score_before = self.state["score"]

        # 可选暂存
        if hold:
            if not self.state["canHold"]:
                return 0.0, True, {"invalid": True}
            new_type = self._do_hold()
            if new_type is None:
                return 0.0, True, {"invalid": True}
            ptype = new_type
        else:
            ptype = self.piece["type"]

        if ori_slot >= len(ORIENTATIONS[ptype]):
            return 0.0, True, {"invalid": True}
        cells = ORIENTATIONS[ptype][ori_slot]

        # 足迹合法性
        for dx, dy, dz in cells:
            if not (0 <= x + dx < C.W and 0 <= z + dz < C.D):
                return 0.0, True, {"invalid": True}

        heights = compute_height_map(self.board)
        y = drop_y_from_heights(heights, x, z, cells)
        max_dy = max(c[1] for c in cells)
        if y >= C.H or y + max_dy >= C.H:
            return 0.0, True, {"invalid": True}

        # 设置方块并硬降
        self.piece["x"] = x
        self.piece["z"] = z
        self.piece["cells"] = [c.copy() for c in cells]
        dist = self.piece["y"] - y
        self.piece["y"] = y
        self.state["score"] += dist * 2  # 硬降分
        self._compute_ghost()

        # 锁定并生成下一块
        self._lock_piece()
        reward = self.state["score"] - score_before
        info = {
            "invalid": False,
            "piece_placed": ptype,
            "dist": dist,
            "done": self.done,
            "score": self.state["score"],
            "lines": self.state["lines"],
            "planes": self.state["planes"],
            "level": self.state["level"],
        }
        return float(reward), self.done, info

    # ---------------- 辅助 ----------------
    def piece_ori_slot(self) -> int:
        p = self.piece
        if p is None:
            return 0
        t = cells_to_tuple(p["cells"])
        return ORI_INDEX[p["type"]].get(t, 0)
