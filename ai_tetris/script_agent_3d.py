"""脚本策略 v4：开放 3D 旋转，优先填补最低未满层的空缺形状。"""
from __future__ import annotations

from . import constants as C
from .engine import ORIENTATIONS, TetrisEngine


def _heightmap(board):
    hs = []
    holes = 0
    for x in range(C.W):
        for z in range(C.D):
            h = 0
            for y in range(C.H):
                if board[x][y][z]:
                    h = y + 1
            hs.append(h)
            for y in range(h):
                if not board[x][y][z]:
                    holes += 1
    return hs, holes


def _layer_counts(board):
    counts = [0] * C.H
    for y in range(C.H):
        cnt = 0
        for x in range(C.W):
            for z in range(C.D):
                if board[x][y][z]:
                    cnt += 1
        counts[y] = cnt
    return counts


def _bump(hs):
    bump = 0
    for x in range(C.W):
        for z in range(C.D):
            idx = x * C.D + z
            if z + 1 < C.D:
                bump += abs(hs[idx] - hs[idx + 1])
            if x + 1 < C.W:
                bump += abs(hs[idx] - hs[idx + C.D])
    return bump


def script_action_3d(env: TetrisEngine) -> int | None:
    mask = env.legal_action_mask(horizontal_only=False)
    idxs = [i for i, m in enumerate(mask) if m]
    if not idxs:
        return None

    board = env.board
    hs, holes = _heightmap(board)
    counts = _layer_counts(board)
    ptype = env.piece["type"]

    # 最低未满层
    lowest_y = C.H
    for yy in range(C.H):
        if counts[yy] < C.W * C.D:
            lowest_y = yy
            break

    best_a = None
    best_score = None

    for a in idxs:
        x, z, ori_slot, hold = env.decode_action(a)
        pt = ptype
        if hold:
            pt = env.hold if env.hold is not None else (env.queue[0] if env.queue else None)
            if pt is None:
                continue
        if ori_slot >= len(ORIENTATIONS.get(pt, [])):
            continue
        cells = ORIENTATIONS[pt][ori_slot]
        max_dy = max(c[1] for c in cells)

        # 边界检查
        ok = True
        for dx, dy, dz in cells:
            if not (0 <= x + dx < C.W and 0 <= z + dz < C.D):
                ok = False
                break
        if not ok:
            continue

        # 用高度图计算落点层 y（与 engine 一致）
        y = 0
        for dx, dy, dz in cells:
            idx = (x + dx) * C.D + (z + dz)
            need = hs[idx] - dy
            if need > y:
                y = need
        if y >= C.H or y + max_dy >= C.H:
            continue

        # 目标层填充匹配：方块最低层正好落在最低未满层
        fill_match = 0
        if y == lowest_y:
            for dx, dy, dz in cells:
                if dy == 0:
                    if not board[x + dx][y][z + dz]:
                        fill_match += 1

        # 近似模拟放置后的高度图/空洞
        new_hs = hs[:]
        new_holes = holes
        for dx, dy, dz in cells:
            idx = (x + dx) * C.D + (z + dz)
            cy = y + dy
            old = hs[idx]
            if old <= cy:
                new_holes += cy - old
                new_hs[idx] = cy + 1

        maxh = max(new_hs)
        avg = sum(new_hs) / len(new_hs)
        bump = _bump(new_hs)

        # 评分：优先形状匹配填空缺，其次低矮、少洞、平整
        score = (20.0 * fill_match
                 - 1.0 * maxh - 0.2 * avg
                 - 0.3 * new_holes - 0.05 * bump)
        if best_score is None or score > best_score:
            best_score = score
            best_a = a

    return best_a
