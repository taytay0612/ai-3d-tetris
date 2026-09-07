"""路径规划：把“目标落点”转换成真实游戏可执行的按键序列。

真实网页执行时，游戏可能已经下落了几格；BFS 会从当前 (x,z,y,orientation)
出发，严格按 JS 的 moveDir / tryRotate 规则（包括踢墙顺序）搜索一条到达
目标落点的最短按键路径。找到后由 browser_ctl 逐键发送。
"""
from __future__ import annotations

from collections import deque

from . import constants as C
from .engine import (
    KICKS,
    ORIENTATIONS,
    ORI_INDEX,
    ROT_FUNCS,
    can_place,
    cells_to_tuple,
    compute_height_map,
    drop_y_from_heights,
)

# 注意：BFS 只考虑 tryRotate 的第一个成功踢墙，和 JS 完全一致。
# 因此父指针只需记录操作名，执行时发送同一个旋转键即可复现。


def board_from_browser(js_board) -> list:
    """浏览器 board[x][y][z](0/1) -> engine bool board。"""
    board = []
    for x in range(C.W):
        board.append([])
        for y in range(C.H):
            board[x].append([bool(js_board[x][y][z]) for z in range(C.D)])
    return board


def plan_path(board, ptype: str, start_x: int, start_z: int, start_y: int,
              start_cells, target_x: int, target_z: int, target_cells):
    """BFS 寻找按键序列。返回 list[str]（moveDir/ROT 操作名）或 None。"""
    oris = ORIENTATIONS[ptype]
    start_ori = ORI_INDEX[ptype].get(cells_to_tuple(start_cells))
    target_ori = ORI_INDEX[ptype].get(cells_to_tuple(target_cells))
    if start_ori is None or target_ori is None:
        return None

    heights = compute_height_map(board)
    target_y = drop_y_from_heights(heights, target_x, target_z, target_cells)
    if target_y >= C.H:
        return None

    start = (start_x, start_z, start_y, start_ori)
    parent = {start: None}
    action_parent = {start: None}
    q = deque([start])
    visited = {start}

    while q:
        x, z, y, ori = q.popleft()
        cells = oris[ori]

        # 到达目标：x,z,姿态一致，且从当前高度硬降能落到 target_y
        if x == target_x and z == target_z and ori == target_ori and y >= target_y:
            # 重建路径
            ops = []
            cur = (x, z, y, ori)
            while parent[cur] is not None:
                ops.append(action_parent[cur])
                cur = parent[cur]
            ops.reverse()
            return ops

        # 四个水平移动
        for op, (dx, dz) in C.MOVE_DIR.items():
            nx, nz = x + dx, z + dz
            ns = (nx, nz, y, ori)
            if ns not in visited and can_place(board, nx, nz, y, cells):
                visited.add(ns)
                parent[ns] = (x, z, y, ori)
                action_parent[ns] = op
                q.append(ns)

        # 四种旋转：与 JS tryRotate 相同，只取第一个成功踢墙
        for op in C.ROT_OPS:
            nc = ROT_FUNCS[op](cells)
            nori = ORI_INDEX[ptype].get(cells_to_tuple(nc), ori)
            for kx, kz, ky in KICKS:
                nx, ny, nz = x + kx, y + ky, z + kz
                if ny < 0 or ny > C.H + 1:
                    continue
                if can_place(board, nx, nz, ny, nc):
                    ns = (nx, nz, ny, nori)
                    if ns not in visited:
                        visited.add(ns)
                        parent[ns] = (x, z, y, ori)
                        action_parent[ns] = op
                        q.append(ns)
                    break  # 只取第一个成功踢墙，与 JS 一致

    return None


def all_placement_mask_for_state(board, ptype: str, can_hold: bool,
                                 hold_type, queue0, horizontal_only: bool = False) -> list[bool]:
    """从当前方块/棋盘计算全落点 mask（假设从出生层可自由移动）。"""
    mask = [False] * C.ACTION_SIZE
    heights = compute_height_map(board)
    _mask_piece(mask, ptype, heights, 0, horizontal_only)
    if can_hold:
        after = hold_type if hold_type is not None else queue0
        if after is not None:
            _mask_piece(mask, after, heights, 1, horizontal_only)
    return mask


def _mask_piece(mask: list[bool], ptype: str, heights, hold: int,
                horizontal_only: bool = False):
    oris = ORIENTATIONS[ptype]
    for ori_slot, cells in enumerate(oris):
        max_dy = max(c[1] for c in cells)
        if horizontal_only and max_dy != 0:
            continue
        for x in range(C.W):
            for z in range(C.D):
                ok = True
                for dx, dy, dz in cells:
                    if not (0 <= x + dx < C.W and 0 <= z + dz < C.D):
                        ok = False
                        break
                if not ok:
                    continue
                y = drop_y_from_heights(heights, x, z, cells)
                if y >= C.H or y + max_dy >= C.H:
                    continue
                idx = (((x * C.D + z) * C.MAX_ORI + ori_slot) * 2) + hold
                mask[idx] = True
