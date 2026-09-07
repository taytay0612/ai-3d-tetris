"""评分函数回归：数据生成。

对每个状态，用 3D 脚本策略计算所有合法动作的评分，
保存 top-K 高分动作 + 随机负样本，以及每个动作的特征和评分。
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from . import constants as C
from .engine import ORIENTATIONS, TetrisEngine
from .features import STATE_DIM, state_vector
from .script_agent_3d import _bump, _heightmap, _layer_counts, script_action_3d


def action_features(board, hs, holes, counts, lowest_y, ptype, x, z, ori_slot,
                    hold_after_type):
    """计算一个落点动作的特征向量。"""
    cells = ORIENTATIONS.get(ptype, [])
    if ori_slot >= len(cells):
        return None
    cells = cells[ori_slot]
    max_dy = max(c[1] for c in cells)

    # 边界
    for dx, dy, dz in cells:
        if not (0 <= x + dx < C.W and 0 <= z + dz < C.D):
            return None

    # 落点层
    y = 0
    for dx, dy, dz in cells:
        idx = (x + dx) * C.D + (z + dz)
        need = hs[idx] - dy
        if need > y:
            y = need
    if y >= C.H or y + max_dy >= C.H:
        return None

    fill_match = 0
    if y == lowest_y:
        for dx, dy, dz in cells:
            if dy == 0 and not board[x + dx][y][z + dz]:
                fill_match += 1

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

    # 层填充情况
    new_count_layer = 0
    # 简化：计算放置层 y 的填充数
    for xx in range(C.W):
        for zz in range(C.D):
            if board[xx][y][zz]:
                new_count_layer += 1
    for dx, dy, dz in cells:
        if dy == 0:
            if not board[x + dx][y][z + dz]:
                new_count_layer += 1
    left = C.W * C.D - new_count_layer
    cleared = 1 if new_count_layer == C.W * C.D else 0

    return np.array([
        x / (C.W - 1), z / (C.D - 1), ori_slot / max(1, C.MAX_ORI - 1),
        1.0 if hold_after_type != ptype else 0.0,
        y / C.H, fill_match / 4.0, maxh / C.H, avg / C.H,
        new_holes / (C.W * C.D), bump / (C.W * C.D * 4.0),
        left / (C.W * C.D), float(cleared),
    ], dtype=np.float32)


def collect_score_data(n_games, start_seed, save_dir, top_k=20, neg_k=20):
    os.makedirs(save_dir, exist_ok=True)
    states = []
    act_feats = []
    scores = []

    t0 = time.time()
    total_steps = 0
    for g in range(n_games):
        env = TetrisEngine(seed=start_seed + g)
        env.reset()
        steps = 0
        while not env.done and steps < 800:
            mask = env.legal_action_mask(horizontal_only=False)
            idxs = [i for i, m in enumerate(mask) if m]
            if not idxs:
                break
            board = env.board
            hs, holes = _heightmap(board)
            counts = _layer_counts(board)
            lowest_y = C.H
            for yy in range(C.H):
                if counts[yy] < C.W * C.D:
                    lowest_y = yy
                    break
            ptype = env.piece["type"]
            state = state_vector(env)

            # 计算每个合法动作的特征和评分
            entries = []
            for a in idxs:
                x, z, ori_slot, hold = env.decode_action(a)
                pt = ptype
                if hold:
                    pt = env.hold if env.hold is not None else (env.queue[0] if env.queue else None)
                    if pt is None:
                        continue
                if ori_slot >= len(ORIENTATIONS.get(pt, [])):
                    continue
                feat = action_features(board, hs, holes, counts, lowest_y, pt, x, z, ori_slot, pt)
                if feat is None:
                    continue
                # 内联计算评分（与脚本策略一致）
                score = _score_action(board, hs, holes, counts, lowest_y, pt, x, z, ori_slot)
                entries.append((score, feat, a))

            if not entries:
                break
            entries.sort(key=lambda e: e[0], reverse=True)
            chosen = entries[:top_k]
            # 负样本：从后面随机取
            rest = entries[top_k:]
            if rest:
                if len(rest) <= neg_k:
                    chosen_neg = rest
                else:
                    idx_neg = np.random.choice(len(rest), neg_k, replace=False)
                    chosen_neg = [rest[i] for i in idx_neg]
            else:
                chosen_neg = []
            for score, feat, a in chosen + chosen_neg:
                states.append(state)
                act_feats.append(feat)
                scores.append(score)

            a = entries[0][2]
            _, d, _ = env.step(a)
            steps += 1
            total_steps += 1
            if d:
                break

        if (g + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"  collected {g + 1}/{n_games} games, {len(scores)} entries, "
                  f"{len(scores) / max(dt, 1e-6):.0f} entries/s")

    states = np.asarray(states, dtype=np.float32)
    act_feats = np.asarray(act_feats, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    np.savez_compressed(os.path.join(save_dir, "score_dataset.npz"),
                        states=states, act_feats=act_feats, scores=scores)
    print(f"dataset saved: states={states.shape} act_feats={act_feats.shape} scores={scores.shape}")


def _score_action(board, hs, holes, counts, lowest_y, ptype, x, z, ori_slot):
    cells = ORIENTATIONS.get(ptype, [])
    if ori_slot >= len(cells):
        return -1e18
    cells = cells[ori_slot]
    max_dy = max(c[1] for c in cells)
    for dx, dy, dz in cells:
        if not (0 <= x + dx < C.W and 0 <= z + dz < C.D):
            return -1e18
    y = 0
    for dx, dy, dz in cells:
        idx = (x + dx) * C.D + (z + dz)
        need = hs[idx] - dy
        if need > y:
            y = need
    if y >= C.H or y + max_dy >= C.H:
        return -1e18
    fill_match = 0
    if y == lowest_y:
        for dx, dy, dz in cells:
            if dy == 0 and not board[x + dx][y][z + dz]:
                fill_match += 1
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
    return (20.0 * fill_match - 1.0 * maxh - 0.2 * avg
            - 0.3 * new_holes - 0.05 * bump)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", type=str, default=os.path.dirname(__file__))
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--neg-k", type=int, default=20)
    args = p.parse_args()
    collect_score_data(args.games, args.seed, args.save_dir, args.top_k, args.neg_k)


if __name__ == "__main__":
    main()
