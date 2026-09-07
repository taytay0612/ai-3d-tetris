"""用评分网络在模拟器中评估：枚举合法动作，网络打分选最高。"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from . import constants as C
from .engine import ORIENTATIONS, TetrisEngine
from .features import state_vector
from .score_data import action_features, _heightmap, _layer_counts
from .score_net import ScoreNet


def choose_action(model, device, env):
    mask = env.legal_action_mask(horizontal_only=False)
    idxs = [i for i, m in enumerate(mask) if m]
    if not idxs:
        return None
    board = env.board
    hs, holes = _heightmap(board)
    counts = _layer_counts(board)
    lowest_y = C.H
    for yy in range(C.H):
        if counts[yy] < C.W * C.D:
            lowest_y = yy
            break
    ptype = env.piece["type"]
    feats = []
    valid_actions = []
    for a in idxs:
        x, z, ori_slot, hold = env.decode_action(a)
        pt = ptype
        if hold:
            pt = env.hold if env.hold is not None else (env.queue[0] if env.queue else None)
            if pt is None:
                continue
        feat = action_features(board, hs, holes, counts, lowest_y, pt, x, z, ori_slot, pt)
        if feat is None:
            continue
        feats.append(feat)
        valid_actions.append(a)
    if not valid_actions:
        return None
    X = torch.from_numpy(np.stack(feats)).to(device)
    S = torch.from_numpy(np.repeat(state_vector(env)[None, :], len(valid_actions), axis=0)).to(device)
    with torch.no_grad():
        preds = model(S, X)
    best_idx = int(torch.argmax(preds).item())
    return valid_actions[best_idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=100)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    model = ScoreNet(ckpt["state_dim"], ckpt["act_dim"], hidden=ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"device: {device}, model: {args.model}")

    scores = []
    planes = []
    t0 = time.time()
    for ep in range(args.episodes):
        env = TetrisEngine(seed=args.seed + ep)
        env.reset()
        steps = 0
        while not env.done and steps < 800:
            a = choose_action(model, device, env)
            if a is None:
                break
            _, d, _ = env.step(a)
            steps += 1
        scores.append(env.state["score"])
        planes.append(env.state["planes"])
        print(f"ep {ep + 1}: score={scores[-1]} planes={planes[-1]} steps={steps}")
    arr = np.array(scores)
    parr = np.array(planes)
    print(f"avg_score={arr.mean():.1f} avg_planes={parr.mean():.2f} "
          f"min_planes={parr.min()} max_planes={parr.max()} elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
