"""训练评分网络（MSE 回归）。"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .features import STATE_DIM
from .score_net import ScoreNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str,
                   default=os.path.join(os.path.dirname(__file__), "score_dataset.npz"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save-path", type=str,
                   default=os.path.join(os.path.dirname(__file__), "score_net.pt"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data = np.load(args.dataset)
    states = data["states"]
    act_feats = data["act_feats"]
    scores = data["scores"]
    n = int(states.shape[0])
    print(f"samples={n} state_dim={states.shape[1]} act_dim={act_feats.shape[1]}")

    model = ScoreNet(states.shape[1], act_feats.shape[1], hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    idx = np.arange(n)
    for epoch in range(args.epochs):
        np.random.shuffle(idx)
        epoch_loss = 0.0
        t0 = time.time()
        for start in range(0, n, args.batch_size):
            mb = idx[start:start + args.batch_size]
            x = torch.from_numpy(states[mb].astype(np.float32)).to(device)
            a = torch.from_numpy(act_feats[mb].astype(np.float32)).to(device)
            y = torch.from_numpy(scores[mb].astype(np.float32)).to(device)
            pred = model(x, a)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(mb)
        epoch_loss /= n
        print(f"epoch {epoch + 1}/{args.epochs} loss={epoch_loss:.4f} time={time.time() - t0:.1f}s")

    torch.save({
        "model_state": model.state_dict(),
        "state_dim": states.shape[1],
        "act_dim": act_feats.shape[1],
        "hidden": args.hidden,
    }, args.save_path)
    print(f"saved {args.save_path}")


if __name__ == "__main__":
    main()
