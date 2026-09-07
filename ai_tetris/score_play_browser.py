"""神经网络评分模型在真实 Edge 中自动玩 3D 俄罗斯方块。"""
from __future__ import annotations

import time

import torch

from . import constants as C
from .browser_ctl import BrowserGameController
from .engine import ORIENTATIONS, TetrisEngine
from .planner import board_from_browser, plan_path
from .score_eval import choose_action


def _decode_action(action: int):
    hold = action & 1
    a = action >> 1
    ori_slot = a % C.MAX_ORI
    a //= C.MAX_ORI
    z = a % C.D
    x = a // C.D
    return x, z, ori_slot, hold


def make_tmp_engine(st):
    tmp = TetrisEngine()
    tmp.board = board_from_browser(st["board"])
    p = st.get("piece")
    tmp.piece = {
        "type": p["type"],
        "x": int(p["x"]),
        "y": int(p["y"]),
        "z": int(p["z"]),
        "cells": [list(c) for c in p["cells"]],
    } if p else None
    tmp.hold = st.get("hold")
    tmp.queue = list(st.get("queue") or [])
    tmp.state = {
        "status": "playing" if st.get("status") in ("playing", "paused") else "idle",
        "score": st.get("score", 0),
        "level": st.get("level", 1),
        "lines": st.get("lines", 0),
        "planes": st.get("planes", 0),
        "combo": st.get("combo", -1),
        "canHold": st.get("canHold", True),
    }
    tmp.done = st.get("status") == "over"
    return tmp


def safe_read_state(ctl, retries=3):
    for i in range(retries):
        try:
            return ctl.read_state()
        except Exception as e:
            print(f"read_state error ({i + 1}/{retries}): {e}")
            time.sleep(0.5)
    raise RuntimeError("read_state failed")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default=r"E:\deepseek\test\ai_tetris\score_net.pt")
    p.add_argument("--port", type=int, default=9222)
    p.add_argument("--episodes", type=int, default=3)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    from .score_net import ScoreNet
    model = ScoreNet(ckpt["state_dim"], ckpt["act_dim"], hidden=ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded {args.model}, device={device}")

    ctl = BrowserGameController(port=args.port)
    if not ctl.connect_existing():
        print("cannot connect to Edge debug port")
        return
    print("connected to Edge")

    try:
        for ep in range(args.episodes):
            ctl.start_game()
            time.sleep(0.3)
            steps = 0
            while steps < 1000:
                st = safe_read_state(ctl)
                if st.get("status") in ("idle", "over"):
                    if st.get("status") == "over":
                        print(f"episode {ep + 1}: score={st.get('score')} planes={st.get('planes')} steps={steps}")
                    break
                if st.get("status") == "paused":
                    ctl.resume_game()
                    time.sleep(0.05)
                    continue
                if st.get("status") != "playing" or not st.get("piece"):
                    time.sleep(0.05)
                    continue

                tmp = make_tmp_engine(st)
                action = choose_action(model, device, tmp)
                if action is None:
                    time.sleep(0.05)
                    continue

                x, z, ori_slot, hold = _decode_action(action)
                cur_st = st
                if hold:
                    ctl.press_key(C.KEY_HOLD, delay=0.05)
                    time.sleep(0.05)
                    cur_st = safe_read_state(ctl)
                    if not cur_st.get("piece"):
                        time.sleep(0.05)
                        continue

                piece = cur_st["piece"]
                ptype = piece["type"]
                if ori_slot >= len(ORIENTATIONS.get(ptype, [])):
                    time.sleep(0.05)
                    continue
                target_cells = ORIENTATIONS[ptype][ori_slot]
                board = board_from_browser(cur_st["board"])
                path = plan_path(
                    board, ptype,
                    int(piece["x"]), int(piece["z"]), int(piece["y"]), piece["cells"],
                    x, z, target_cells,
                )
                if path is None:
                    time.sleep(0.05)
                    continue

                if steps % 50 == 0:
                    print(f"ep {ep + 1} step {steps}: piece={ptype} action={x},{z},ori{ori_slot},hold{hold} path={path}")
                for op in path:
                    key = C.KEY_FOR_DIR.get(op) or C.KEY_FOR_ROT.get(op)
                    if key is None:
                        continue
                    ctl.press_key(key, delay=0.015)
                ctl.press_key(C.KEY_HARD_DROP, delay=0.05)
                time.sleep(0.12)
                steps += 1
            time.sleep(0.3)
    except Exception as e:
        print(f"script error: {e}")
    finally:
        try:
            st = safe_read_state(ctl)
            if st.get("status") == "paused":
                ctl.press_key(C.KEY_PAUSE, delay=0.05)
        except Exception:
            pass
        ctl.close()
        print("done")


if __name__ == "__main__":
    main()
