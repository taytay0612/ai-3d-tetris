"""全局常量：与 3d-tetris.html 中的游戏规则保持一致。"""
from __future__ import annotations

# 场地
W = 7      # X 宽度（左右）
D = 7      # Z 深度（前后）
H = 14     # Y 高度（向上），0 为底部，H-1 为可见顶层，H 为隐藏出生层

# 计时参数（仅真实网页执行时参考；训练模拟器直接使用离散 step）
LOCK_MS = 500
SOFT_MS = 36
DAS_MS = 160
ARR_MS = 38
ROT_MS = 90
MAX_RESETS = 15

# 方块定义（与 JS 完全一致；初始均为水平放置，dy=0）
SHAPES = {
    "I": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
    "O": [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]],
    "T": [[1, 0, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1]],
    "S": [[1, 0, 0], [2, 0, 0], [0, 0, 1], [1, 0, 1]],
    "Z": [[0, 0, 0], [1, 0, 0], [1, 0, 1], [2, 0, 1]],
    "J": [[0, 0, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1]],
    "L": [[2, 0, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1]],
    "1": [[0, 0, 0]],
}

# 颜色（仅用于显示/调试）
PIECE_COLORS = {
    "I": {"base": "#22d3ee", "r": 34, "g": 211, "b": 238},
    "O": {"base": "#facc15", "r": 250, "g": 204, "b": 21},
    "T": {"base": "#c084fc", "r": 192, "g": 132, "b": 252},
    "S": {"base": "#4ade80", "r": 74, "g": 222, "b": 128},
    "Z": {"base": "#f87171", "r": 248, "g": 113, "b": 113},
    "J": {"base": "#60a5fa", "r": 96, "g": 165, "b": 250},
    "L": {"base": "#fb923c", "r": 251, "g": 146, "b": 60},
    "1": {"base": "#e2e8f0", "r": 226, "g": 232, "b": 240},
}

PIECE_TYPES = ["I", "O", "T", "S", "Z", "J", "L", "1"]
BAG = PIECE_TYPES.copy()          # 7-bag + 单方块，每袋 8 个
QUEUE_INITIAL = 6                 # JS 中保持队列至少 6 个

# 消除得分表（下标 = 同时消除平面数）
PLANE_PTS = [0, 100, 300, 500, 800]

# 操作映射（低层按键 -> 移动/旋转）
# 俯视视角：A(+X 屏幕左)、D(-X 屏幕右)、W(-Z 屏幕上/前)、S(+Z 屏幕下/后)
MOVE_DIR = {
    "left": (1, 0),     # A
    "right": (-1, 0),   # D
    "forward": (0, -1), # W
    "back": (0, 1),     # S
}
KEY_FOR_DIR = {"left": "a", "right": "d", "forward": "w", "back": "s"}
ROT_OPS = ["rotYCCW", "rotYCW", "rotZCCW", "rotZCW"]
KEY_FOR_ROT = {
    "rotYCCW": "ArrowUp",
    "rotYCW": "ArrowDown",
    "rotZCCW": "ArrowLeft",
    "rotZCW": "ArrowRight",
}
KEY_HARD_DROP = " "
KEY_HOLD = "c"
KEY_PAUSE = "p"
KEY_RESTART = "r"
KEY_SOFT_DROP = "x"
KEY_MUTE = "m"

# 动作空间（目标落点）：x(7) * z(7) * orientation_slot(24) * hold(2)
MAX_ORI = 24
ACTION_SIZE = W * D * MAX_ORI * 2
