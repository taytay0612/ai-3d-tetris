"""AI 3D Tetris 训练包。"""
import os
import sys

# 优先使用项目本地 CUDA 版 PyTorch（如果存在），避免 site-packages 中的 CPU 版
_CUDA_TORCH = os.path.join(os.path.dirname(__file__), "torch_cu128")
if not os.path.isdir(_CUDA_TORCH):
    # 本机原项目路径兜底；其他机器会自动使用系统 PyTorch
    _CUDA_TORCH = r"E:\deepseek\test\ai_tetris\torch_cu128"
if os.path.isdir(_CUDA_TORCH):
    sys.path.insert(0, _CUDA_TORCH)
