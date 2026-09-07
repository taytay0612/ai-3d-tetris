# AI 3D Tetris

训练 AI 玩 3D 俄罗斯方块（`E:\deepseek\test\3d-tetris.html`）的项目。  
成功模型使用**神经网络评分函数**：对当前所有合法落点打分，选最高分执行。

## 成功标准
- 模拟器 20 局平均消除平面：**42 平面** ✅（标准为 ≥20）

---

## 文件夹结构

```
ai_tetris_success/
├── ai_tetris/                     # Python 包
│   ├── constants.py               # 游戏常量、方块、旋转、按键
│   ├── engine.py                  # 与网页规则一致的模拟器
│   ├── features.py                # 状态向量、高度图、空洞、势函数
│   ├── script_agent_3d.py         # 3D 脚本策略（生成训练示范）
│   ├── score_data.py              # 生成评分回归数据集
│   ├── score_net.py               # 评分网络结构
│   ├── score_train.py             # 训练评分网络
│   ├── score_eval.py              # 模拟器评估评分网络
│   ├── score_play_browser.py      # 浏览器自动玩（神经网络）
│   ├── browser_ctl.py             # Edge CDP 控制（启动/读状态/发按键）
│   └── planner.py                 # 目标落点 → 按键序列路径规划
├── score_net.pt                   # 训练好的神经网络模型（1.04MB）
├── 启动调试Edge.bat               # 打开带调试端口的游戏网页
├── 启动神经网络演示.bat           # 让神经网络 AI 自动玩
└── README.md
```

---

## 运行演示（最简单）

1. 双击 `启动调试Edge.bat`  
   会打开游戏网页，并让 Edge 监听 `9222` 调试端口。

2. 双击 `启动神经网络演示.bat`  
   AI 开始自动玩。日志写在 `E:\deepseek\test\ai_tetris\score_play_log.txt`。

## 命令行运行

```powershell
cd E:\deepseek\test\ai_tetris_success
$env:PYTHONPATH='E:\deepseek\test\ai_tetris_success'
python -m ai_tetris.score_play_browser --model E:\deepseek\test\ai_tetris_success\score_net.pt --port 9222 --episodes 3
```

---

## 重新训练

### 1. 生成评分数据集
```powershell
cd E:\deepseek\test\ai_tetris_success
$env:PYTHONPATH='E:\deepseek\test\ai_tetris_success'
python -m ai_tetris.score_data --games 100 --save-dir E:\deepseek\test\ai_tetris_success --top-k 10 --neg-k 10
```

### 2. 训练评分网络
```powershell
python -m ai_tetris.score_train --dataset E:\deepseek\test\ai_tetris_success\score_dataset.npz --epochs 30 --batch-size 4096 --hidden 256 --lr 0.001 --save-path E:\deepseek\test\ai_tetris_success\score_net.pt
```

### 3. 模拟器评估
```powershell
python -m ai_tetris.score_eval --model E:\deepseek\test\ai_tetris_success\score_net.pt --episodes 20
```

---

## 依赖

- Python 3.14
- PyTorch（GPU 版或 CPU 版均可；GPU 更快）
- numpy

> 本机 GPU 训练需要 CUDA PyTorch。当前项目默认会尝试加载
> `E:\deepseek\test\ai_tetris\torch_cu128`（如果存在），
> 否则使用系统安装的 PyTorch。

---

## 其他

- `E:\deepseek\test\ai_tetris_failed_delete_me\` 是训练过程中失败/旧版文件，可删除。
- 浏览器控制使用 Edge CDP，不需要 Playwright/Selenium。
