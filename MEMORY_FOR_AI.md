# MEMORY FOR AI

> 这是给下一个 AI 对话的精华记忆。新 AI 接手时先读本文件。

## 项目目标
训练 AI 玩 3D 俄罗斯方块，网页游戏为 `E:\deepseek\test\3d-tetris.html`。  
成功标准：模拟器平均消除平面 ≥20。

## 当前状态：已达标
- 成功模型：`score_net.pt`（神经网络评分函数）。
- 模拟器 20 局平均消除平面：**42**，最高 59。
- 浏览器演示可用：双击 `启动调试Edge.bat` 再双击 `启动神经网络演示.bat`。

## 成功模型原理
- 不使用 PPO/DQN 直接训练策略。
- 先写了一个 3D 脚本策略（`script_agent_3d.py`），它每步枚举所有合法落点，
  用“最低未满层缺口形状匹配 + 高度/空洞/平整度”评分，选最高分。
  脚本策略模拟器平均 50 平面。
- 然后用脚本策略生成 `(状态, 动作特征, 评分)` 数据，
  训练神经网络 `ScoreNet` 回归该评分。
- 测试时每步枚举合法落点，网络打分，选最高分，再路径规划成按键执行。

## 为什么评分回归成功，而 BC 分类/PPO/DQN 失败
- BC 分类：训练准确率即使 93%，整局 100+ 步，错误累积后表现接近 0。
- PPO 微调 BC：会破坏 BC 策略，平均平面回到 0。
- DQN 微调 BC：环境统计有 bug，且稀疏奖励下未学会消层。
- 评分回归：保留了“每步搜索 + 排序选优”的结构，网络只需拟合评分，
  即使分数有误差，只要排序大致正确就能工作。

## 文件作用（成功项目）
见 README.md。核心：
- `engine.py`：模拟器，与网页 JS 规则一致。
- `script_agent_3d.py`：3D 脚本策略，用于生成训练数据。
- `score_data.py`：生成评分回归数据集。
- `score_net.py`：评分网络（5 层全连接，27 万参数）。
- `score_train.py`：训练评分网络。
- `score_eval.py`：模拟器评估。
- `score_play_browser.py`：浏览器自动玩。
- `browser_ctl.py`：Edge CDP 控制。
- `planner.py`：目标落点转按键序列。

## 运行命令
```powershell
cd E:\deepseek\test\ai_tetris_success
$env:PYTHONPATH='E:\deepseek\test\ai_tetris_success'
python -m ai_tetris.score_eval --model E:\deepseek\test\ai_tetris_success\score_net.pt --episodes 20
```

## 环境坑
- 本机 site-packages 中的 PyTorch 是 CPU 版。CUDA 版 torch 解压在
  `E:\deepseek\test\ai_tetris\torch_cu128`，`ai_tetris/__init__.py` 会自动优先加载。
- 当前工具 shell 权限受限，`multiprocessing.Pipe` 会 PermissionError；
  用户桌面 PowerShell 可以跑多进程。
- Edge CDP 调试端口：从工具 shell 直接启动 Edge 可能崩溃；
  用 `explorer.exe` 启动 bat 文件的方式可以成功打开调试端口。
- 浏览器演示时不要和训练同时跑，训练会占满 CPU 导致 Edge JS 读取超时。

## 未来方向
- 如果继续提升：可尝试 DQN 微调评分网络（学习率要极小，避免崩坏）。
- 或者把脚本策略的“最低层空缺形状”加入网络输入，减少对脚本评分的依赖。
- 真实游戏微调有风险：直接 RL 可能让模型退化回 0 平面。

## 可删除文件夹
`E:\deepseek\test\ai_tetris_failed_delete_me\` 是失败/旧文件，可删除。
