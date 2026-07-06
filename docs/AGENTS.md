# InstinctMJ Agent Guide

更新日期：2026-07-06

## 1. 项目概览

InstinctMJ 是 Project Instinct 在 mjlab 框架上的环境端实现，用于 Unitree G1 29-DOF 人形机器人的 RL 训练。当前活跃任务系列为 **Interaction / Sitting Part2Link**。

```text
仓库:   /home/yangke/KY/InstinctMJ
框架:   mjlab 1.4.0 (MuJoCo Warp GPU 仿真)
训练:   instinct_rl (PPO via OnPolicyRunner)
Python: >=3.11,<3.14
```

## 2. 环境变量

以下变量已写入 `.vscode/settings.json`，新终端自动加载：

| 变量 | 值 | 用途 |
|------|-----|------|
| `INSTINCT_PART2LINK_ASSET_CACHE` | `~/.cache/instinct_mj/part2link_assets` | 椅子 `.obj` 网格缓存（19 个 chair） |
| `INSTINCT_PART2LINK_DATASET_ROOT` | `~/KY/InstinctLab_interact/datasets/interaction/output_npz_29dof_with_object` | NPZ 运动数据集 |
| `MUJOCO_GL` | `egl` | 默认无头渲染 |

有 GUI 时覆盖：`export MUJOCO_GL=glfw`

## 3. 关键路径

| 路径 | 内容 |
|------|------|
| `src/instinct_mj/tasks/interaction/` | Sitting Part2Link 任务定义 |
| `src/instinct_mj/tasks/interaction/mdp/part2link.py` | Part2Link reward、event、curriculum、visualization |
| `src/instinct_mj/tasks/interaction/config/g1/` | G1 任务配置（env cfg + RL cfg） |
| `src/instinct_mj/motion_reference/` | 运动参考系统（Part2LinkMotion、HOI data） |
| `src/instinct_mj/envs/manager_based_rl_env.py` | `InstinctRlEnv` 主环境类 |
| `src/instinct_mj/scripts/instinct_rl/train.py` | 训练入口 |
| `src/instinct_mj/scripts/instinct_rl/play.py` | 推理回放入口 |
| `src/instinct_mj/assets/unitree_g1.py` | G1 机器人资产（29-DOF MJCF） |
| `/tmp/mjlab/` | **mjlab 1.4.0 框架源码** |
| `/home/yangke/KY/mjlab/` | → symlink to `/tmp/mjlab/` |
| `/home/yangke/.cache/instinct_mj/part2link_assets/` | GLB→OBJ 转换缓存 |
| `~/.local/bin/uv` | uv 包管理器 |

## 4. 常用命令

### 环境管理

```bash
cd /home/yangke/KY/InstinctMJ

# 同步依赖（修改 pyproject.toml 后）
uv lock --python 3.11 && uv sync --python 3.11

# 安装 dev 工具
uv pip install pytest --python 3.11
```

### 列出任务

```bash
.venv/bin/instinct-list-envs
```

### 训练

```bash
# 无头训练（生产模式）
.venv/bin/instinct-train \
  Instinct-Interaction-Sitting-Part2Link-G1-v0 \
  --num-envs 2048 --device cuda:0 --headless True

# Transformer 版本
.venv/bin/instinct-train \
  Instinct-Interaction-Sitting-Part2Link-Transformer-G1-v0 \
  --num-envs 2048 --device cuda:0 --headless True

# 带窗口调试（num_envs=1）
export MUJOCO_GL=glfw
.venv/bin/instinct-train \
  Instinct-Interaction-Sitting-Part2Link-G1-v0 \
  --num-envs 1 --viewer native

# 从 checkpoint 恢复
.venv/bin/instinct-train \
  Instinct-Interaction-Sitting-Part2Link-G1-v0 \
  --num-envs 2048 --headless True \
  --agent.resume True --agent.load_run 2026-07-06_16-28-05
```

### CLI 覆盖

```bash
--env.headless true          # 关闭 GUI debug 可视化
--env.scene.num_envs 2048    # 覆盖环境数
--agent.max_iterations 50000 # 覆盖迭代次数
--agent.algorithm.learning_rate 5.0e-4
--gpu-ids [0,1]              # 多 GPU
--gpu-ids all                # 全部 GPU
```

### 推理/回放

```bash
# 按 run 目录名加载
.venv/bin/instinct-play \
  Instinct-Interaction-Sitting-Part2Link-G1-Play-v0 \
  --load-run 2026-07-06_16-28-05 --viewer native

# 直接指定 checkpoint 文件
.venv/bin/instinct-play \
  Instinct-Interaction-Sitting-Part2Link-Transformer-G1-Play-v0 \
  --checkpoint-file logs/instinct_rl/g1_interaction_part2link/2026-07-06_16-28-05/model_700.pt \
  --viewer native
```

### 资产准备（首次或 GLB 更新后）

```bash
.venv/bin/instinct-prepare-part2link-assets --alphas 1.0
```

### 测试

```bash
uv run --python 3.11 python -m pytest tests/test_part2link_metadata_and_rewards.py -v
```

## 5. 已注册的 Part2Link 任务

| Task ID | 策略类型 |
|---------|---------|
| `Instinct-Interaction-Sitting-Part2Link-G1-v0` | MLP + Conv2d depth encoder |
| `Instinct-Interaction-Sitting-Part2Link-G1-Play-v0` | 同上（Play） |
| `Instinct-Interaction-Sitting-Part2Link-Transformer-G1-v0` | Transformer + Conv2d depth encoder |
| `Instinct-Interaction-Sitting-Part2Link-Transformer-G1-Play-v0` | 同上（Play） |

## 6. 架构要点

### 物体资产加载

椅子通过 **mocap body** 加载（无动力学）。所有 19 个变体在 Scene 编译时预加载，运行时通过显示/隐藏（mocap pose 控制）切换。**无法在训练中途动态添加新网格**。

### NPZ metadata authority

Part2Link 的 body/part/relation/order 全部从 NPZ 的 `sparse_contact_*` 字段读取。`sofa.json` 仅做 warning 校验，不阻断。

### 依赖版本约束

- `mjlab==1.4.0` → editable from `../mjlab` (symlink to `/tmp/mjlab`)
- `mujoco-warp~=3.9.0` (git rev `e65a72e`)
- `torch` / `torchvision` / `triton` → 均从 `pytorch-cu118` 索引解析（兼容驱动 550）
- `warp-lang>=1.14.0` (from nvidia index)

### 可视化系统

- Motion Reference markers：`MotionReferenceManager.debug_vis()` — links, relative_links, root frames
- Camera depth debug：`camera_cfg.debug_vis` + `depth_image.params.debug_vis`（由 `--env.headless` 控制）
- Part2Link markers：`Part2LinkDebugVisualizer` monitor term — 绘制 contact points (yellow)、part centers (orange)、realtime vectors (red)、GT vectors (blue)
- 所有 debug 通过 `env.update_visualizers(visualizer)` → `MonitorManager.debug_vis()` 管线驱动

## 7. 常见陷阱

1. **`--load-run` 是目录名不是完整路径**。Play 时 `--load-run` 只填 run 目录名（如 `2026-07-06_16-28-05`），它会在 `logs/instinct_rl/<experiment_name>/` 下匹配。想指定完整路径用 `--checkpoint-file`。

2. **Play 任务名必须匹配训练的实验名**。Transformer 任务的 `experiment_name=g1_interaction_part2link_transformer`，非 Transformer 是 `g1_interaction_part2link`。

3. **`uv sync` 会重建 venv**。改 `pyproject.toml` 后需要 `rm uv.lock && uv lock --python 3.11 && uv sync --python 3.11`。

4. **CUDA 驱动 ≤ 12.4**。系统驱动 550 最高支持 CUDA 12.4，因此 PyTorch 必须用 cu118 构建。

5. **MUJOCO_GL=egl 时不能开 viewer**。`--viewer native` 需要 `glfw` 和 DISPLAY。

6. **Part2Link 的 `apply_env_spacing=False`**。Motion reference 已在 fill 时加了 env_origins，event 不能再加一遍。
