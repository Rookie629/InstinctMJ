# Server Environment Setup Guide

> 目标机器 agent 参考文档。在干净 Linux x86_64 服务器上从零搭建 InstinctMJ 训练/推理环境。

## Prerequisites

| 项目 | 要求 |
|---|---|
| OS | Ubuntu 20.04+ / Debian 11+ (Linux x86_64) |
| Python | 3.10 ~ 3.13（推荐 3.11） |
| Git | 2.x |
| SSH | 已配置 GitHub SSH key |
| GPU | NVIDIA + CUDA 11.8+（训练需要） |
| Disk | ≥ 50 GB（含模型、数据、日志） |

```bash
# 安装 uv（如果没有）
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Directory Layout

```
<WORKSPACE>/                       # 例如 ~/KY 或 /workspace
├── mjlab/                         # 框架层（git clone, tag v1.4.0）
├── InstinctMJ/                    # 环境层（git clone）
├── instinct_rl/                   # 训练层（uv sync 自动拉取）
└── InstinctLab_interact/          # 数据层（symlink 或目录）
    └── datasets/interaction/
        └── output_npz_29dof_with_object/
            ├── sofa_exntend_obj/  # GLB chair variants
            ├── sparse_contact_maps/
            └── metadata.yaml
```

## Step 1 — Clone Repositories

```bash
WORKSPACE=<your-workspace-path>     # 设置你的工作区路径
mkdir -p $WORKSPACE && cd $WORKSPACE

# mjlab 框架（必须锁定 v1.4.0）
git clone git@github.com:mujocolab/mjlab.git
cd mjlab && git checkout v1.4.0 && cd ..

# InstinctMJ 环境
git clone git@github.com:Rookie629/InstinctMJ.git
cd InstinctMJ
git checkout <branch-name>          # 按需切换分支，例如 main / contact
cd ..
```

## Step 2 — Align mujoco-warp Revision

mjlab 1.4.0 和 InstinctMJ 的 `pyproject.toml` 对 `mujoco-warp` 使用了不同的 Git commit。必须对齐到 mjlab 的版本：

```bash
cd $WORKSPACE/InstinctMJ

# 提取 mjlab 的 mujoco-warp rev
MJLAB_WARP_REV=$(grep 'mujoco-warp.*rev' $WORKSPACE/mjlab/pyproject.toml \
    | head -1 | sed 's/.*rev = "//' | sed 's/"//')

echo "mjlab mujoco-warp rev: $MJLAB_WARP_REV"

# 更新 InstinctMJ pyproject.toml
sed -i "s|mujoco-warp = { git = \"https://github.com/google-deepmind/mujoco_warp\", rev = \"[^\"]*\" }|mujoco-warp = { git = \"https://github.com/google-deepmind/mujoco_warp\", rev = \"$MJLAB_WARP_REV\" }|" \
    pyproject.toml

# 将版本约束也从 ~=3.9.0 降为 ~=3.8.0（与 mjlab 一致）
sed -i 's|"mujoco-warp>=3.9.0.1,~=3.9.0"|"mujoco-warp>=3.8.0.3,~=3.8.0"|' pyproject.toml
```

## Step 3 — Install Dependencies

```bash
cd $WORKSPACE/InstinctMJ

# 删除旧的 lock & venv（如果存在）
rm -f uv.lock
rm -rf .venv

# 重新解析依赖 + 安装
uv lock --python 3.11
uv sync --python 3.11
```

## Step 4 — Data Setup

### Part2Link（interaction / sitting task）

```bash
cd $WORKSPACE/InstinctMJ

# 方式 A：symlink（推荐）
ln -s /path/to/InstinctLab_interact/datasets/interaction \
    data/datasets/interaction

# 方式 B：环境变量
export INSTINCT_PART2LINK_DATASET_ROOT="/path/to/datasets/interaction/output_npz_29dof_with_object"
```

当前工作区已生成 Part2Link 派生资产时，可直接使用仓库根目录的环境脚本：

```bash
cd $WORKSPACE/InstinctMJ
source part2link_env.sh
```

该脚本当前指向：

```bash
export INSTINCT_PART2LINK_DATASET_ROOT=/home/yangke/KY/InstinctLab_interact/datasets/interaction/output_npz_29dof_with_object
export INSTINCT_PART2LINK_ASSET_CACHE=/home/yangke/KY/InstinctLab_interact/datasets/interaction/output_npz_29dof_with_object/derived/part2link_assets
export INSTINCT_PART2LINK_COLLISION_CACHE=/home/yangke/KY/InstinctLab_interact/datasets/interaction/output_npz_29dof_with_object/derived/part2link_coacd_collisions
export SITTING_PART2LINK_ALPHA_VALUES=1.0,0.8,0.5,0.0
```

如需在 play/viewer 中直接检查真实 CoACD 碰撞体，可临时启用碰撞可视模式：

```bash
export INSTINCT_PART2LINK_COLLISION_VISUAL_MODE=collision
```

默认模式会显示完整 visual mesh，并把 CoACD hull 放在隐藏 collision group；MuJoCo viewer 的 `Convex Hull` 按钮可能显示 visual mesh 的粗凸包，不代表真实接触体。该开关只建议调试时使用，训练默认保持未设置。

当前 collision cache 已完整生成 18 个 chair。由于仍缺 `chair_51_alpha_0p00` 和 `chair_55` 的四档 alpha，先限制 chair 列表以避免 fail-fast：

```bash
export SITTING_PART2LINK_CHAIR_NAMES=chair_14,chair_15,chair_17,chair_18,chair_20,chair_22,chair_28,chair_30,chair_32,chair_33,chair_37,chair_39,chair_40,chair_41,chair_43,chair_44,chair_46,chair_48
```

数据目录预期结构：
```
output_npz_29dof_with_object/
├── sofa_exntend_obj/          # GLB chair mesh variants
│   └── chair_XX/ffd_bbox_coarse/morph_path/alpha_X.XX.glb
├── sparse_contact_maps/       # per-motion contact NPZ
├── metadata.yaml
├── chair/                     # motion NPZ
└── sofa/                      # motion NPZ
```

### Part2Link CoACD Collision Cache（可选，用于凸分解碰撞体）

```bash
export INSTINCT_PART2LINK_COLLISION_CACHE="/path/to/collision_cache"
```

目录结构：
```
collision_cache/
└── <variant_name>/
    ├── collision_000.obj
    ├── collision_001.obj
    └── ...
```

### HOI（perceptive HOI task）

6 个 OMOMO 物体 OBJ 文件：
```
~/Datasets/OMOMO/data/captured_objects/
├── floorlamp_cleaned_simplified.obj
├── largebox_cleaned_simplified.obj
├── whitechair_cleaned_simplified.obj
├── trashcan_cleaned_simplified.obj
├── smalltable_cleaned_simplified.obj
└── suitcase_cleaned_simplified.obj
```

路径定义在 `src/instinct_mj/tasks/shadowing/perceptive_hoi/mdp/object_variant_hoi.py`。

### HOI Multi-Variant（可选）

```bash
export INSTINCT_HOI_VARIANT_ROOT="/path/to/hoi_variant_meshes"
```

目录结构：
```
variant_root/
├── floorlamp/
│   ├── variant_0.obj
│   └── variant_1.obj
├── largebox/
│   └── ...
```

使用 `prepare_hoi_variants.py` 从 GLB 转换：
```bash
uv run python -m instinct_mj.scripts.prepare_hoi_variants \
    --variant-root /path/to/source \
    --asset-cache /path/to/cache
```

## Step 5 — Verify

```bash
cd $WORKSPACE/InstinctMJ

# 1. 列出所有注册 task
uv run instinct-list-envs

# 2. 验证 mjlab 导入
uv run python -c "import mjlab; import mujoco; \
    print(f'mjlab={mjlab.__file__}'); \
    print(f'mujoco={mujoco.__version__}')"

# 3. Smoke test — 创建 env（不训练，只验证场景编译通过）
uv run python -c "
from instinct_mj.tasks.registry import get_task_env_cfg
cfg = get_task_env_cfg('Instinct-Locomotion-Flat-G1-v0')
print('Locomotion env cfg OK')
"

# 4. Play test（需要图形环境或 headless render）
uv run instinct-play Instinct-Locomotion-Flat-G1-Play-v0 \
    --load-run <run_name> --num_envs 1
```

## Training Quick Start

```bash
# 基础 locomotion
uv run instinct-train Instinct-Locomotion-Flat-G1-v0

# Part2Link sitting interaction（需要 Step 4 数据）
uv run instinct-train Instinct-Interaction-Sitting-Part2Link-G1-v0

# Perceptive HOI shadowing（需要 Step 4 数据）
uv run instinct-train Instinct-Perceptive-HOI-Shadowing-G1-v0
```

## Common Issues

| 问题 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: No module named 'mjlab'` | mjlab 不在 `../mjlab` 或未 editable install | 检查目录结构，确保 mjlab/ 与 InstinctMJ/ 同级 |
| `No solution found ... mujoco-warp` | mujoco-warp Git rev 冲突 | 按 Step 2 对齐到 mjlab 的版本 |
| `onnxruntime ... no wheel for cp310` | Python 3.10 不支持 onnxruntime 1.24 | `uv sync --python 3.11` |
| `/tmp/mjlab` 消失 | 系统清理 /tmp | mjlab 必须放在永久路径，不能用 /tmp |
| Part2Link 场景编译失败 | 缺少数据 | 检查 `data/datasets/interaction` symlink 是否有效 |
| GPU OOM | 默认 num_envs=4096 | 减小 `num_envs` 或使用 `env_spacing` |

## Environment Variables Reference

| 变量 | 默认值 | 用途 |
|---|---|---|
| `INSTINCT_PART2LINK_DATASET_ROOT` | `data/datasets/interaction/output_npz_29dof_with_object` | Part2Link motion + mesh 数据 |
| `INSTINCT_PART2LINK_ASSET_CACHE` | `None` | Part2Link OBJ 缓存目录 |
| `INSTINCT_PART2LINK_COLLISION_CACHE` | `None` | CoACD 凸分解碰撞体缓存 |
| `INSTINCT_PART2LINK_COLLISION_VISUAL_MODE` | `mesh` | 设为 `collision` 时在 play/viewer 中直接显示 CoACD 碰撞体 |
| `SITTING_PART2LINK_CHAIR_NAMES` | 20 个 chair | 逗号分隔的 chair 名称过滤 |
| `SITTING_PART2LINK_ALPHA_VALUES` | `1.0,0.8,0.5,0.0` | 逗号分隔的 morph alpha 值 |
| `INSTINCT_HOI_VARIANT_ROOT` | `None` | HOI multi-variant mesh 目录 |
| `INSTINCT_HOI_VARIANT_ASSET_CACHE` | `~/.cache/instinct_mj/hoi_assets` | HOI OBJ 缓存目录 |

## File Inventory (for agent reference)

| 文件 | 作用 |
|---|---|
| `pyproject.toml` | 项目依赖声明 |
| `uv.lock` | 锁定依赖版本 |
| `src/instinct_mj/tasks/registry.py` | Task 注册中心 |
| `src/instinct_mj/envs/mdp/events/object_variant.py` | 通用 variant 管理（共享） |
| `src/instinct_mj/envs/mdp/events/randomization.py` | Robot + Object DR events |
| `src/instinct_mj/envs/mdp/curriculums/object_difficulty.py` | Object curriculum |
| `src/instinct_mj/tasks/interaction/mdp/part2link.py` | Part2Link reward / variant 逻辑 |
| `src/instinct_mj/tasks/interaction/config/g1/g1_interaction_sitting_part2link_shadowing_cfg.py` | Part2Link G1 config |
| `src/instinct_mj/tasks/shadowing/perceptive_hoi/mdp/object_variant_hoi.py` | HOI variant catalog builder |
| `src/instinct_mj/tasks/shadowing/perceptive_hoi/config/g1/perceptive_shadowing_cfg.py` | HOI G1 config |
| `src/instinct_mj/scripts/prepare_hoi_variants.py` | HOI asset 准备脚本 |
| `tools/view_mesh_collision.py` | Mesh 碰撞可视化工具 |
