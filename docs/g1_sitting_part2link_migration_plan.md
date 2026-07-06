# G1 Sitting Part2Link -> InstinctMJ 迁移计划

更新日期：2026-07-06

## 1. 背景与目标

将 InstinctLab 中的 `g1_interaction_sitting_part2link_shadowing_cfg` 任务迁移到 InstinctMJ，使用 mjlab-native layout 实现 sitting Part2Link interaction task。

目标仓库：

```text
/home/yangke/KY/InstinctMJ
```

目标落点：

```text
src/instinct_mj/tasks/interaction/
```

迁移原则：

- 不修改 `mjlab` 本体。
- 不修改 InstinctLab 源仓库。
- 不引入 IsaacLab compatibility layer。
- Motion joint order 使用 InstinctMJ G1 MJCF native joint order。
- Reward 的目标 robot body / object part 不再通过显式 `link_names` 配置传入，而是从 NPZ metadata 读取。
- NPZ 是 Part2Link reward 的 authority；`sofa.json` 只做语义/几何校验和 warning。

## 2. 源任务核心能力

源任务围绕 sitting interaction shadowing 做了以下扩展：

- Sitting Part2Link task config。
- ObjectMotion sparse contact 加载。
- sofa/chair alpha asset variant 加载。
- Part2Link vector tracking reward。
- Forbidden contact penalty。
- Seat contact reward。
- tracking sigma curriculum。
- object alpha staged curriculum/training。
- Transformer runner/policy config。
- alpha staged training script。

核心 reward 是：

```text
part2link_vector_guidance_gauss
```

语义：

- 对 pelvis、torso、左右 wrist 等 sparse robot bodies 到 object part center 的 current vector 进行 tracking。
- GT vector 从 motion sparse metadata 读取。
- Gaussian tracking 参数保持源任务设置：
  - `tracking_sigma = 0.25`
  - `tracking_tolerance = 0.08`
  - `precision_scale_range = (0.8, 1.4)`
  - reward weight `1.0`

## 3. 目标任务注册

新增 interaction task package，并注册四个 task：

```text
Instinct-Interaction-Sitting-Part2Link-G1-v0
Instinct-Interaction-Sitting-Part2Link-G1-Play-v0
Instinct-Interaction-Sitting-Part2Link-Transformer-G1-v0
Instinct-Interaction-Sitting-Part2Link-Transformer-G1-Play-v0
```

相关文件：

```text
src/instinct_mj/tasks/interaction/__init__.py
src/instinct_mj/tasks/interaction/config/__init__.py
src/instinct_mj/tasks/interaction/config/g1/__init__.py
src/instinct_mj/tasks/__init__.py
```

## 4. Motion 加载设计

新增 Part2Link motion 数据类型：

```text
src/instinct_mj/motion_reference/motion_reference_part2link_data.py
src/instinct_mj/motion_reference/motion_files/part2link_motion_cfg.py
src/instinct_mj/motion_reference/motion_files/part2link_motion.py
```

基于现有 `HoiMotionReferenceData` 扩展 sparse contact 字段：

- `sparse_contact_link_part_center_vector_w`
- `sparse_contact_relation_matrix`
- `sparse_contact_link_part_proximity`
- `sparse_contact_link_part_point_proximity`
- `sparse_contact_part_points_local`
- `sparse_contact_part_point_mask`
- `sparse_contact_part_centers_local`
- `sparse_contact_robot_body_names`
- `sparse_contact_robot_interest_names`
- `sparse_contact_part_names`

加载要求：

- NPZ 必须包含 `sparse_contact_robot_body_names`。
- 缺失 metadata 时直接报错，不 fallback 到硬编码 body list。
- `joint_pos` 按 InstinctMJ G1 MJCF native joint order 重排。
- Object pose 从 NPZ 的 `box_pos` / `box_quat` 读取。
- Sparse tensors 按 motion reference frame 填入 `Part2LinkMotionReferenceData`。

样例 NPZ 已确认：

```text
sparse bodies:
  pelvis_contour_link
  torso_link
  left_rubber_hand_link
  right_rubber_hand_link

sparse parts:
  seat
  armrest_left
  armrest_right
  back

sparse_contact_link_part_center_vector_w shape:
  (743, 4, 4, 3)
```

目标 G1 MJCF native joint order 已确认是 29 dof，且样例 NPZ 原始 joint order 与 native order 不一致，因此 reorder 是必要步骤。

## 5. Asset 加载设计

默认 dataset root：

```text
/home/yangke/KY/InstinctLab_interact/datasets/interaction/output_npz_29dof_with_object
```

外部 chair asset 来源：

```text
sofa_exntend_obj/<chair>/ffd_bbox_coarse/morph_path/alpha_*.glb
```

新增 asset prepare/cache 脚本：

```text
src/instinct_mj/scripts/prepare_part2link_assets.py
```

新增 console script：

```text
instinct-prepare-part2link-assets
```

策略：

- 大型 mesh/cache 不进 git。
- 支持从 GLB 转成 MuJoCo 可加载 OBJ cache。
- Runtime 优先使用 `INSTINCT_PART2LINK_ASSET_CACHE` 中的缓存 mesh。
- 若没有 cache，则尝试直接使用 dataset 中的 GLB。

默认 single-alpha staged training：

- 每个训练进程只加载一个 alpha。
- 默认 alpha 从 `SITTING_PART2LINK_ALPHA_VALUES` 读取。
- 若传入多个 alpha，当前 config 只取最后一个作为本进程 stage alpha。
- 每个 stage 加载 20 个 chair variants。

默认 chair list：

```text
chair_14, chair_15, chair_17, chair_18, chair_20,
chair_22, chair_28, chair_30, chair_32, chair_33,
chair_37, chair_39, chair_40, chair_41, chair_43,
chair_44, chair_46, chair_48, chair_51, chair_55
```

## 6. Runtime Object Variant State

新增 mjlab-native object variant runtime state：

- active chair id
- active alpha
- active object scale
- active precision scale
- active part centers
- active part points
- active point masks

Reset 行为：

- 每个 env 从当前加载的 chair variants 中选择一个 active chair。
- 从 motion reference initial object pose 写入 active chair entity。
- 其余 chair entity 移到隐藏位置。
- 采样 scale 和 precision scale。

Step/update 行为：

- 从 motion reference 当前 frame 更新 active chair pose。
- 保持 inactive chairs 隐藏。

相关实现：

```text
src/instinct_mj/tasks/interaction/mdp/part2link.py
```

## 7. Reward 设计

### 7.1 Part2Link Vector Guidance

Reward：

```text
part2link_vector_guidance_gauss
```

Config params 保留：

```text
object_name
metadata_root
tracking_sigma
tracking_tolerance
precision_scale_range
debug_vis
```

删除：

```text
link_names
```

行为：

- 从 NPZ metadata 读取 sparse robot body names 和 part names。
- 根据 metadata relation matrix 选择 should-contact pairs。
- 当前 robot link position 与 active object part center 计算 current vector。
- 与 NPZ sparse GT vector 做 Gaussian tracking。
- `sofa.json` 只做 alias/order/relation 校验并 warning，不作为 reward authority。

Target XML body alias：

```text
left_rubber_hand_link  -> left_rubber_hand
right_rubber_hand_link -> right_rubber_hand
```

这是 task-local metadata alias，不是 reward 参数中的硬编码 target list。

### 7.2 Forbidden Contact Penalty

Reward：

```text
part2link_forbidden_contact_penalty
```

行为：

- 从 NPZ relation matrix 中读取 forbidden pairs，relation 值为 `-1`。
- 使用 active part points 和 current robot body positions 计算距离。
- forbidden pair 距离低于阈值时产生 penalty。

保留源权重：

```text
weight = -1.0
```

### 7.3 Seat Contact Reward

Reward：

```text
object_contact_reference_phase
```

行为：

- 使用 active seat part points 计算 pelvis/hip 与 seat 的接触距离。
- 参考相位从 NPZ sparse relation/proximity 推出：
  - part 为 `seat`
  - relation 为 `1`
  - sparse proximity active

保留源权重：

```text
weight = 3.0
```

## 8. Env / RL Config

新增 G1 env config：

```text
src/instinct_mj/tasks/interaction/config/g1/g1_interaction_sitting_part2link_shadowing_cfg.py
```

新增 Transformer env config：

```text
src/instinct_mj/tasks/interaction/config/g1/g1_interaction_sitting_part2link_transformer_shadowing_cfg.py
```

新增 RL cfg：

```text
src/instinct_mj/tasks/interaction/config/g1/rl_cfgs.py
```

RL config 包含：

- depth encoder policy
- Part2Link Transformer policy
- motion-ref encoder 复刻源 Transformer 设置
- experiment name：

```text
g1_interaction_part2link_transformer
```

## 9. Public Interfaces

环境变量：

```text
INSTINCT_PART2LINK_DATASET_ROOT
INSTINCT_PART2LINK_ASSET_CACHE
SITTING_PART2LINK_ALPHA_VALUES
SITTING_PART2LINK_CHAIR_NAMES
```

新增 console scripts：

```text
instinct-prepare-part2link-assets
instinct-train-part2link-alpha-stages
```

Alpha staged training 默认顺序：

```text
1.0, 0.8, 0.5, 0.2, 0.0
```

训练脚本行为：

- 每个 alpha stage 设置 `SITTING_PART2LINK_ALPHA_VALUES=<alpha>`。
- 调用 InstinctMJ 的 train module。
- stage 之间自动尝试 resume 最新 run。

## 10. Test Plan

Unit tests：

- 加载样例 `sofa/replay_contact_map.g1_retargeted.npz`。
- 验证 sparse tensor shapes。
- 验证 body names / part names / relation/order。
- 验证 joint reorder 使用目标 MJCF native joint order。
- 验证 `sofa.json` 与 NPZ alias 校验只 warning，不阻断。
- fake tensor reward 测试：
  - current vector 等于 GT 时 reward 接近 1。
  - forbidden pair 距离低于阈值时产生 penalty。

Smoke tests：

```bash
uv run --python 3.11 pytest tests/test_part2link_metadata_and_rewards.py -q
uv run --python 3.11 instinct-list-envs Part2Link
uv run --python 3.11 instinct-train Instinct-Interaction-Sitting-Part2Link-Transformer-G1-v0 --num-envs 1 --agent.max_iterations 0 --viewer none
```

Visual checks：

- Play cfg/native viewer 检查 active chair。
- 检查 depth image。
- 检查 object pose reset/update。
- 检查 Part2Link debug markers。

## 11. 当前进度

已完成：

- interaction package 和 task registration。
- Part2Link motion loader/data。
- sparse contact metadata loading。
- MJCF native joint reorder。
- object variant runtime。
- Part2Link rewards。
- G1 env config。
- Transformer env/RL config。
- asset prepare script。
- alpha staged training script。
- unit test 文件。

已验证：

- 所有新增/修改 Python 文件 `py_compile` 通过。
- 样例 NPZ sparse metadata 可读。
- G1 MJCF native joint order 可读。
- 样例 NPZ 原始 joint order 与目标 native order 不一致，确认 reorder 必要。
- `git diff --check` 通过。

未完成：

- 尚未在完整 InstinctMJ runtime 环境跑通 `instinct-list-envs`。
- 尚未跑通 train smoke。
- 尚未做 native viewer visual check。

当前阻塞：

- 当前机器 `uv` 默认选择 `/usr/bin/python3` 3.10。
- `onnxruntime==1.24.2` 只提供 cp311/cp312/cp313 wheel，导致 `uv run --frozen ...` 无法构建环境。
- 系统 `python3` 缺 `torch`。
- `InstinctLab` conda env 有 torch，但缺 `mjlab` 和 `pygments`，不能作为 InstinctMJ runtime env。

## 12. 下一个 Agent 建议

优先准备 Python 3.11+ 的 InstinctMJ runtime 环境，然后按顺序跑：

```bash
cd /home/yangke/KY/InstinctMJ
uv run --python 3.11 pytest tests/test_part2link_metadata_and_rewards.py -q
uv run --python 3.11 instinct-list-envs Part2Link
uv run --python 3.11 instinct-train Instinct-Interaction-Sitting-Part2Link-Transformer-G1-v0 --num-envs 1 --agent.max_iterations 0 --viewer none
```

如果 asset 加载失败，先准备 cache：

```bash
uv run --python 3.11 instinct-prepare-part2link-assets --alphas 1.0
export INSTINCT_PART2LINK_ASSET_CACHE=<printed-cache-path>
```

Runtime 重点检查：

- `EntityCfg(spec_fn=...)` mesh/object entity 是否符合当前 mjlab API。
- `MotionReferenceManager` 是否正确 fill Part2Link sparse tensors。
- active chair hide/show 是否正确。
- object pose reset/update 是否正确。
- reward metadata body alias 是否正确。
- `sofa.json` part alias mismatch 是否只 warning。
- Transformer policy input component names 是否与 observation group 完全匹配。

