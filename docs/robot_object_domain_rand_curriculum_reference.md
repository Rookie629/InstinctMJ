# 机器人-物体 Domain Randomization 与 Curriculum 参考

更新日期：2026-07-07

本文归纳 InstinctLab interaction sitting Part2Link 中“多物体资产 + alpha 形变变体”作为 object domain randomization 和 curriculum 目标的当前实现、实际效果、可复用方法，以及迁移到 InstinctMJ 时需要保留的边界条件。

## 1. 目标

当前目标不是只训练机器人跟踪单条 motion，而是让策略在不同椅子几何、不同形变程度和不同 reward 精度下学会稳定的 sitting interaction：

- 多个 chair asset：改变物体整体几何、靠背/扶手/座面位置和接触点布局。
- 多个 alpha variant：同一 chair 沿 FFD morph path 的离散几何变体。
- active scale：改变真实物体尺寸。
- precision scale：不改变物体，只改变 Part2Link reward 的容差和 sigma。
- alpha curriculum：训练早期使用接近参考/容易的几何，随后逐步切到更难或更远的形变。

## 2. 数据和资产组织

InstinctLab 当前资产根目录：

```text
datasets/interaction/output_npz_29dof_with_object/sofa_exntend_obj
```

每个 chair 目录包含：

```text
chair_xx/
├── ffd_bbox_coarse/morph_path/
│   ├── alpha_0p00.usd
│   ├── alpha_0p10.usd
│   ├── ...
│   └── alpha_1p00.usd
└── contact_point_transfer/
    └── contact_points_local.npz
```

训练配置当前选用 20 个 chair：

```text
chair_14, chair_15, chair_17, chair_18, chair_20,
chair_22, chair_28, chair_30, chair_32, chair_33,
chair_37, chair_39, chair_40, chair_41, chair_43,
chair_44, chair_46, chair_48, chair_51, chair_55
```

默认 alpha 子集：

```text
0.0, 0.2, 0.5, 0.8, 1.0
```

因此 collection 模式一次场景构建会导入：

```text
20 chairs × 5 alpha variants = 100 object variants
```

InstinctMJ 侧当前 AGENTS.md 记录的是 19 个 chair OBJ 缓存。若要复现实验，应先确认 MJ 侧实际 asset cache 是否与这里的 20 chair 列表一致。

## 3. 当前实现结构

核心实现文件：

```text
source/instinctlab/instinctlab/tasks/interaction/config/g1/
  g1_interaction_sitting_part2link_shadowing_cfg.py

source/instinctlab/instinctlab/tasks/interaction/mdp/
  object_variant_guidance.py

scripts/instinct_rl/
  train_part2link_alpha_stages.py
```

### 3.1 Catalog

`load_object_variant_catalog()` 扫描 chair 目录，收集：

- `ObjectVariant`: chair name、alpha、USD path、contact point path、alpha index。
- `centers_local`: 每个 variant 的 part center。
- `points_local`: 每个 variant 的 part contact points。
- `point_valid_mask`: 接触点有效 mask。

这些数据被缓存到 `_CATALOG_CACHE`，运行时通过 `ObjectVariantRuntimeState` 记录每个 env 当前激活的 variant、scale、precision scale 和接触几何。

### 3.2 两种 object mode

当前配置由环境变量控制：

```text
SITTING_PART2LINK_OBJECT_MODE=collection | single
SITTING_PART2LINK_ALPHA_VALUES=0.0,0.2,0.5,0.8,1.0
```

`collection` 模式：

- 用 `RigidObjectCollectionCfg` 预加载所有 chair × alpha variant。
- 每次 reset 随机选 chair，再按当前 curriculum alpha 选择最近的 alpha variant。
- 激活 variant 放到参考 object pose，其余 variant 移到 inactive 位置。
- 这是真正支持“训练中 alpha curriculum 改变几何”的模式。

`single` 模式：

- 用 `MultiUsdFileCfg` 在 scene startup 时为每个 env 随机 spawn 一个 object。
- 只加载一个固定 alpha stage 的资产，例如 `alpha=1.0`。
- reset 时保留 `active_variant_ids`，不会重新切换 USD。
- 适合 staged training，不能在单个 run 内实现几何随 alpha curriculum 连续变化。

## 4. Domain Randomization 方式

### 4.1 Chair 随机化

collection 模式在 reset 时执行：

```text
chair_ids = randint(unique_chairs)
variant_ids = select_nearest_alpha(chair_ids, current_alpha)
```

效果：

- 同一 batch 内不同 env 会看到不同 chair。
- 同一个 env 每次 reset 也可能换 chair。
- contact filter 会覆盖所有 variant prim，保证 pelvis/hip/knee/ankle 等 contact sensor 能匹配当前 active object。

### 4.2 Alpha 变体随机化/课程化

`ObjectAlphaCurriculum` 维护：

```text
initial_alpha = 1.0
final_alpha   = 0.0
start_step    = 0
end_step      = 3_000_000
```

每步根据 `env.common_step_counter` 线性更新：

```text
env._interaction_object_current_alpha
```

collection 模式下，reset 时会读取这个值，并从离散 alpha 档位中选最近的 variant。因为资产是离散的，实际几何变化是分段的，不是连续 morph。

single 模式下，该变量虽然会更新，但不会改变已经 spawn 的 USD。这个是历史实验中最重要的边界条件。

### 4.3 Active scale

范围：

```text
active_scale ~ Uniform(0.8, 1.2)
```

作用：

- 改变 USD/root scale。
- 同步缩放当前 variant 的 part center、contact points 和 gt vector。
- 属于真实几何 domain randomization。

collection 模式在 prestartup 为每个 env × variant 预采样 scale，并记录到 `variant_scales`。reset 激活某个 variant 时，取对应 scale。

single 模式在 prestartup 为每个 env 当前 spawned object 采样一次 scale。

### 4.4 Precision scale

范围：

```text
precision_scale ~ Uniform(0.8, 1.4)
```

采样位置：

```text
reset_object_variant_by_reference()
```

作用：

- 不改变几何。
- 缩放 Part2Link reward 的 tolerance 和 sigma。
- 一个 env 在一次 episode 内共享同一个 precision scale，直到下一次 reset。

因此它是 reward precision randomization，不是 asset domain randomization。

## 5. Curriculum 设计

当前 Part2Link 有两条 curriculum：

```text
object_alpha_curriculum:
  alpha 1.0 -> 0.0 over 3,000,000 steps

tracking_sigma_annealing:
  sigma 0.35 -> 0.12 over 3,000,000 steps
```

两者配合后的含义：

- 早期：几何更接近初始 stage，reward 更宽。
- 后期：几何切向更难 alpha，reward 更精。
- reset 是几何切换边界；step 内不换 active variant。

这个设计适合并行 RL：课程变量全局推进，具体 env 在各自 reset 时吃到新的 difficulty。

## 6. Part2Link Reward 如何使用当前 variant

motion NPZ 提供：

- robot reference。
- object pose/velocity。
- relation matrix。
- 每帧 link-part target vector。
- contact/proximity phase。

当前 active chair variant 提供：

- `active_centers_local`
- `active_points_local`
- `active_point_valid_mask`
- `active_scale`

required pair 的 vector guidance：

```text
current_vector = link_pos_in_object_local - active_part_center_local
gt_vector      = sparse_contact_link_part_center_vector_w * active_scale
valid_mask     = relation == 1 AND contact/proximity phase active
```

reward 公式：

```text
effective_error = max(norm(current_vector - gt_vector) - tolerance, 0)
reward_pair     = exp(-(effective_error^2) / sigma^2)
```

其中：

```text
tolerance = 0.08 * precision_scale
sigma     = current_tracking_sigma * precision_scale
```

forbidden pair penalty 使用当前 chair 的 contact points：

```text
relation == -1
nearest_distance(link, active_part_points)
penalty = ((threshold - distance) / threshold)^2
```

当前阈值：

```text
forbidden_distance_threshold = 0.10 m
```

## 7. 当前实现效果

### 7.1 已实现的正向效果

- collection 模式下，多 chair × 多 alpha 的几何随机化已经能在 reset 时真实生效。
- `active_scale` 会同步影响 USD、part center、contact points 和目标向量，几何一致性是闭环的。
- `precision_scale` 和 `tracking_sigma_annealing` 给 reward 精度提供了随机化和课程化。
- object contact sensors 的 filter 会根据 object mode 切换，collection 模式能覆盖所有 variant prim。
- reset 时 inactive variants 被移到远离 active object 的位置，避免多个 object 同时参与接触。

### 7.2 已观察到的问题

历史 Transformer baseline 到约 47k iterations 的诊断显示：

- Mean episode reward 从约 0.03 到 14.28，但主要来自 episode length 变长。
- Mean reward/timestep 只从约 0.034 到 0.039，交互质量几乎没有提升。
- Imitation reward 和 interaction reward 比例约为 17.5:1，interaction 信号被淹没。
- `part2link_vector_guidance_gauss` per-step 约 0.0015，`seat_object_contact_ref_phase` per-step 约 0.0012。
- `part2link_forbidden_contact_penalty` 基本为 0，没有有效约束 forbidden contact。
- `object_nonseat_contact` termination 上升，说明不该接触的位置开始被使用。
- `base_pos_error_xy` 从约 0.040 m 恶化到约 0.165 m，可能出现“学会存活但减少主动推/拉物体”的行为。
- 训练窗口内 alpha 从 1.0 到约 0.992，只推进 0.8%，短训练里 object 难度几乎不变。

结论：机制层面的 domain randomization/curriculum 已搭好，但 reward 权重和 forbidden penalty 逻辑仍是主要瓶颈。不能只看 total episode reward，需要同时看 per-step reward、Part2Link term、forbidden violation 和 base XY error。

## 8. Staged Single 训练脚本

`train_part2link_alpha_stages.py` 的目的，是避免一次加载全部 alpha assets：

```text
alpha order: 1.0, 0.8, 0.5, 0.2, 0.0
```

每个 stage 设置：

```text
SITTING_PART2LINK_OBJECT_MODE=single
SITTING_PART2LINK_ALPHA_VALUES=<当前 alpha>
```

然后顺序 resume 训练。

这种方式的特点：

- 显存/场景复杂度更低。
- 每个 stage 的几何确实固定在当前 alpha。
- alpha 难度切换发生在 run/stage 边界，而不是 env reset 边界。
- 不等价于 collection 模式下的在线 curriculum，但更容易控制和调试。

adaptive 模式会读取 TensorBoard scalar，用 mean reward plateau、Part2Link 不退化和 forbidden 不恶化作为切换依据。不过当前 forbidden penalty 基本无信号，因此该条件的可靠性有限。

## 9. 对 InstinctMJ 的迁移参考

InstinctMJ 当前设计要点：椅子通过 mocap body 加载，所有变体在 scene 编译时预加载，运行时通过 mocap pose 控制显示/隐藏和切换，不能训练中途动态添加新 mesh。

因此推荐映射如下：

| InstinctLab | InstinctMJ 等价实现 |
| --- | --- |
| `RigidObjectCollectionCfg` 预加载 variants | scene 编译时预加载 mocap object bodies |
| inactive variants 移到远处 | mocap pose 移到 inactive 区域或禁用可见/碰撞等价机制 |
| reset 选择 active variant | reset 时设置 active body pose，更新 active variant id |
| `active_centers_local/points_local` | 根据 active variant id 从 metadata/cache 中 gather |
| `active_scale` | 若 MJ mesh 不能 runtime scale，预生成 scale variant 或仅作为数据缩放变量使用 |
| `precision_scale` | reset 时采样，存 env state，只作用于 reward |
| alpha curriculum | 全局 alpha state + reset 时 nearest alpha selection |
| staged single | 每个 run 只加载一个 alpha subset，stage 间 resume |

迁移时最重要的是保持三件事一致：

1. active variant 的几何和 reward 中使用的 part center/contact points 必须同源。
2. alpha curriculum 必须在 reset 触发 active variant 更新，否则只是日志变量。
3. scale 若改变可见/碰撞几何，也必须同步改变 Part2Link target vector 和 contact geometry。

## 10. 推荐改进顺序

1. 先确认 InstinctMJ 当前 asset cache 的 chair 列表、alpha 列表、metadata shape 和 InstinctLab 一致。
2. 给 active variant、active alpha、active scale、precision scale 加 TensorBoard/monitor。
3. 修复或重写 forbidden penalty 的 phase/mask 逻辑，至少记录 forbidden violation count。
4. 提高 interaction reward 权重或降低 imitation reward 权重，避免 17.5:1 的信号淹没。
5. 把评估指标从 total episode reward 扩展为 per-step reward、Part2Link pair error、base XY error、nonseat contact rate。
6. 如果 collection 模式太重，优先用 staged single 做稳定训练；如果要验证在线 curriculum，再使用 collection/mocap preloaded variant pool。

## 11. 最小复现检查表

- [ ] asset root 中每个 chair 都有目标 alpha 的 mesh/USD/OBJ。
- [ ] 每个 chair 的 contact points 与 alpha index 对齐。
- [ ] reset 后 active variant id、active alpha 和可见/碰撞物体一致。
- [ ] active object pose 来自 motion NPZ 的 object pose。
- [ ] inactive variants 不参与接触、深度图或 reward。
- [ ] Part2Link required pair 和 forbidden pair 使用同一 relation matrix。
- [ ] `active_scale` 同时影响几何、part centers、contact points 和 gt vector。
- [ ] `precision_scale` 只影响 tolerance/sigma，并按 episode 采样。
- [ ] alpha curriculum 的日志变化能对应到 reset 后 active alpha 变化。
- [ ] 评估时报告 per-step reward 和 interaction term，而不只报告 episode sum。

