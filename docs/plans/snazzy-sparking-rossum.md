# Multi-Variant Object Domain Randomization & Curriculum Training

> **Date**: 2026-07-07 | **Ref**: `docs/robot_object_domain_rand_curriculum_reference.md`

---

## 1. Overview

### 背景

当前 InstinctMJ 框架的 domain randomization 体系是**机器人中心**的：

| 随机化对象 | 属性 | 覆盖范围 |
|---|---|---|
| 机器人 | mass, friction, COM, PD gains | 所有任务 |
| 交互物体 | **无** | — |

Perceptive HOI 任务有 6 个硬编码 OMOMO 物体（`floorlamp`, `largebox`, `whitechair`, `trashcan`, `smalltable`, `suitcase`），每个只有一个固定 mesh，没有几何变体、没有物理属性随机化。

Part2Link interaction 任务已实现完善的多变体系统（20 chairs × N alpha variants + scale DR + precision DR + curriculum），但耦合在 `tasks/interaction/mdp/part2link.py` 内部，不能复用。

### 目标

1. **物体 Domain Randomization** — friction / COM / mass 随机化
2. **多变体物体支持** — 预加载多个几何变体，runtime 切换
3. **物体 Curriculum** — alpha geometry / scale / precision 渐进难度
4. **通用可复用** — variant 管理模块独立于具体任务
5. **向后兼容** — 不破坏现有 single-mesh HOI 训练

### 核心设计决策

| 决策 | 说明 |
|---|---|
| **Degenerate catalog** | Single-mesh 模式是 catalog 的特例（每种物体 1 variant），所有代码路径统一 |
| **Mocap body 范式** | 所有 variant 预加载为 mocap body，inactive 移到 scene 下方 100m |
| **物体 DR 默认启用** | 即使 single-mesh 模式，friction/COM DR 也生效 |
| **Env var gating** | Multi-variant 通过 `INSTINCT_HOI_VARIANT_ROOT` 启用 |
| **从 Part2Link 抽取** | 泛化现有 variant 系统，不另起炉灶 |

---

## 2. Architecture

### 目标文件布局

```
src/instinct_mj/
  envs/mdp/
    events/
      __init__.py                  # [MODIFY] + from .object_variant import *
      randomization.py             # [MODIFY] + randomize_object_{friction,com,mass}()
      object_variant.py            # [NEW] 通用 variant 管理（从 part2link 抽取泛化）
    curriculums/
      __init__.py                  # [MODIFY] + from .object_difficulty import *
      object_difficulty.py         # [NEW] 4 个 curriculum 类
  tasks/
    interaction/mdp/
      part2link.py                 # [REFACTOR] import 共享类型，保留 Part2Link-specific 逻辑
    shadowing/perceptive_hoi/
      mdp/                         # [NEW] HOI task-local MDP package
        __init__.py
        object_variant_hoi.py      # [NEW] OMOMO-specific catalog builder + entity factory
      perceptive_env_cfg.py        # [MODIFY] + make_hoi_object_dr_events()，扩展 make_hoi_events/curriculum
      config/g1/
        perceptive_shadowing_cfg.py # [MODIFY] catalog-driven entity creation，env var gating
  scripts/
    prepare_hoi_variants.py        # [NEW] Asset 准备脚本
```

### 模块依赖

```
Shared Layer (envs/mdp/)
├── object_variant.py        ← ObjectVariant/Catalog/RuntimeState, reset/update events
├── randomization.py         ← randomize_object_{friction,com,mass}
└── object_difficulty.py     ← ObjectAlpha/Scale/Precision Curriculum

HOI Layer (tasks/shadowing/perceptive_hoi/)
├── mdp/object_variant_hoi.py  ← build_hoi_{single,multi}_variant_catalog, make_hoi_variant_entities
├── perceptive_env_cfg.py      ← make_hoi_object_dr_events, make_hoi_events/curriculum
└── config/g1/perceptive_shadowing_cfg.py ← G1PerceptiveHoiShadowingEnvCfg

Part2Link Layer (tasks/interaction/)
└── mdp/part2link.py          ← import shared types from object_variant.py
```

### Data Flow: Single-Mesh Mode（默认，向后兼容）

```
_HOI_CATALOG (degenerate: 1 variant/type)
  │
  ├─→ make_hoi_variant_entities(catalog)
  │     └─→ EntityCfg × 6 — 等同当前硬编码
  ├─→ make_hoi_object_dr_events(catalog.variant_names)
  │     └─→ EventTermCfg × 6 × 2 (friction + COM per entity) — 新增
  └─→ make_hoi_events(object_dr_events=...)
        ├─→ 保留现有 reset/update_rigid_objects_state_by_reference (single-object 路径)
        └─→ 注入 object DR events

Runtime:
  startup → randomize_object_friction/COM on each object entity
  reset   → reset_rigid_objects_state_by_reference (motion ref → entity pose)
  interval → update_rigid_objects_state_by_reference (track motion ref)
```

### Data Flow: Multi-Variant Mode（`INSTINCT_HOI_VARIANT_ROOT` 启用）

```
_HOI_CATALOG (N variants × 6 types, e.g. 5×6=30 entities)
  │
  ├─→ make_hoi_variant_entities(catalog)
  │     └─→ EntityCfg × 30
  ├─→ make_hoi_object_dr_events(catalog.variant_names)
  │     └─→ EventTermCfg × 30 × 2
  └─→ make_hoi_events(
        object_dr_events=...,
        variant_reset_event=EventTermCfg(func=reset_object_variant_by_reference),
        variant_update_event=EventTermCfg(func=update_object_variant_by_reference),
      )

Runtime:
  startup → randomize_object_friction/COM on each variant entity
  reset   → reset_object_variant_by_reference()
             ├─→ _get_or_create_variant_state()
             ├─→ 每个 object type 随机选 1 variant
             ├─→ Active: motion ref pose; Inactive: scene 下方 100m
             └─→ 采样 active_scale, precision_scale
  interval → update_object_variant_by_reference()
             ├─→ Active variant 跟踪 ref trajectory
             └─→ Inactive variants 保持隐藏
  curriculum → ObjectAlphaCurriculum: 更新 active_alpha
             → ObjectScaleCurriculum: 更新 scale_distribution_params
```

---

## 3. Step 1 — 通用 Variant 管理模块

### 新文件: `src/instinct_mj/envs/mdp/events/object_variant.py`

从 `part2link.py` 抽取并泛化以下内容：

#### 核心 Dataclasses

```python
@dataclass(frozen=True)
class ObjectVariant:
    name: str                      # entity name, "chair_14_alpha_1p00"
    object_type: str               # semantic type, "chair_14"
    variant_index: int             # index within type
    mesh_path: Path
    contact_points_path: Path | None  # optional
    metadata: dict                 # {"alpha": 1.0, ...}

@dataclass
class ObjectVariantCatalog:
    variants: list[ObjectVariant]
    variant_names: list[str]       # all entity names
    object_types: list[str]        # unique type names
    type_to_variant_indices: dict[str, list[int]]
    centers_local: torch.Tensor | None  # [V, P, 3]，可选
    points_local: torch.Tensor | None   # [V, P, N, 3]，可选
    point_valid_mask: torch.Tensor | None
    part_names: list[str]

@dataclass
class ObjectVariantRuntimeState:
    catalog: ObjectVariantCatalog
    active_variant_ids: torch.Tensor       # [E] long
    active_scale: torch.Tensor             # [E] float
    active_precision_scale: torch.Tensor   # [E] float
    active_centers_local: torch.Tensor | None
    active_points_local: torch.Tensor | None
    active_point_valid_mask: torch.Tensor | None
    reference_position_offsets: torch.Tensor  # [E, 3]
```

#### 迁移函数清单

| 从 part2link.py | 泛化改动 |
|---|---|
| `ObjectVariant` (L26-33) | 增加 `object_type`, `variant_index`, `metadata` |
| `ObjectVariantCatalog` (L36-44) | 增加 `object_types`, `type_to_variant_indices`；centers/points → Optional |
| `ObjectVariantRuntimeState` (L47-57) | active_alpha 移除（改为 curriculum 动态计算）|
| `_CATALOG_CACHE` (L60) | 不变 |
| `_alpha_to_token()` (L72) | 不变 |
| `_sanitize_variant_name()` (L80) | 不变 |
| `_read_part_names()` (L144) | 不变 |
| `_resolve_chair_dirs()` (L103) | 泛化为 `_resolve_object_dirs()` |
| `_resolve_mesh_path()` (L120) | 泛化，支持 .obj/.glb/.usd |
| `load_object_variant_catalog()` (L151) | 参数: `extension_root` + `object_types` + `variant_specs` |
| `_select_variants_for_reset()` (L357) | 每种 object type 独立 random select |
| `_inactive_pose()` (L336) | 不变 |
| `_INACTIVE_OBJECT_SPACING` (L61) | 不变 |
| `reset_object_variant_by_reference()` (L398) | 泛化参数签名 |
| `update_object_variant_by_reference()` (L472) | 泛化参数签名 |
| `_get_or_create_variant_state()` | 泛化 |
| `_active_object_state_w()` (L516) | 迁移 |

#### 保留在 part2link.py

- `part2link_vector_guidance_gauss` / `part2link_forbidden_contact_penalty` / `seat_object_contact` — Part2Link-specific reward
- `_compute_part2link_vectors_and_mask` — reward helper
- `Part2LinkDebugVisualizer` — Part2Link-specific
- `TrackingSigmaCurriculum` — reward-specific curriculum

### 修改: `src/instinct_mj/envs/mdp/events/__init__.py`

```python
from .motion_reference import *
from .object_variant import *     # NEW
from .randomization import *
from .terrain import *
```

---

## 4. Step 2 — 物体 DR Events

### 修改: `src/instinct_mj/envs/mdp/events/randomization.py`

新增三个函数，镜像 robot DR 但 target 指向物体 entity：

```python
def randomize_object_friction(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    static_friction_range: tuple[float, float] = (0.3, 1.6),
    dynamic_friction_range: tuple[float, float] = (0.3, 1.2),
) -> None:
    """Randomize object geom friction. Wraps dr.geom_friction()."""

def randomize_object_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_range: dict[str, tuple[float, float]],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
) -> None:
    """Randomize object COM. Wraps dr.body_ipos()."""

def randomize_object_mass(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_range: tuple[float, float] = (0.8, 1.2),
    operation: Literal["add", "scale", "abs"] = "scale",
) -> None:
    """Randomize object mass. Wraps dr.body_mass()."""
```

实现要点:
- 复用现有 `_DR_ADD_CURRENT` custom operation
- 使用 `SceneEntityCfg("entity_name", ...)` 指向物体
- friction 对 mocap 物体立即生效（影响接触动力学）
- COM/mass 对 mocap 物体影响接触力，trajectory 不变；主要面向未来 free-body

---

## 5. Step 3 — 物体 Curriculum

### 新文件: `src/instinct_mj/envs/mdp/curriculums/object_difficulty.py`

```python
class ObjectAlphaCurriculum(ManagerTermBase):
    """Alpha 1.0→0.0，几何从 canonical 到 harder morph。"""
    def __call__(self, env, env_ids,
                 initial_alpha=1.0, final_alpha=0.0,
                 start_step=0, end_step=3_000_000) -> dict[str, float]:
        ...

class ObjectScaleCurriculum(ManagerTermBase):
    """Scale range 从窄到宽: (0.95,1.05) → (0.8,1.2)。"""
    def __call__(self, env, env_ids,
                 initial_scale_range=(0.95, 1.05),
                 final_scale_range=(0.8, 1.2),
                 start_step=0, end_step=1_000_000) -> dict[str, float]:
        ...

class ObjectPrecisionScaleCurriculum(ManagerTermBase):
    """Precision 从宽到严: (1.0,1.6) → (0.8,1.4)。"""
    def __call__(self, env, env_ids,
                 initial_precision_range=(1.0, 1.6),
                 final_precision_range=(0.8, 1.4),
                 start_step=0, end_step=1_000_000) -> dict[str, float]:
        ...

class ObjectVariantCountCurriculum(ManagerTermBase):
    """逐步引入更多 variant: 1 → N。"""
    def __call__(self, env, env_ids,
                 initial_count=1, final_count=5,
                 start_step=0, end_step=2_000_000) -> dict[str, float]:
        ...
```

### 修改: `src/instinct_mj/envs/mdp/curriculums/__init__.py`

```python
from .motion_reference import *
from .object_difficulty import *  # NEW
```

---

## 6. Step 4 — HOI Variant Builder

### 新文件: `src/instinct_mj/tasks/shadowing/perceptive_hoi/mdp/__init__.py`

```python
from .object_variant_hoi import *
```

### 新文件: `src/instinct_mj/tasks/shadowing/perceptive_hoi/mdp/object_variant_hoi.py`

桥接通用 `object_variant.py` 到 OMOMO 物体:

```python
OMOMO_OBJECT_TYPES = ("floorlamp", "largebox", "whitechair",
                       "trashcan", "smalltable", "suitcase")

# 从当前 perceptive_shadowing_cfg.py 迁移的硬编码数据
OMOMO_MESH_FILE_PATHS: dict[str, str]
OMOMO_MESH_SCALES: dict[str, tuple[float, float, float]]

def build_hoi_single_mesh_catalog(device="cpu") -> ObjectVariantCatalog:
    """退化 catalog: 每种 OMOMO 物体 1 variant。等价于当前硬编码。"""

def build_hoi_multi_variant_catalog(
    extension_root: str | Path,
    object_types: Sequence[str] = OMOMO_OBJECT_TYPES,
    asset_cache: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> ObjectVariantCatalog:
    """Multi-variant catalog: 从 extension_root/{type}/ 扫描变体。
    预期: {type}/variant_0.obj, variant_1.glb, contact_points_local.npz
    """

def make_hoi_variant_entities(
    catalog: ObjectVariantCatalog,
    include_reference: bool = False,
    robot_cfg = G1_CFG,
) -> dict[str, EntityCfg]:
    """catalog-driven EntityCfg dict。替换当前 _make_hoi_entities() 硬编码循环。"""

def make_hoi_variant_camera_mesh_prim_paths(
    catalog: ObjectVariantCatalog,
) -> list[str]:
    """camera mesh_prim_paths 包含所有 variant entity。"""
```

---

## 7. Step 5-6 — HOI Config Integration

### 修改: `perceptive_env_cfg.py`

```python
def make_hoi_object_dr_events(
    object_names: list[str],
    friction_range=(0.5, 2.0),
    com_range={"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
) -> dict[str, EventTermCfg]:
    """为 object_names 中每个 entity 生成 friction + COM DR events。"""

def make_hoi_events(
    object_dr_events: dict | None = None,
    variant_reset_event: EventTermCfg | None = None,
    variant_update_event: EventTermCfg | None = None,
) -> dict[str, EventTermCfg]:
    """扩展版。不传参=与当前完全相同。"""

def make_hoi_curriculum(
    object_curriculum_terms: dict | None = None,
) -> dict[str, CurriculumTermCfg]:
    """扩展版。不传参=与当前完全相同。"""
```

### 修改: `config/g1/perceptive_shadowing_cfg.py`

**模式选择**（模块级）:

```python
HOI_VARIANT_ROOT = os.getenv("INSTINCT_HOI_VARIANT_ROOT", None)
HOI_VARIANT_ASSET_CACHE = os.getenv("INSTINCT_HOI_VARIANT_ASSET_CACHE", None)
_enable_multi_variant = HOI_VARIANT_ROOT is not None

def _load_hoi_catalog() -> ObjectVariantCatalog: ...

_HOI_CATALOG = _load_hoi_catalog()
```

**Entity 构建替换**:

```python
# 当前: 硬编码 MESH_FILE_PATHS 循环
# 改为:
def _make_hoi_entities(*, include_reference=False):
    return make_hoi_variant_entities(_HOI_CATALOG,
        include_reference=include_reference, robot_cfg=G1_CFG)
```

**scene_object_names 始终是 6 个 type name**（不是 variant names）:

```python
scene_object_names = _HOI_CATALOG.object_types
```

Motion reference 数据中 object slot 按 type 索引，variant 选择在 reset event 内部处理。

**Event wiring** (`__post_init__`):

```python
# 总是注入物体 DR
object_dr = make_hoi_object_dr_events(_HOI_CATALOG.variant_names)
self.events.update(object_dr)

if _enable_multi_variant:
    # 替换为 variant-aware events
    self.events["reset_rigid_objects_state_by_reference"] = EventTermCfg(
        func=reset_object_variant_by_reference, mode="reset",
        params={"object_entity_names": _HOI_CATALOG.variant_names, ...})
    self.events["update_rigid_objects_state_by_reference"] = EventTermCfg(
        func=update_object_variant_by_reference, mode="interval",
        interval_range_s=(0.02, 0.02), params={...})
    self.curriculum["object_alpha_curriculum"] = CurriculumTermCfg(...)
    self.curriculum["object_scale_curriculum"] = CurriculumTermCfg(...)
```

**PLAY 模式**: `G1PerceptiveHoiShadowingEnvCfg_PLAY` 同步处理新 events 的 disable。

### 向后兼容性保证

| 检查项 | 保证方式 |
|---|---|
| 不设置 env var → 行为不变 | `build_hoi_single_mesh_catalog()` 产生与当前硬编码完全相同的 entity |
| Entity name/mesh/scale 一致 | catalog variant 的 `name == object_type`，mesh path 相同 |
| Motion reference 索引不变 | `scene_object_names` 仍是 6 个 type name |
| Camera paths 不变 | single-mesh 模式 `make_hoi_variant_camera_mesh_prim_paths()` 产生相同路径 |

---

## 8. Step 7 — Asset Pipeline

### 新文件: `src/instinct_mj/scripts/prepare_hoi_variants.py`

```python
"""Prepare variant meshes for HOI object DR.

Usage:
    python -m instinct_mj.scripts.prepare_hoi_variants \
        --variant-root ~/Datasets/OMOMO/object_variants \
        --asset-cache ~/.cache/instinct_mj/hoi_assets \
        --object-types floorlamp,largebox,whitechair
"""

def _export_glb_to_obj(glb_path: Path, obj_path: Path) -> None:
    """Convert GLB → OBJ via trimesh."""

def _validate_contact_points(npz_path: Path) -> bool:
    """Check keys: alpha_values, points_local, center_local."""
```

输入结构: `{root}/{type}/variant_*.glb` + `contact_points_local.npz` (optional)
输出结构: `{cache}/{type}/variant_*.obj`

环境变量: `INSTINCT_HOI_VARIANT_ROOT` (启用 multi-variant), `INSTINCT_HOI_VARIANT_ASSET_CACHE` (OBJ 缓存)

---

## 9. Step 8 — Part2Link Refactor

### 修改: `src/instinct_mj/tasks/interaction/mdp/part2link.py`

```python
# 从共享模块 import
from instinct_mj.envs.mdp.events.object_variant import (
    ObjectVariant, ObjectVariantCatalog, ObjectVariantRuntimeState,
    load_object_variant_catalog,
    reset_object_variant_by_reference, update_object_variant_by_reference,
    _get_or_create_variant_state, _select_variants_for_reset,
    _inactive_pose, _CATALOG_CACHE,
    # ... 所有工具函数
)

# 删除/注释原有重复定义，改为 "→ Moved to object_variant.py"
```

保留 Part2Link-specific 逻辑: reward functions, TrackingSigmaCurriculum, Part2LinkDebugVisualizer

现有 interaction 任务和 `train_part2link_alpha_stages.py` 无需修改（import 路径不变）。

---

## 10. DR 参数范围参考

| 参数 | 初始（保守）| 后期（激进）| 说明 |
|---|---|---|---|
| Object friction (slide) | (0.5, 1.5) | (0.2, 2.0) | 接触滑移 |
| Object COM offset | ±1cm | ±3cm | 接触力矩 |
| Active scale | (0.95, 1.05) | (0.8, 1.2) | 物体尺寸 |
| Precision scale | (1.0, 1.6) | (0.8, 1.4) | Reward tolerance |
| Alpha morph | 1.0 | 0.0 | 几何变形程度 |

---

## 11. Verification

### Test 1: Single-Mesh 回归
```bash
# 不设置 INSTINCT_HOI_VARIANT_ROOT
python -m instinct_mj.scripts.instinct_rl.play \
    --task Instinct-Perceptive-HOI-Shadowing-G1-Play-v0 --num_envs 1
```
- [ ] Entity 数量/名称与当前一致
- [ ] 物体位置/姿态与 ref 一致
- [ ] Camera depth 不变

### Test 2: 物体 DR 验证
- [ ] 不同 run 之间物体 geom friction 不同
- [ ] COM 偏移在合理范围
- [ ] Robot DR 不受影响

### Test 3: Multi-Variant 功能
```bash
export INSTINCT_HOI_VARIANT_ROOT=/tmp/test_hoi_variants
# 准备最小测试数据: 2 types × 2 variants each
```
- [ ] 所有 variant entities 在场景中
- [ ] 不同 env 获得不同 variant
- [ ] Inactive variants 不可见

### Test 4: Curriculum
- [ ] TensorBoard `object_alpha/scale/precision` 随时间变化

### Test 5: Part2Link 回归
```bash
python -m instinct_mj.scripts.instinct_rl.train \
    --task Instinct-Interaction-Sitting-Part2Link-G1-v0 \
    --num_envs 64 --max_iterations 10
```
- [ ] 场景编译正常，训练 metrics 一致

---

## 12. Implementation Order

```
Step 1: object_variant.py (NEW) ─────────────────────────────────┐
    ↓                                                             │
Step 2: randomization.py (MODIFY) ───┐                             │
    ↓                                ↓                             │
Step 3: object_difficulty.py (NEW)   Step 4: object_variant_hoi.py (NEW)
    ↓                                ↓                             │
    ├────────────────────────────────┤                             │
    ↓                                ↓                             │
Step 5-6: perceptive_env_cfg.py + perceptive_shadowing_cfg.py (MODIFY)
    ↓                                                             │
Step 7: prepare_hoi_variants.py (NEW) ── (独立，可并行)             │
    ↓                                                             │
Step 8: part2link.py (REFACTOR) ←─────────────────────────────────┘
```

Step 7（asset 脚本）与其他步骤完全独立，可并行开发。其余步骤有严格依赖顺序。

---

## 13. Review Follow-up Fix Plan

> 本节基于已完成代码审查追加，用于指导后续 agent 修复当前实现中的行为漏洞。目标是先恢复默认 HOI single-mesh 行为，再补齐 multi-variant 与 curriculum 的真实接线，最后完成 Part2Link 去重抽取。

### Fix 1: 恢复 HOI object 顺序稳定性

**问题**

`ObjectVariantCatalog.object_types` 当前通过 `sorted(set(self.chair_names))` 生成，会改变 OMOMO 原始物体顺序。HOI 的 `reset_rigid_objects_state_by_reference()` / `update_rigid_objects_state_by_reference()` 按 `scene_object_names` 索引写入物体状态，不按名字匹配；顺序变化会导致 object pose slot 与 entity 错配。

**修改文件**

- `src/instinct_mj/envs/mdp/events/object_variant.py`
- `src/instinct_mj/tasks/shadowing/perceptive_hoi/mdp/object_variant_hoi.py`

**修改要求**

1. `ObjectVariantCatalog.object_types` 必须保留 catalog 中首次出现顺序，不得排序。
2. `type_to_variant_indices` 必须同样保持 variant 在 catalog 中的原始顺序。
3. `build_hoi_single_mesh_catalog()` 返回的 object type 顺序必须与 `OMOMO_OBJECT_TYPES` 完全一致。
4. `motion_reference_cfg.scene_object_names` 在默认 single-mesh 模式下必须等于：

```python
[
    "floorlamp",
    "largebox",
    "whitechair",
    "trashcan",
    "smalltable",
    "suitcase",
]
```

**验证**

- 添加或运行轻量检查：构造 `build_hoi_single_mesh_catalog()`，断言 `catalog.object_types == list(OMOMO_OBJECT_TYPES)`。
- 手动检查默认 HOI config 中 `scene_object_names` 未被排序。

---

### Fix 2: 统一 HOI multi-variant catalog 与 runtime reset 数据源

**问题**

HOI multi-variant catalog 由 `build_hoi_multi_variant_catalog()` 从 `root/type/*.obj|*.glb|*.usd` 构建，但 `reset_object_variant_by_reference()` 内部又调用通用 `load_object_variant_catalog()`，该函数要求 Part2Link 风格目录：

```text
ffd_bbox_coarse/morph_path/
contact_point_transfer/contact_points_local.npz
```

这会导致 HOI multi-variant 场景创建成功后，在 reset 阶段重新加载失败或使用不同 catalog。

**修改文件**

- `src/instinct_mj/envs/mdp/events/object_variant.py`
- `src/instinct_mj/tasks/shadowing/perceptive_hoi/config/g1/perceptive_shadowing_cfg.py`
- 必要时：`src/instinct_mj/tasks/shadowing/perceptive_hoi/mdp/object_variant_hoi.py`

**修改要求**

1. `reset_object_variant_by_reference()` 应支持直接接收已构建的 `ObjectVariantCatalog`，例如新增参数：

```python
catalog: ObjectVariantCatalog | None = None
```

2. `_get_or_create_variant_state()` 也应支持传入 catalog；当 catalog 非空时不得再从 `extension_root` 重载。
3. HOI multi-variant config 中的 reset event 必须传入 `_HOI_CATALOG`，确保 scene entities、motion reference 和 runtime state 使用同一个 catalog。
4. 不要新增兼容层或 adapter；这是对现有 runtime state 初始化参数的最小扩展。
5. 保留 Part2Link 原有 `extension_root/chair_names/alpha_values/asset_cache` 路径，避免破坏现有 interaction 任务。

**验证**

- 构造最小 HOI catalog（2 object types × 2 variants），调用 `_get_or_create_variant_state(..., catalog=catalog)` 不访问磁盘。
- `reset_object_variant_by_reference()` 在传入 catalog 时不调用 `load_object_variant_catalog()`。

---

### Fix 3: 将 active variant state 改为按 object type 管理

**问题**

当前 `_select_variants_for_reset()` 返回 `[num_envs]`，每个 env 只激活一个全局 variant。HOI 需要每个 object type 各自激活一个 variant，即状态形状应表达 `[num_envs, num_object_types]` 或等价结构。

**修改文件**

- `src/instinct_mj/envs/mdp/events/object_variant.py`
- `src/instinct_mj/tasks/interaction/mdp/part2link.py`（若仍保留本地实现，也必须同步；推荐先执行 Fix 6）

**修改要求**

1. 将 `ObjectVariantRuntimeState.active_variant_ids` 从 `[E]` 改为 `[E, T]`，其中 `T = len(catalog.object_types)`。
2. `active_alpha` / `active_scale` / `active_precision_scale` 应同步改为 `[E, T]`，或明确只用于单物体任务时保持兼容，但 HOI 必须能表达每类物体独立状态。
3. `_select_variants_for_reset()` 应对每个 object type 独立采样一个 variant，返回 `[num_reset, num_object_types]`。
4. reset/update 循环中，判断某个 entity 是否 active 时不能再用 `variant_ids == variant_index` 的一维比较，而应通过 catalog 中 `variant_index -> object_type_index` 的关系判断。
5. HOI update 必须对每个 object type 使用对应 object slot，而不是固定 `object_name="box"` 或 fallback slot 0。
6. Part2Link 只有一个 object type，应仍能退化为 `[E, 1]`，对 reward helper 的 active object 查询保持行为一致。

**验证**

- 最小 catalog：2 types × 2 variants。
- 对每个 env reset 后，每个 type 恰好一个 active variant。
- inactive variants 被移动到隐藏位置。
- `state.active_variant_ids.shape == (num_envs, num_object_types)`。

---

### Fix 4: 让 variant sampling 和 curriculum 真正生效

**问题**

当前 curriculum term 写入：

- `env._object_variant_current_alpha`
- `env._object_variant_scale_range`
- `env._object_variant_precision_range`
- `env._object_variant_active_count`

但 reset 逻辑没有读取这些属性；同时 `_select_variants_for_reset()` 总是 `candidates[0]`，没有随机选 variant，也没有按 alpha/count 限制。

**修改文件**

- `src/instinct_mj/envs/mdp/events/object_variant.py`
- `src/instinct_mj/envs/mdp/curriculums/object_difficulty.py`
- `src/instinct_mj/tasks/shadowing/perceptive_hoi/config/g1/perceptive_shadowing_cfg.py`

**修改要求**

1. `_select_variants_for_reset()` 在每个 object type 的候选列表内随机采样，不得固定 `candidates[0]`。
2. 若存在 `env._object_variant_active_count`，只允许每个 type 的前 `active_count` 个候选参与采样。
3. 若 catalog variants 有 alpha metadata，且存在 `env._object_variant_current_alpha`，应选择最接近当前 alpha 的候选，或在不超过当前难度范围的候选中随机采样。具体策略必须简单、可解释，并在注释中写清楚。
4. reset 时优先读取：

```python
scale_distribution_params = getattr(env, "_object_variant_scale_range", scale_distribution_params)
precision_scale_range = getattr(env, "_object_variant_precision_range", precision_scale_range)
```

5. HOI config 若只启用 `ObjectAlphaCurriculum` / `ObjectScaleCurriculum`，不要声明未接线的 precision/count curriculum；若启用，则必须接入 reset。

**验证**

- 固定随机种子下，多次 reset 能采到不同 variant。
- 改变 `env._object_variant_active_count` 后，可采样 variant 数量随之变化。
- 改变 `env._object_variant_scale_range` 后，`state.active_scale` 落在新范围内。
- TensorBoard/manager metrics 中 curriculum 数值变化与 reset state 一致。

---

### Fix 5: 修正 HOI variant event 参数

**问题**

HOI multi-variant config 中 reset event 传入 `"object_name": "box"`，但 HOI `scene_object_names` 中没有 `box`。当前 `_scene_object_slot()` 会 warning 后 fallback 到 slot 0，导致所有 active object 都跟踪第一个 object slot。

**修改文件**

- `src/instinct_mj/tasks/shadowing/perceptive_hoi/config/g1/perceptive_shadowing_cfg.py`
- `src/instinct_mj/envs/mdp/events/object_variant.py`

**修改要求**

1. HOI variant reset/update 不应传入 `"object_name": "box"`。
2. reset/update 应按 catalog.object_types 遍历每个 object type，并用 object type 名称在 `motion_ref.cfg.scene_object_names` 中找到对应 slot。
3. Part2Link 可继续使用 `object_name="box"`，因为其 motion reference 只有一个 object slot。
4. 对 HOI 缺失 object slot 的处理应与现有 HOI motion buffer 逻辑一致；不要新增静默 fallback 到 slot 0 的行为用于 HOI multi-object 路径。

**验证**

- HOI 6 个 object type 每个都写入对应 reference slot。
- 删除 `"object_name": "box"` 后不再出现 fallback warning。

---

### Fix 6: 真正完成 Part2Link shared variant 抽取

**问题**

`src/instinct_mj/tasks/interaction/mdp/part2link.py` 当前先 import 共享 `object_variant.py` 的同名类和函数，随后又重新定义这些类和函数，导致 import 被覆盖。实际并未复用共享模块。

**修改文件**

- `src/instinct_mj/tasks/interaction/mdp/part2link.py`
- `src/instinct_mj/envs/mdp/events/object_variant.py`

**修改要求**

1. 删除 `part2link.py` 中已经迁移到 `object_variant.py` 的重复定义：

- `ObjectVariant`
- `ObjectVariantCatalog`
- `ObjectVariantRuntimeState`
- `_CATALOG_CACHE`
- `_INACTIVE_OBJECT_SPACING`
- `_alpha_to_token`
- `_alpha_to_glb_name`
- `_sanitize_variant_name`
- `_parse_alpha_values`
- `_resolve_chair_dirs` / 对应共享版本
- `_resolve_mesh_path`
- `_read_part_names`
- `load_object_variant_catalog`
- `_get_or_create_variant_state`
- `_safe_unit_quat`
- `_root_pos` / `_root_quat` / `_root_lin_vel` / `_root_ang_vel`
- `_write_pose_velocity`
- `_scene_object_slot`
- `_inactive_pose`
- `_select_variants_for_reset`
- `_update_active_variant_state`
- `_invalidate_dynamic_mesh_sensors`
- `reset_object_variant_by_reference`
- `update_object_variant_by_reference`
- `_active_object_state_w`

2. 保留 Part2Link-specific 内容：

- `part2link_vector_guidance_gauss`
- `part2link_forbidden_contact_penalty`
- `seat_object_contact`
- `_compute_part2link_vectors_and_mask`
- `Part2LinkDebugVisualizer`
- `TrackingSigmaCurriculum`

3. 若共享实现需要 Part2Link-only alias，例如 `_parse_chair_names`，在共享模块中以最小方式保留别名，不要在 Part2Link 中复制完整实现。
4. 删除未使用 import，保持 diff surgical。

**验证**

- `rg -n "^class ObjectVariant|^def load_object_variant_catalog|^def reset_object_variant_by_reference" src/instinct_mj/tasks/interaction/mdp/part2link.py` 不应再命中。
- Part2Link train/play smoke test 可以构建 env。

---

### Fix 7: 补充最小回归测试/检查脚本

**修改文件**

- 优先使用现有测试目录；若项目暂无 tests，可新增轻量脚本或在计划验证中记录手动命令。

**最小覆盖**

1. `build_hoi_single_mesh_catalog()` 顺序测试。
2. `ObjectVariantCatalog.type_to_variant_indices` 顺序测试。
3. 2 types × 2 variants 的 `_select_variants_for_reset()` shape 和 active count 测试。
4. curriculum scale range 被 reset 读取的测试。
5. Part2Link 模块不再定义重复 shared symbols 的 grep 检查。

**验证命令**

```bash
python3 -m py_compile \
    src/instinct_mj/envs/mdp/events/object_variant.py \
    src/instinct_mj/envs/mdp/events/randomization.py \
    src/instinct_mj/envs/mdp/curriculums/object_difficulty.py \
    src/instinct_mj/tasks/shadowing/perceptive_hoi/mdp/object_variant_hoi.py \
    src/instinct_mj/tasks/shadowing/perceptive_hoi/perceptive_env_cfg.py \
    src/instinct_mj/tasks/shadowing/perceptive_hoi/config/g1/perceptive_shadowing_cfg.py \
    src/instinct_mj/tasks/interaction/mdp/part2link.py
```

如环境可用，再运行：

```bash
python -m instinct_mj.scripts.instinct_rl.play \
    --task Instinct-Perceptive-HOI-Shadowing-G1-Play-v0 --num_envs 1
```

```bash
python -m instinct_mj.scripts.instinct_rl.train \
    --task Instinct-Interaction-Sitting-Part2Link-G1-v0 \
    --num_envs 64 --max_iterations 10
```

---

### Recommended Fix Order

```text
1. Fix object type ordering first.
   Verify: default HOI single-mesh scene_object_names unchanged.

2. Pass the already-built HOI catalog into variant reset/runtime state.
   Verify: no second disk catalog load in HOI multi-variant reset.

3. Change active variant state from one active entity per env to one active variant per object type.
   Verify: each HOI object type has exactly one active variant.

4. Wire curriculum attributes into reset sampling.
   Verify: scale/precision/alpha/count affect reset state.

5. Remove HOI object_name="box" fallback.
   Verify: each object type uses its own motion reference slot.

6. Remove duplicated Part2Link shared variant code.
   Verify: Part2Link imports shared definitions and smoke test still works.

7. Add minimal regression tests or scripted checks.
   Verify: py_compile plus catalog/sampling assertions pass.
```
