# InstinctLab 可视化迁移说明

更新日期：2026-07-06

本文总结当前 InstinctLab repo 中与 G1 interaction / sitting Part2Link 相关的可视化实现，并给出迁移到 InstinctMJ / mjlab-native 的建议。目标是让后续迁移时能直接按模块搬行为，而不是在 IsaacLab 的 `VisualizationMarkers`、USD prim path 和 Omniverse callback 里重新摸一遍。

源仓库：

```text
/home/yangke/KY/InstinctLab_interact
```

目标仓库：

```text
/home/yangke/KY/InstinctMJ
```

## 1. 总览

当前 repo 的可视化主要分为五类：

| 类别 | 源实现 | 训练必需 | 迁移优先级 |
| --- | --- | --- | --- |
| Motion reference markers | `motion_reference_cfg.py` / `motion_reference_manager.py` | 否 | 中 |
| Depth / camera debug | `compatible_tiled_camera.py` / `exteroception.py` / train-play patch | 否，但调试 depth 很有用 | 高 |
| Part2Link reward debug markers | `object_variant_guidance.py` | 否，但对迁移 reward 很关键 | 高 |
| Legacy sparse contact reward markers | `sparse_contact_reward.py` | 否，当前 Part2Link cfg 中基本是注释备用 | 低 |
| Direct replay quality monitor | `scripts/direct_replay_npz_quality_monitor.py` | 否，是离线诊断工具 | 中 |

迁移到 InstinctMJ 时，不应该保留 IsaacLab 的可视化对象模型：

```text
isaaclab.markers.VisualizationMarkers
isaaclab.markers.VisualizationMarkersCfg
isaaclab.sim.SphereCfg / CylinderCfg / UsdFileCfg
USD prim_path based visual groups
```

目标应使用 mjlab-native 入口：

```text
mjlab.viewer.ViewerConfig
mjlab.viewer.NativeMujocoViewer
mjlab.viewer.ViserPlayViewer
mjlab.viewer.OffscreenRenderer
mjlab.viewer.debug_visualizer.DebugVisualizer
instinct_mj.visualization.marker_cfg.VisualizationMarkersCfg
```

InstinctMJ 当前已有一个很重要的可视化入口：

```text
InstinctRlEnv.update_visualizers(visualizer)
```

它会调用父类 visualizer 更新，并额外注册：

```text
self.manager_visualizers["monitor_manager"] = self.monitor_manager
terrain.debug_vis(visualizer)
```

因此迁移自定义 Part2Link marker 时，推荐走：

```text
reward/observation 计算 debug cache
monitor term 或 sensor term 实现 debug_vis(visualizer)
viewer tick 调用 visualizer.add_sphere/add_cylinder/add_frame
```

不推荐在 reward 函数里直接持有 viewer object。

## 2. Motion Reference 可视化

### 2.1 源文件

```text
source/instinctlab/instinctlab/motion_reference/motion_reference_cfg.py
source/instinctlab/instinctlab/motion_reference/motion_reference_manager.py
scripts/instinct_rl/play.py
```

源配置字段：

```python
reference_prim_path: str | None
visualizer_cfg: VisualizationMarkersCfg
visualizing_marker_types: list[str]
visualizing_robot_offset: Sequence[float]
visualizing_robot_from: Literal["aiming_frame", "reference_frame"]
```

支持的 marker types：

```text
root
links
relative_links
```

源 marker 语义：

- `root`：画 reference root frame。
- `links`：画 motion reference 中每个 link 的 world position，绿色 sphere。
- `relative_links`：画 reference-relative link frame。
- 如果 scene 中有 reference robot articulation，会把 reference robot joint/root state 写到 reference view。

源 play 脚本还有轻量 patch：

```python
_enable_motion_reference_visualization(env_cfg)
```

行为：

- `motion_reference.debug_vis = True`
- 若 `visualizing_marker_types` 为空，则默认 `["links"]`
- 如果 scene 没有 `robot_reference`，则不设置 `reference_prim_path`

### 2.2 目标状态

InstinctMJ 已经基本迁移了 motion reference 可视化：

```text
src/instinct_mj/motion_reference/motion_reference_cfg.py
src/instinct_mj/motion_reference/motion_reference_manager.py
```

目标 `MotionReferenceManager.debug_vis(visualizer)` 已经使用 mjlab-native `DebugVisualizer`：

- `visualizer.add_frame(...)` 画 root / relative frames。
- `visualizer.add_sphere(...)` 画 link markers。
- `visualizer.get_env_indices(self._num_envs)` 控制当前 viewer 画哪些 env。

目标 Part2Link cfg 中也已经设置：

```python
visualizing_robot_offset=(2.0, 0.0, 0.0)
visualizing_robot_from="reference_frame"
visualizing_marker_types=["relative_links", "links"]
```

Play cfg 中：

```python
motion_reference_cfg_play.debug_vis = True
motion_reference_cfg_play.reference_entity_name = "robot_reference"
```

### 2.3 迁移建议

Motion reference 这一块不需要从源 repo 再大搬一次。后续只需要验证：

- `robot_reference` entity 存在时，reference robot 是否跟随 motion reference。
- `links` marker 是否画在 motion reference link position 上。
- `relative_links` frame 是否正确使用 `reference_link_pos_relative_w` / `reference_link_quat_relative_w`。
- Play cfg viewer 的 `entity_name` 是否对准 `robot_reference` 或 `robot`。

## 3. Depth / Camera Debug 可视化

### 3.1 源文件

```text
source/instinctlab/instinctlab/sensors/compatible_tiled_camera.py
source/instinctlab/instinctlab/envs/mdp/observations/exteroception.py
scripts/instinct_rl/train.py
scripts/instinct_rl/play.py
source/instinctlab/instinctlab/tasks/interaction/config/g1/g1_interaction_sitting_part2link_shadowing_cfg.py
```

### 3.2 源行为

源 repo 有两套 depth debug。

第一套在 `CompatibleTiledCamera` 内：

```python
DepthPointCloudDebugMixin
CompatibleTiledCameraCfg
```

行为：

- 当 sensor cfg `debug_vis=True` 时，创建 `VisualizationMarkers`。
- 从 camera output 读取 `distance_to_image_plane` 或 `depth`。
- 用 IsaacLab `create_pointcloud_from_depth(...)` 转成 world-frame point cloud。
- 过滤 far plane 和非正 depth。
- 支持：
  - `debug_vis_data_type`
  - `debug_vis_max_envs`
  - `debug_vis_stride`
  - `debug_vis_max_points`
  - `debug_vis_clip_far_plane`
- 最后用 sphere markers 画 depth point cloud。

第二套在 observation term `visualizable_image(...)` 内：

- 正常返回 policy depth/rgb image observation。
- `debug_vis_mode="image"` 时，用 OpenCV `cv2.imshow(...)` 显示拼接后的 image。
- `debug_vis_mode="pointcloud"` 或 `"both"` 时，把 depth 转 world point cloud 后画 sphere markers。
- 支持同样的 max env / stride / max points 控制。

源 train/play 脚本通过 `--visualize_camera` 开关做 patch：

```python
_enable_camera_visualization(env_cfg)
```

行为：

- 自动找到 policy depth observation 的 `sensor_cfg.name`。
- 优先打开对应 scene camera，其次 `tiled_camera`、`camera`。
- 设置：
  - `camera_cfg.debug_vis = True`
  - `camera_cfg.debug_vis_data_type = "distance_to_image_plane"`
  - `camera_cfg.debug_vis_max_envs = 1`
  - observation params `debug_vis=True`
  - 默认 `debug_vis_mode="image"`
  - 默认 `debug_vis_max_points=2048`

源 Part2Link PLAY cfg 默认打开：

```python
self.scene.tiled_camera.debug_vis = True
self.scene.tiled_camera.debug_vis_data_type = "distance_to_image_plane"
self.scene.tiled_camera.debug_vis_max_envs = 4
self.observations.policy.depth_image.params["debug_vis"] = True
self.observations.policy.depth_image.params["debug_vis_mode"] = "image"
self.observations.policy.depth_image.params["debug_vis_point_radius"] = 0.012
self.observations.policy.depth_image.params["debug_vis_max_envs"] = 1
self.observations.policy.depth_image.params["debug_vis_stride"] = 1
self.observations.policy.depth_image.params["debug_vis_max_points"] = 2048
```

### 3.3 目标状态

InstinctMJ 已经有 observation image debug：

```text
src/instinct_mj/envs/mdp/observations/exteroception.py
```

当前目标 Part2Link cfg 已经在 non-headless / play 下打开：

```python
camera_cfg.debug_vis = not self.headless
self.observations["policy"].terms["depth_image"].params["debug_vis"] = not self.headless
```

目标 play 脚本会在非 native / 无 display 时关闭 GUI-dependent debug visualization：

```text
src/instinct_mj/scripts/instinct_rl/play.py
```

相关函数：

```python
_disable_headless_debug_visualization(env_cfg)
```

行为：

- 关闭 sensor cfg 的 `debug_vis`。
- 关闭 command cfg 的 `debug_vis`。
- 关闭 observation term params 中的 `debug_vis`。

### 3.4 迁移建议

优先保留两层能力：

1. **Image window debug**

   如果目标 `visualizable_image(...)` 已经能 `cv2.imshow`，只需要确认 Part2Link cfg 把 params 设齐：

   ```python
   debug_vis=True
   debug_vis_mode="image"
   debug_vis_max_envs=1
   debug_vis_stride=1
   debug_vis_max_points=2048
   ```

2. **Point cloud debug**

   源的 pointcloud debug 依赖 IsaacLab `create_pointcloud_from_depth(...)`。迁到 mjlab 时不要直接复制这个 import。可选实现：

   - 使用 mjlab camera sensor 已有的 depth/intrinsic/extrinsic 数据，自己写 depth-to-pointcloud。
   - 或者先不迁 pointcloud，只保留 image debug 和 native viewer camera。

建议 v1 只迁 image debug；pointcloud debug 等 depth sensor runtime smoke 通过后再补。

## 4. Part2Link Reward Debug Markers

### 4.1 源文件

```text
source/instinctlab/instinctlab/tasks/interaction/mdp/object_variant_guidance.py
source/instinctlab/instinctlab/tasks/interaction/config/g1/g1_interaction_sitting_part2link_shadowing_cfg.py
scripts/instinct_rl/train.py
```

这是 sitting Part2Link 最值得迁移的可视化部分。

### 4.2 源 marker 语义

源 `part2link_vector_guidance_gauss(...)` 的 debug params：

```python
debug_vis: bool = False
debug_vis_max_envs: int = 1
debug_vis_show_contact_points: bool = True
debug_vis_show_part_centers: bool = True
debug_vis_point_radius: float = 0.025
debug_vis_center_radius: float = 0.035
debug_vis_line_radius: float = 0.008
debug_vis_ignore_contact_phase: bool = False
```

源 marker types：

| Marker | Primitive | Color | 含义 |
| --- | --- | --- | --- |
| `contact_point` | sphere | yellow | active object part contact points |
| `part_center` | sphere | orange | object part centers |
| `realtime_vector` | cylinder | red | current part center -> robot link vector |
| `gt_vector` | cylinder | blue | part center -> sparse GT vector endpoint |

源逻辑拆分：

```python
_build_part2link_point_visualizer_cfg(...)
_build_part2link_vector_visualizer_cfg(...)
_get_part2link_debug_visualizers(...)
_set_part2link_debug_visibility(...)
_transform_local_points_to_world(...)
_line_marker_pose(...)
_update_part2link_debug_visualization(...)
```

核心可视化数据流：

```text
object_pos_w / object_quat_w
link_pos_w
active part_centers_local
active part_points_local
active point_valid_mask
gt_vector_local
valid_mask
```

先把 object-local part centers / points 转到 world：

```text
part_centers_w = object_pos_w + quat_apply(object_quat_w, part_centers_local)
part_points_w  = object_pos_w + quat_apply(object_quat_w, part_points_local)
```

再画两类线：

```text
realtime_vector:
  start = part_center_w
  end   = link_pos_w

gt_vector:
  start = part_center_w
  end   = object_pos_w + quat_apply(object_quat_w, part_center_local + gt_vector_local)
```

`valid_mask` 控制哪些 link-part pair 被画。默认使用 reward 的 contact phase mask；当 `debug_vis_ignore_contact_phase=True` 时，改为只看 relation 或 GT vector 是否非零，方便查看所有 sparse pair。

### 4.3 目标状态

InstinctMJ Part2Link reward 当前保留了 `debug_vis` 参数，但还没有实际绘制：

```text
src/instinct_mj/tasks/interaction/mdp/part2link.py
```

当前状态类似：

```python
def part2link_vector_guidance_gauss(..., debug_vis: bool = False):
    del precision_scale_range, debug_vis
    ...
```

目标 PLAY cfg 已经打开了：

```python
self.rewards["part2link_vector_guidance_gauss"].params["debug_vis"] = True
```

所以目前迁移缺口是：

```text
cfg 开关已接上，但 Part2Link marker drawing 未实现。
```

### 4.4 推荐迁移方式

不要在 reward 函数里直接画。推荐拆成两步：

1. 在 reward/MDP helper 中缓存 debug data：

   ```python
   env._part2link_debug_cache = {
       "object_pos_w": object_pos_w.detach(),
       "object_quat_w": object_quat_w.detach(),
       "link_pos_w": link_pos_w.detach(),
       "part_centers_local": state.active_centers_local.detach(),
       "part_points_local": state.active_points_local.detach(),
       "point_valid_mask": state.active_point_valid_mask.detach(),
       "gt_vector_local": gt_vector.detach(),
       "valid_mask": debug_valid_mask.detach(),
       "max_envs": debug_vis_max_envs,
       "show_contact_points": debug_vis_show_contact_points,
       "show_part_centers": debug_vis_show_part_centers,
       "point_radius": debug_vis_point_radius,
       "center_radius": debug_vis_center_radius,
       "line_radius": debug_vis_line_radius,
   }
   ```

2. 新增一个 monitor term 或 scene-level debug object 实现：

   ```python
   def debug_vis(self, visualizer: DebugVisualizer) -> None:
       cache = getattr(self.env, "_part2link_debug_cache", None)
       if cache is None:
           return
       ...
       visualizer.add_sphere(...)
       visualizer.add_cylinder(...)
   ```

推荐放置：

```text
src/instinct_mj/tasks/interaction/mdp/part2link.py
```

或拆成：

```text
src/instinct_mj/tasks/interaction/mdp/part2link_debug.py
```

若使用 monitor manager，需要在 Part2Link cfg 的 `monitors` dict 中添加 term。目标 env 已经把 `monitor_manager` 注册到 visualizers：

```python
self.manager_visualizers["monitor_manager"] = self.monitor_manager
```

### 4.5 Primitive 映射

IsaacLab 源：

```python
VisualizationMarkers.visualize(
    translations=...,
    orientations=...,
    scales=...,
    marker_indices=...,
)
```

InstinctMJ / mjlab 目标：

```python
visualizer.add_sphere(center=point, radius=radius, color=rgba)
visualizer.add_cylinder(start=start, end=end, radius=radius, color=rgba)
visualizer.add_frame(position=pos, rotation_matrix=rot, scale=scale, ...)
```

因此 Part2Link vector 不需要保留 `_line_marker_pose(...)` 中的 cylinder orientation 计算。目标可以直接：

```python
visualizer.add_cylinder(start=part_center_w, end=link_pos_w, radius=line_radius, color=red)
visualizer.add_cylinder(start=part_center_w, end=gt_end_w, radius=line_radius, color=blue)
```

颜色建议保持源语义：

```python
CONTACT_POINT_COLOR = (1.0, 0.85, 0.1, 1.0)
PART_CENTER_COLOR = (1.0, 0.35, 0.05, 1.0)
REALTIME_VECTOR_COLOR = (1.0, 0.05, 0.05, 1.0)
GT_VECTOR_COLOR = (0.05, 0.35, 1.0, 1.0)
```

### 4.6 性能和容量控制

源代码通过 `debug_vis_max_envs` 控制最多画几个 env。目标还应该额外使用：

```python
env_ids = list(visualizer.get_env_indices(env.num_envs))
```

建议最终绘制 env 数：

```text
selected_env_ids = env_ids[:debug_vis_max_envs]
```

不要默认画全部 env 的所有 points / vectors。Part2Link pair 数虽然不大，但多个 env + part points + vectors 在 native viewer 下仍然会拖慢 play。

## 5. Legacy Sparse Contact Reward Markers

### 5.1 源文件

```text
source/instinctlab/instinctlab/tasks/interaction/mdp/sparse_contact_reward.py
source/instinctlab/instinctlab/tasks/interaction/mdp/contact_geometry.py
source/instinctlab/instinctlab/tasks/interaction/config/g1/g1_interaction_sitting_part2link_shadowing_cfg.py
```

这套是旧 sparse contact reward 的 debug visualizer，当前 sitting Part2Link cfg 中相关 reward 基本是注释备用。

### 5.2 源行为

Debug params：

```python
debug_vis
debug_vis_max_envs
debug_vis_show_all_points
debug_vis_show_nearest
debug_vis_point_radius
debug_vis_nearest_point_radius
debug_vis_arrow_length_scale
debug_vis_arrow_thickness_scale
debug_vis_nearest_arrow_thickness_scale
debug_vis_part_names
debug_vis_distance_bucket_thresholds
```

Marker 语义：

- all mandatory points：blue-ish sphere。
- all arrows：blue arrow from link to target points。
- nearest points：按距离 bucket 着色：
  - near: green
  - mid: yellow
  - far: red
- nearest arrows：同样按 bucket 着色。

源 helper：

```python
extract_selected_contact_debug_data(...)
extract_mandatory_contact_debug_data(...)
bucketize_distance_visualization(...)
resolve_direction_to_arrow_marker(...)
get_visualizer_default_scale(...)
```

### 5.3 迁移建议

v1 不建议迁移这套，原因：

- 当前 Part2Link 使用的是 `part2link_vector_guidance_gauss`，不是旧 `SparseContactReward`。
- 旧 reward 的可视化和旧 JSON metadata / hardcoded link mapping 耦合更强。
- 如果迁移，会把不再作为 authority 的 sparse contact JSON 逻辑带回目标 repo。

保留文档级语义即可。若未来需要 nearest-point contact debug，可以复用它的颜色和 bucket 语义，但数据源应从 NPZ metadata / active chair runtime state 走。

## 6. Direct Replay Quality Monitor

### 6.1 源文件

```text
scripts/direct_replay_npz_quality_monitor.py
```

这是一个独立 replay/diagnostic 脚本，不是训练 env 的一部分。

### 6.2 源可视化内容

脚本构建单 env scene，逐帧写入：

- robot root pose / velocity
- robot joint state
- replay object root pose / velocity

然后每帧画：

| Marker | 含义 |
| --- | --- |
| `objectContactState` | object contact scalar，未接触黄色，接触蓝色 |
| `sparseContact` | part centers / active parts / sparse robot link positions |
| `sparseContactVectors` | relation vectors、active vectors、stored center vectors |

核心 helper：

```python
_create_contact_state_marker()
_create_sparse_contact_marker()
_create_contact_vector_marker()
_line_marker_pose()
_object_local_points_to_world()
_object_local_vectors_to_world()
_contact_vector_marker_inputs()
_remap_motion_joints_to_robot_order()
```

脚本还会打印 quality monitor 指标：

- robot base position error
- robot base quaternion error
- joint position max error
- object position error
- object quaternion error

### 6.3 迁移建议

建议迁移为 InstinctMJ 独立脚本，而不是塞进训练 task：

```text
src/instinct_mj/scripts/direct_replay_part2link_quality_monitor.py
```

迁移时需要替换：

| IsaacLab source | InstinctMJ target |
| --- | --- |
| `SimulationContext` | `mjlab.sim.Simulation` / existing InstinctMJ env or scene script style |
| `InteractiveScene` | `InstinctScene` / `SceneCfg` |
| `Articulation.write_*_to_sim` | InstinctMJ entity write APIs |
| `RigidObject.write_*_to_sim` | InstinctMJ entity write APIs |
| `VisualizationMarkers` | `DebugVisualizer` or native viewer overlay primitives |

如果只想尽快检查 Part2Link NPZ 与 object pose，先迁移数据打印和 joint reorder；marker overlay 可以第二步补。

## 7. Play / Train 脚本中的可视化开关

### 7.1 源行为

源 train/play 共同支持：

```text
--visualize_camera
```

相关逻辑：

- 如果 task 是 sitting / Part2Link / Transformer Part2Link，自动 `enable_cameras=True`。
- `--visualize_camera` 打开 camera cfg 和 observation params 中的 debug flags。
- train 里还会同时打开 Part2Link reward debug:

```python
reward_params["debug_vis"] = True
reward_params.setdefault("debug_vis_max_envs", 1)
reward_params.setdefault("debug_vis_show_contact_points", True)
reward_params.setdefault("debug_vis_show_part_centers", True)
reward_params["debug_vis_ignore_contact_phase"] = False
```

Play cfg 则默认打开 camera depth image 和 Part2Link reward debug。

### 7.2 目标行为

目标 InstinctMJ play/train 采用 mjlab viewer：

```text
src/instinct_mj/scripts/instinct_rl/play.py
src/instinct_mj/scripts/instinct_rl/train.py
```

关键差异：

- viewer backend 是 `native` / `viser` / `none`。
- headless 或非-native viewer 会关闭 GUI-dependent debug visualization。
- play cfg 通过 `ViewerConfig` 控制 camera：

```python
ViewerConfig(
    lookat=(0.0, 0.0, 0.0),
    distance=2.1213,
    elevation=45.0,
    azimuth=0.0,
    origin_type=ViewerConfig.OriginType.ASSET_ROOT,
    entity_name="robot",
)
```

### 7.3 迁移建议

目标不一定要复刻 `--visualize_camera` 的 CLI 行为。更推荐：

- Play cfg 默认打开视觉 debug。
- Train 默认关闭视觉 debug。
- 如果需要训练时开 viewer，用已有：

```bash
uv run instinct-train <task> --viewer native
```

- 如果需要 play 可视化，用：

```bash
uv run instinct-play <task> --viewer native
```

Part2Link marker 完成迁移后，可在 play cfg 中保留：

```python
self.rewards["part2link_vector_guidance_gauss"].params["debug_vis"] = True
```

但实际绘制应由 monitor/sensor debug visualizer 完成。

## 8. 推荐迁移顺序

### Step 1: 确认可用的目标可视化入口

确认这些目标文件已存在并可用：

```text
src/instinct_mj/envs/manager_based_rl_env.py
src/instinct_mj/monitors/monitor_manager.py
src/instinct_mj/visualization/marker_cfg.py
src/instinct_mj/motion_reference/motion_reference_manager.py
src/instinct_mj/scripts/instinct_rl/play.py
```

成功标准：

- native viewer 可以启动。
- `MotionReferenceManager.debug_vis(...)` 可以画 reference links。
- headless 模式下不会尝试打开 GUI debug。

### Step 2: 补 Part2Link debug cache

在目标：

```text
src/instinct_mj/tasks/interaction/mdp/part2link.py
```

给 `part2link_vector_guidance_gauss(...)` 增加这些 debug params：

```python
debug_vis_max_envs: int = 1
debug_vis_show_contact_points: bool = True
debug_vis_show_part_centers: bool = True
debug_vis_point_radius: float = 0.025
debug_vis_center_radius: float = 0.035
debug_vis_line_radius: float = 0.008
debug_vis_ignore_contact_phase: bool = False
```

当 `debug_vis=True` 时，把必要 tensors 存到 `env._part2link_debug_cache`。

注意：目标 reward 已经改成 NPZ metadata authority，不要重新引入源里的 `link_names` 参数。

### Step 3: 新增 Part2Link debug monitor

新增一个 monitor term，例如：

```python
class Part2LinkDebugVisualizer(MonitorTerm):
    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        ...
```

或者根据目标 monitor API 写成当前 repo 风格的 monitor term。它只负责：

- 读取 `env._part2link_debug_cache`。
- 选取 `visualizer.get_env_indices(...)` 与 `debug_vis_max_envs`。
- 画 spheres 和 cylinders。

### Step 4: 接入 Part2Link PLAY cfg

在：

```text
src/instinct_mj/tasks/interaction/config/g1/g1_interaction_sitting_part2link_shadowing_cfg.py
```

Play cfg 中保留 / 增加：

```python
self.rewards["part2link_vector_guidance_gauss"].params["debug_vis"] = True
self.rewards["part2link_vector_guidance_gauss"].params["debug_vis_max_envs"] = 1
self.rewards["part2link_vector_guidance_gauss"].params["debug_vis_show_contact_points"] = True
self.rewards["part2link_vector_guidance_gauss"].params["debug_vis_show_part_centers"] = True
self.rewards["part2link_vector_guidance_gauss"].params["debug_vis_ignore_contact_phase"] = False
```

并添加 monitor term 到 `monitors` dict。

### Step 5: 只在需要时迁移 direct replay

如果 runtime smoke 通过后还需要离线诊断，再迁移：

```text
scripts/direct_replay_npz_quality_monitor.py
```

建议目标脚本先做：

- NPZ load。
- joint reorder。
- object pose replay。
- quality metrics。

然后再补 contact state / sparse contact / vector markers。

## 9. 验证清单

### Motion Reference

```bash
uv run instinct-play Instinct-Interaction-Sitting-Part2Link-Transformer-G1-Play-v0 --viewer native --agent random
```

检查：

- reference robot 或 reference link markers 可见。
- markers 与当前 motion frame 对齐。
- `visualizing_robot_offset` 不导致 reference 和 policy robot 重叠。

### Camera

检查：

- native viewer 中 camera sensor 能正常更新。
- depth observation debug image 不阻塞 simulation。
- headless / `viewer none` 下 debug flags 被关闭。

### Part2Link Markers

检查：

- yellow contact points 在 active chair part surface 附近。
- orange part centers 在 seat/back/armrest 中心附近。
- red realtime vectors 从 part center 指向当前 robot sparse bodies。
- blue GT vectors 从 part center 指向 motion sparse GT endpoint。
- `debug_vis_ignore_contact_phase=True` 时能看到更多 relation pairs。
- `debug_vis_max_envs=1` 时只画一个 env。

### Direct Replay

检查：

- frame 0 object pose 与 NPZ 一致。
- robot joint reorder 与 target MJCF native order 一致。
- contact state marker 跟 `object_contact` frame phase 对齐。
- sparse vectors 与 reward debug markers 语义一致。

## 10. 不要迁移的内容

这些内容不建议迁移到 InstinctMJ：

- IsaacLab / Omniverse app launcher 逻辑。
- USD `prim_path` 可视化组织方式。
- `isaaclab.markers.VisualizationMarkers` 生命周期管理。
- `isaaclab.sensors.camera.utils.create_pointcloud_from_depth` 直接 import。
- 旧 sparse contact reward 中的 hardcoded `link_name_map` 作为新 Part2Link reward authority。
- 在 reward 函数中直接依赖 viewer object。

## 11. 当前目标 repo 缺口

截至本文档生成时，InstinctMJ 中已有：

- motion reference debug visualizer。
- native/viser/offscreen viewer flow。
- image debug observation。
- Part2Link cfg 中的 `debug_vis` 开关。

仍缺：

- Part2Link reward debug cache。
- Part2Link debug monitor / visualizer term。
- Part2Link points/vectors 的 `DebugVisualizer` 绘制实现。
- 可选 direct replay quality monitor 迁移。

