# MJLab 运行时逻辑完整分析

> 基于 [InstinctMJ](https://github.com/project-instinct/InstinctMJ) 源码，参照 IsaacLab 概念体系撰写。
> 分析时间：2026-06-25

---

## 1. 架构全景（与 IsaacLab 对应）

| IsaacLab | mjlab / InstinctMJ |
|----------|---------------------|
| Hydra + `@configclass` | Python `@dataclass`（无 YAML） |
| `SimConfig` + `PhysxCfg` | `SimulationCfg` + `MujocoCfg` |
| `InteractiveScene` | `Scene` + `SceneCfg` |
| `ArticulationView` / `RigidObjectView` | `Entity` + `EntityData`（WarpBridge 张量） |
| `ManagerBasedRLEnv` | `ManagerBasedRlEnv` |
| `ManagerBase` / `ManagerTerm` | `ManagerBase` / `ManagerTermBase` |
| `ActionManager` / `ObservationManager` 等 | 同名 managers |
| `EventManager` (mode based) | 同名 + `startup` / `reset` / `interval` / `step` 四模式 |
| `DirectRLEnv` | （无等价物；仅 ManagerBased 模式） |
| RSL-RL `OnPolicyRunner` | `instinct_rl` `OnPolicyRunner` |
| VecEnv wrapper | `InstinctRlVecEnvWrapper` |
| Task registry (gymnasium) | 显式 `_REGISTRY` dict |

核心架构遵循 **Manager-Based RL** 模式：

```
环境 = Scene（物理场景） + Simulation（物理引擎） + 一组 Manager（职责分解）
```

---

## 2. 配置系统（纯 Python Dataclass，非 Hydra）

与 IsaacLab 使用 Hydra YAML 不同，mjlab 采用**纯 Python dataclass 配置**。

### 2.1 环境总配置 — `ManagerBasedRlEnvCfg`

```python
@dataclass(kw_only=True)
class ManagerBasedRlEnvCfg:
    decimation: int                              # 每个策略步的物理步数
    scene: SceneCfg                              # 场景（实体、地形、传感器）
    observations: dict[str, ObservationGroupCfg] # 观察组
    actions: dict[str, ActionTermCfg]            # 动作项
    rewards: dict[str, RewardTermCfg]            # 奖励项
    terminations: dict[str, TerminationTermCfg]  # 终止条件
    commands: dict[str, CommandTermCfg]          # 指令生成
    events: dict[str, EventTermCfg]              # 事件（域随机化 + 重置）
    curriculum: dict[str, CurriculumTermCfg]     # 课程学习
    metrics: dict[str, MetricsTermCfg]           # 指标记录
    recorders: dict[str, RecorderTermCfg]        # 数据录制
    sim: SimulationCfg                           # 物理仿真配置
    episode_length_s: float                      # Episode 时长
    is_finite_horizon: bool                      # 有限/无限视界
    auto_reset: bool                             # 自动重置
    scale_rewards_by_dt: bool                    # 奖励时间缩放
```

### 2.2 InstinctMJ 扩展 — `InstinctLabRLEnvCfg`

```python
@dataclass
class InstinctLabRLEnvCfg(ManagerBasedRlEnvCfg):
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    monitors: object | None = None   # MonitorManager 配置
```

### 2.3 任务注册系统

位于 [`src/instinct_mj/tasks/registry.py`](../src/instinct_mj/tasks/registry.py)：

```python
_REGISTRY: dict[str, _TaskCfg] = {}

def register_instinct_task(
    task_id: str,
    env_cfg_factory: Callable[[], ManagerBasedRlEnvCfg],
    play_env_cfg_factory: Callable[[], ManagerBasedRlEnvCfg],
    instinct_rl_cfg_factory: Callable[[], InstinctRlOnPolicyRunnerCfg],
    runner_cls: type | None = None,
) -> None:
    _REGISTRY[task_id] = _TaskCfg(...)
```

每个任务注册时提供**三个工厂函数**：

| 工厂函数 | 用途 |
|----------|------|
| `env_cfg_factory` | 构建训练用环境配置 |
| `play_env_cfg_factory` | 构建推理回放用环境配置 |
| `instinct_rl_cfg_factory` | 构建 PPO 算法配置 |

**已注册的 16 个任务：**

| 任务 ID | 系列 |
|---------|------|
| `Instinct-Locomotion-Flat-G1-v0` | Locomotion |
| `Instinct-Locomotion-Flat-G1-Play-v0` | Locomotion (play) |
| `Instinct-BeyondMimic-Plane-G1-v0` | Shadowing / BeyondMimic |
| `Instinct-BeyondMimic-Plane-G1-Play-v0` | Shadowing / BeyondMimic (play) |
| `Instinct-Shadowing-WholeBody-Plane-G1-v0` | Shadowing / WholeBody |
| `Instinct-Shadowing-WholeBody-Plane-G1-Play-v0` | Shadowing / WholeBody (play) |
| `Instinct-Perceptive-Shadowing-G1-v0` | Shadowing / Perceptive |
| `Instinct-Perceptive-Shadowing-G1-Play-v0` | Shadowing / Perceptive (play) |
| `Instinct-Perceptive-Shadowing-G1-OneMotion-v0` | Shadowing / Perceptive (OneMotion) |
| `Instinct-Perceptive-Shadowing-G1-OneMotion-Play-v0` | Shadowing / Perceptive (OneMotion play) |
| `Instinct-Perceptive-Vae-G1-v0` | Shadowing / Perceptive (VAE) |
| `Instinct-Perceptive-Vae-G1-Play-v0` | Shadowing / Perceptive (VAE play) |
| `Instinct-Perceptive-HOI-Shadowing-G1-v0` | Shadowing / Perceptive HOI |
| `Instinct-Perceptive-HOI-Shadowing-G1-Play-v0` | Shadowing / Perceptive HOI (play) |
| `Instinct-Parkour-Target-Amp-G1-v0` | Parkour (AMP) |
| `Instinct-Parkour-Target-Amp-G1-Play-v0` | Parkour (AMP play) |

---

## 3. 场景与仿真构建

### 3.1 Scene 组装流程

```python
# 1. 创建 Scene — 基于 scene.xml 模板组装 MJCF spec
self.scene = Scene(self.cfg.scene, device=device)
#   ├── _add_terrain()     → TerrainEntity（平面 / 高度场 / 三角网格）
#   ├── _add_entities()    → Entity（机器人、物体），设置 keyframes，挂载到 worldbody
#   └── _add_sensors()     → Sensor（接触力、IMU、相机、射线投射）

# 2. 编译 spec → 创建 Simulation
self.sim = Simulation(
    num_envs=self.scene.num_envs,
    cfg=self.cfg.sim,
    spec=self.scene.spec,
    variant_info=self.scene.collect_variant_info(),
    device=device,
)

# 3. 后编译初始化
self.scene.initialize(
    mj_model=self.sim.mj_model,
    model=self.sim.model,
    data=self.sim.data,
)
#   → 为每个 Entity 分配 EntityData（所有状态张量）
#   → 创建 SensorContext（如需传感器）
#   → 将 sensor_context 注入 sim.set_sensor_context()

# 4. 扩展模型字段用于域随机化
self.sim.expand_model_fields(event_manager.domain_randomization_fields)
```

### 3.2 InstinctMJ 自定义 Scene

[`src/instinct_mj/envs/scene.py`](../src/instinct_mj/envs/scene.py) 覆盖地形添加逻辑：

```python
class InstinctScene(Scene):
    def _add_terrain(self):
        # 根据 cfg.class_type 选择正确的 TerrainImporter
        if isinstance(terrain_cfg, TerrainImporterCfg):
            terrain = terrain_cfg.class_type(terrain_cfg, device=self._device)
        else:
            terrain = TerrainEntity(terrain_cfg, device=self._device)
```

### 3.3 关键差异：GPU 加速的 MuJoCo（MJWarp）

mjlab 在 **MuJoCo Warp** 之上运行，所有 `num_envs` 个环境在一次 kernel 启动中**批量**完成物理步进：

| 概念 | 类型 | 说明 |
|------|------|------|
| `sim.mj_model` | `mjModel` | 原生 MuJoCo 模型（1 个，所有环境共享） |
| `sim.mj_data` | `mjData` | 原生 MuJoCo 数据（1 个，所有环境共享） |
| `sim.model` | `WarpBridge` | **向量化** Warp 模型张量，形状 `(num_envs, ...)` |
| `sim.data` | `WarpBridge` | **向量化** Warp 数据张量，形状 `(num_envs, ...)` |
| `sim.wp_model` | `wp.Model` | Warp 级模型 |
| `sim.wp_data` | `wp.Data` | Warp 级数据 |

---

## 4. Manager 加载顺序（关键约束）

Manager 的加载顺序**不能改变**，因为它们之间存在严格的依赖关系：

```
load_managers() {
    1. EventManager       ← 首先加载，收集域随机化字段
    2. sim.expand_model_fields()  ← 基于 EventManager 信息分配 GPU 内存
    3. CommandManager     ← 必须早于 ObservationManager（观察可能引用指令）
    4. ActionManager      ← 定义动作空间
    5. ObservationManager ← 定义观察空间
    6. TerminationManager ← 依赖场景状态
    7. RewardManager      ← 依赖终止信息
    8. CurriculumManager  ← 可调整其他 Manager 参数
    9. MetricsManager     ← 读取所有其他 Manager 的状态
   10. RecorderManager    ← 录制任意数据
   11. _configure_gym_env_spaces()  ← 构建 Gymnasium 观察/动作空间
   12. event_manager.apply("startup")  ← 触发启动时的域随机化
}
```

### InstinctMJ 的 Manager 替换

```python
# InstinctRlEnv.load_managers() 覆盖：
def load_managers(self):
    # 1. 将扁平奖励 dict 包装为 MultiRewardCfg，支持分组奖励
    reward_group_cfg = self._as_multi_reward_cfg(self.cfg.rewards)
    super().load_managers()

    # 2. 用 MultiRewardManager 替换标准 RewardManager
    #    输出 → dict[str, Tensor] 而非单个 Tensor
    #    日志 → Episode_Reward/rewards_<term>/{sum,timestep}
    if reward_group_cfg is not None:
        self.reward_manager = MultiRewardManager(
            self.cfg.rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
        )

    # 3. 创建 MonitorManager（step/episode 级别的自定义日志）
    self.monitor_manager = MonitorManager(self.cfg.monitors, self)
```

---

## 5. `step()` 生命周期 — 核心运行时循环

这是整个系统最重要的方法。

```python
def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
    # 返回: (obs, reward, terminated, truncated, extras)
```

### 5.1 阶段一：动作处理 + 物理步进（Decimation 循环）

```
action [num_envs, action_dim]
│
├── action_manager.process_action(action)
│   ├── 将动作按 term 维度切片
│   ├── 保存历史: prev_prev ← prev ← current ← new
│   └── 每个 ActionTerm.process_actions(slice)
│       └── 应用 scale + offset → entity.set_joint_position_target()
│
└── FOR i IN range(decimation):              ← 典型值 4 次
    ├── action_manager.apply_action()
    │   └── entity.write_data_to_sim()       ← 将目标写入 sim.data.ctrl
    ├── scene.write_data_to_sim()            ← 所有实体的根状态 + 关节状态写入 sim
    ├── sim.step()                           ← mj_step:
    │                                          前向运动学 → 积分 → 碰撞检测
    │                                          (num_envs 个环境并行 GPU 步进)
    ├── scene.update(dt=physics_dt)          ← 从 sim 读回状态
    └── metrics_manager.compute_substep()     ← 子步指标记录
```

**关键点**：与 IsaacLab 每个 manager 步进时只调用一次 `apply_action()` 不同，mjlab 的 `apply_action()` 在**每个 decimation 子步**都执行一次。

### 5.2 阶段二：终止检查 + 奖励计算

```
episode_length_buf += 1

termination_manager.compute()
├── time_out()            → 检查 episode 是否超时
├── illegal_contact()     → 检查非法接触（躯干、手臂触地）
├── bad_orientation()     → 检查躯干角度是否超出阈值
├── root_height_below_minimum()  → 检查根部高度
├── nan_detection()       → 检测 NaN 状态
└── ...
→ reset_buf: bool[N]       哪些环境需要重置
→ reset_terminated: bool[N]  真正的终止（失败）
→ reset_time_outs: bool[N]   时间截断

reward_manager.compute(dt=step_dt)
├── 每个 RewardTerm 计算加权奖励
├── track_lin_vel_xy_yaw_frame_exp()  → 速度跟踪
├── track_ang_vel_z_world_exp()       → 角速度跟踪
├── feet_air_time_positive_biped()    → 足部腾空时间
├── contact_slide()                   → 足部滑动惩罚
├── flat_orientation_l2()             → 躯干姿态惩罚
├── joint_torques_l2()                → 关节力矩惩罚
├── action_rate_l2()                  → 动作平滑惩罚
├── dof_acc_l2()                      → 关节加速度惩罚
├── stand_still()                     → 静止时惩罚多余动作
├── is_terminated()                   → 终止惩罚（负权重）
└── ...
→ reward_buf [num_envs, num_reward_terms]
```

**注意**：此时派生量（xpos、xquat 等）滞后一个物理子步。MuJoCo 的 `mj_step` 在积分*之前*运行 `mj_forward`，因此经过最后的物理子步后，派生量（xpos、xquat、site_xpos、cvel、sensordata）滞后 `qpos`/`qvel` 一个子步。对于奖励塑形和终止检查而言，这种滞后可忽略不计，且**对所有环境和所有步长保持一致**，因此 MDP 是良定义的。

### 5.3 阶段三：重置已终止环境

```
IF auto_reset AND any(reset_buf):
    reset_env_ids = reset_buf.nonzero().squeeze(-1)

    recorder_manager.record_pre_reset(reset_env_ids)

    _reset_idx(reset_env_ids)
    │
    ├── curriculum_manager.compute(env_ids)
    │   └── 根据训练进度更新环境参数（如地形难度、采样权重）
    │
    ├── sim.reset(env_ids)
    │   └── 将 qpos 和 qvel 重置为 qpos0
    │
    ├── scene.reset(env_ids)
    │   └── 将实体状态重置为初始值（root state + joint state）
    │
    ├── event_manager.apply("reset", env_ids, global_env_step_count)
    │   ├── reset_root_state_uniform()      ← 随机化初始位置和朝向
    │   ├── reset_joints_by_scale()         ← 随机缩放默认关节角度
    │   ├── apply_external_force_torque()   ← 施加随机初始力/力矩
    │   └── push_by_setting_velocity()      ← 随机初始速度扰动
    │
    ├── [各 Manager 的 reset(env_ids)]
    │   ├── observation_manager.reset()     ← 清除历史缓冲区和延迟缓冲区
    │   ├── action_manager.reset()          ← 清零动作历史
    │   ├── reward_manager.reset()          ← 清零累计奖励
    │   ├── command_manager.reset()         ← 重新采样指令
    │   ├── curriculum_manager.reset()      ← 记录课程状态
    │   ├── event_manager.reset()           ← 重新采样 interval 事件的定时器
    │   ├── termination_manager.reset()     ← 清零终止状态
    │   └── metrics_manager.reset()         ← 清零指标累计值
    │
    └── episode_length_buf[env_ids] = 0
```

### 5.4 阶段四：后重置计算

```
sim.forward()
#   单次 mj_forward 调用刷新所有环境的派生量。
#   对于未重置环境：解析 decimation 循环后的运动学子步滞后。
#   对于已重置环境：获取新写入的重置状态的派生量。
#   这一"一次调用覆盖两种情况"的设计避免了 forward() 被调用两次。

command_manager.compute(dt=step_dt)
├── UniformVelocityCommand
│   ├── 如果距上次采样时间 >= resampling_time_range，重新采样目标速度
│   ├── lin_vel_x, lin_vel_y, ang_vel_z, heading
│   └── rel_standing_envs=0.2: 20% 环境指定为"静止"指令
├── MotionCommand（Shadowing 任务）
│   └── 从参考运动缓冲区读取下一帧姿态
└── ...

event_manager.apply("step", dt=step_dt)
└── 应用每步持续力（如 apply_body_impulse 生命周期管理）

event_manager.apply("interval", dt=step_dt)
├── 检查每个环境的定时器是否超时（不满足 1e-6 阈值）
├── 对已超时的环境：
│   ├── 应用事件函数（如 push_by_setting_velocity）
│   └── 从 interval_range_s 重新采样下一个触发时间
└── is_global_time: 如果为 True，所有环境共享同一计时器

sim.sense()
#   读取所有环境的传感器数据（sensordata），包括：
#   - 接触力传感器（contact forces + found flag）
#   - IMU（加速度计 + 陀螺仪）
#   - 相机（RGB、深度、分割）
#   - 射线投射器（深度距离）

observation_manager.compute(update_history=True)
│
├── FOR EACH group ("policy", "critic"):
│   └── FOR EACH term:
│       ├── term.func(env)                    ← 调用观察函数
│       │   ├── base_lin_vel()                → (num_envs, 3)
│       │   ├── base_ang_vel()                → (num_envs, 3)
│       │   ├── projected_gravity()           → (num_envs, 3)
│       │   ├── joint_pos_rel()               → (num_envs, 29)
│       │   ├── joint_vel()                   → (num_envs, 29)
│       │   ├── last_action()                 → (num_envs, action_dim)
│       │   ├── generated_commands()          → (num_envs, cmd_dim)
│       │   ├── body_pos()                    → 身体部件位置
│       │   ├── contact_forces()              → 接触力历史
│       │   └── ...
│       ├── noise.apply()                     ← 添加噪声（仅策略组，critic 组保持清洁）
│       ├── clip(min, max)                    ← 数值裁剪
│       ├── scale * obs                       ← 缩放到合理范围
│       ├── delay_buffer.append(obs)          ← 延迟缓冲（模拟传感器延迟，以步为单位）
│       │   └── delay_buffer.compute()        ← 返回延迟后的值
│       ├── history_buffer.append(obs)        ← 循环缓冲（时间上下文堆叠）
│       │   └── 展平或保持 (T, dim) 形状
│       └── NaN 检查（按 group 配置的策略：
│           disabled / warn / sanitize / error）
│
│   最终输出:
│   obs = {
│       "policy": {
│           "base_ang_vel": Tensor(N, 3),
│           "projected_gravity": Tensor(N, 3),
│           "velocity_commands": Tensor(N, 4),
│           "joint_pos": Tensor(N, 29),
│           "joint_vel": Tensor(N, 29),
│           "actions": Tensor(N, act_dim),
│       },
│       "critic": { ... 类似但无噪声，可能包含特权信息 ... },
│   }

recorder_manager.record_post_step()
#   录制任意指定的数据（observations、actions 等）用于离线分析

RETURN (obs, reward_buf, terminated, truncated, extras)
```

### 5.5 InstinctMJ 的 step() 扩展

```python
# InstinctRlEnv.step() 在基类 step() 之上添加：
def step(self, action):
    obs, reward, terminated, truncated, extras = super().step(action)
    monitor_infos = self.monitor_manager.update(dt=self.step_dt)
    extras.setdefault("step", {})
    extras["step"].update(monitor_infos)
    return obs, reward, terminated, truncated, extras
```

`MonitorManager` 提供自定义的逐步（step）和逐回合（episode）指标记录，如关节统计数据、身体统计数据、shadowing 指标等。

---

## 6. `reset()` 生命周期 — 初始启动

```
reset(seed=None, env_ids=None, options=None)
│
├── IF seed is not None:
│   └── self.seed(seed)  → np.random + torch.manual_seed
│
├── IF env_ids is None:
│   └── env_ids = all environments [0, num_envs)
│
├── _reset_idx(env_ids)               ← 与 step 内相同的重置逻辑
│   ├── curriculum_manager.compute(env_ids)
│   ├── sim.reset(env_ids)
│   ├── scene.reset(env_ids)
│   ├── event_manager.apply("reset", env_ids)
│   └── [各 Manager 的 reset(env_ids)]
│
├── scene.write_data_to_sim()         ← 将重置状态写入 sim
├── sim.forward()                     ← 计算派生量
├── command_manager.compute(dt=0.0)   ← 首次采样指令
├── sim.sense()                       ← 传感器读取
├── obs = observation_manager.compute(update_history=True)
├── recorder_manager.record_post_reset(env_ids)
│
└── RETURN (obs, extras)
```

---

## 7. 观察处理管道（Observation Pipeline）

每个观察项经过以下可配置的处理管道：

```
compute() → noise → clip → scale → delay → history → (concat)
```

### 7.1 各阶段详解

| 阶段 | 机制 | 配置 |
|------|------|------|
| **Noise**（噪声） | 仅在 `enable_corruption=True` 时应用于策略组（训练期间），critic 组保持清洁 | `UniformNoiseCfg(n_min, n_max)` 或自定义 `NoiseModelCfg` |
| **Clip**（裁剪） | 将值钳制到 `[min, max]` 范围 | `clip: tuple[float, float]` |
| **Scale**（缩放） | 乘以缩放因子以归一化到合理范围 | `scale: float` 或 `tuple[float, ...]` |
| **Delay**（延迟） | `DelayBuffer` 模拟传感器延迟（以步为单位），每个环境独立采样滞后期 | `delay_min_lag` / `delay_max_lag` / `delay_per_env` |
| **History**（历史） | `CircularBuffer` 提供时间上下文（如堆叠 3 帧） | `history_length` / `flatten_history_dim` |
| **NaN 策略** | 对每组可配置为 disabled / warn / sanitize / error | `nan_policy` / `nan_check_per_term` |

### 7.2 两个观察组的意义

- **策略组（"policy"）**：有噪声、干净的观察用于策略网络推理
- **Critic 组（"critic"）**：无噪声、可能包含特权信息（如真值线速度）、无历史延迟，用于价值函数估计

这种设计与 IsaacLab 的 `actor_obs` / `critic_obs` 分离完全一致。

---

## 8. 训练管道

```
CLI: instinct-train <task_id> [--agent.* --env.*]
│
├── 1. tyro 解析 task_id（从已注册列表中精确匹配）
├── 2. TrainConfig.from_task(task_id)
│   ├── load_env_cfg(task_id)         ← 从注册表获取 env_cfg_factory()
│   └── load_instinct_rl_cfg(task_id) ← 返回 InstinctRlOnPolicyRunnerCfg
│
├── 3. CLI 覆盖（点标记法）
│   示例: --agent.max_iterations 10000
│         --env.scene.num_envs 2048
│         --env.sim.mujoco.timestep 0.004
│         --agent.algorithm.learning_rate 5.0e-4
│
├── 4. 设备 / 种子 / 分布式解析
│   ├── 读取 LOCAL_RANK, RANK, WORLD_SIZE 环境变量
│   ├── 启用 Warp kernel 缓存隔离（多 rank 场景避免并发 JIT 编译冲突）
│   ├── wp.set_device(device)  ← 确保传感器 kernel 在正确的 GPU 上运行
│   └── torch.distributed.init_process_group(backend="nccl")
│
├── 5. 运动文件解析（Tracking 任务特有）
│   ├── 从命令行 --motion-file 或
│   ├── 从 WandB artifact --registry-name 或
│   └── 从 env config 中已配置的路径
│   └── validate_tracking_motion_file()
│
├── 6. 环境创建
│   ├── env = InstinctRlEnv(
│   │     cfg=cfg.env,
│   │     device=device,
│   │     render_mode="rgb_array" (if video) else None,
│   │   )
│   ├── [可选] env = VideoRecorder(env, ...)   ← 训练视频录制
│   └── vec_env = InstinctRlVecEnvWrapper(
│           env,
│           policy_group="policy",
│           critic_group="critic",
│       )
│       └── 在 __init__ 中调用 env.reset()  ← runner 不会在 rollout 前调用 reset
│
├── 7. Runner 创建
│   ├── runner = OnPolicyRunner(
│   │     vec_env,
│   │     agent_cfg_dict,        ← InstinctRlOnPolicyRunnerCfg.to_dict()
│   │     log_dir=str(log_dir),
│   │     device=device,
│   │   )
│   ├── runner.add_git_repo_to_log()  ← 记录 git diff 用于可复现性
│   └── [可选] runner.load(resume_path)  ← 从 checkpoint 恢复
│
├── 8. 训练循环: runner.learn(num_learning_iterations, init_at_random_ep_len=True)
│   │
│   ├── FOR iteration IN range(max_iterations):
│   │   │
│   │   ├── rollout_step()  × num_steps_per_env  (默认 24)
│   │   │   ├── obs_dict = observation_manager.compute()
│   │   │   ├── packed_obs = _pack_observations(obs_dict)
│   │   │   │   → 展平每组: [base_ang_vel(3), proj_grav(3), ...]
│   │   │   │   → policy_tensor (num_envs, policy_dim)
│   │   │   │   → critic_tensor (num_envs, critic_dim)
│   │   │   ├── actions, values, log_probs = policy(packed_obs)
│   │   │   ├── next_obs_packed, rewards, dones, extras = vec_env.step(actions)
│   │   │   └── 存储到 rollout buffer (obs, actions, rewards, dones, values, log_probs)
│   │   │
│   │   ├── compute_returns()
│   │   │   └── GAE (Generalized Advantage Estimation)
│   │   │       gamma=0.99, lam=0.95
│   │   │
│   │   ├── update()  × num_learning_epochs  (默认 5)
│   │   │   ├── num_mini_batches 次 (默认 4)
│   │   │   ├── PPO clip loss (clip_param=0.2)
│   │   │   ├── value loss (value_loss_coef=1.0)
│   │   │   ├── entropy bonus (entropy_coef=0.005)
│   │   │   ├── [可选] KL loss（蒸馏）
│   │   │   ├── [可选] discriminator loss（AMP）
│   │   │   ├── gradient clipping (max_grad_norm=1.0)
│   │   │   └── AdamW optimizer (learning_rate=1.0e-3, adaptive schedule)
│   │   │
│   │   └── IF iteration % save_interval == 0:
│   │       └── save checkpoint → logs/instinct_rl/<experiment>/<timestamp>/model_<iter>.pt
│   │
│   └── [信号处理: SIGINT / SIGTERM / SIGQUIT → 优雅退出]
│
└── 9. 清理
    ├── viewer.close()
    ├── vec_env.close()
    └── dist.destroy_process_group()
```

### 8.1 InstinctRlVecEnvWrapper — 关键适配器

[`src/instinct_mj/rl/vecenv_wrapper.py`](../src/instinct_mj/rl/vecenv_wrapper.py) 将 `ManagerBasedRlEnv` 适配为 `instinct_rl.VecEnv` 契约：

```python
class InstinctRlVecEnvWrapper(VecEnv):
    def step(self, actions: torch.Tensor):
        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
        packed_obs = self._pack_observations(obs_dict)
        dones = (terminated | truncated).long()

        # 多奖励处理：MultiRewardManager 返回 dict[str, Tensor]
        # → 堆叠为 (num_envs, num_rewards)
        if isinstance(rewards, dict):
            rewards = torch.stack(list(rewards.values()), dim=-1)

        return packed_obs["policy"], rewards, dones, extras
```

### 8.2 策略架构配置

[`src/instinct_mj/rl/config.py`](../src/instinct_mj/rl/config.py) 定义了复合策略类型：

| 配置类 | class_name | 用途 |
|--------|-----------|------|
| `InstinctRlActorCriticCfg` | `"ActorCritic"` | 标准 MLP actor-critic |
| `InstinctRlActorCriticRecurrentCfg` | `"ActorCriticRecurrent"` | 带 GRU RNN |
| `InstinctRlMoEActorCriticCfg` | `"MoEActorCritic"` | Mixture of Experts |
| `InstinctRlVaeActorCriticCfg` | `"VaeActor"` | VAE actor（感知任务） |
| `InstinctRlEncoderActorCriticCfg` | `"EncoderActorCritic"` | 带编码器（处理深度图等） |
| `EstimatorActorCriticCfg` | `"EstimatorActorCritic"` | 带状态估计器（处理部分可观测） |

---

## 9. 完整数据流程图

```
                           ┌──────────────────────────────────────────────────┐
                           │              TRAINING LOOP                       │
                           │                                                  │
    ┌─────────────┐        │   ┌──────────────┐     ┌──────────────────┐     │
    │  OnPolicy   │◄───────┼───│ VecEnvWrapper│◄────│ InstinctRlEnv    │     │
    │  Runner     │        │   │ (pack/flat)  │     │ (ManagerBased)   │     │
    │  (PPO)      │──action┼──►│              │────►│                  │     │
    └─────────────┘        │   └──────────────┘     └──────┬───────────┘     │
                           │                               │                 │
                           └───────────────────────────────┼─────────────────┘
                                                           │
            ┌──────────────────────────────────────────────┼──────────────────────┐
            │                  ENVIRONMENT STEP             │                      │
            │                                               ▼                      │
            │  ActionManager                                │                      │
            │  ┌────────────────────────────────────────┐   │                      │
            │  │ process_action(action) → store history │───┼── action [N, act]    │
            │  │ apply_action() → write to entities     │   │                      │
            │  └────────────────────────────────────────┘   │                      │
            │                                               ▼                      │
            │  Decimation Loop (×4)                                                   │
            │  ┌────────────────────────────────────────┐                          │
            │  │ scene.write_data_to_sim()              │                          │
            │  │ sim.step() — GPU batch physics         │                          │
            │  │ scene.update(physics_dt)               │                          │
            │  │ metrics.substep()                      │                          │
            │  └────────────────────────────────────────┘                          │
            │                                               │                      │
            │  Termination + Reward                         ▼                      │
            │  ┌────────────────────────────────────────┐                          │
            │  │ termination_manager.compute()          │                          │
            │  │ reward_manager.compute(step_dt)        │                          │
            │  └────────────────────────────────────────┘                          │
            │                                               │                      │
            │  Auto Reset (if needed)                       ▼                      │
            │  ┌────────────────────────────────────────┐                          │
            │  │ curriculum → sim.reset → scene.reset   │                          │
            │  │ events("reset") → managers.reset()     │                          │
            │  └────────────────────────────────────────┘                          │
            │                                               │                      │
            │  Post-Reset Compute                           ▼                      │
            │  ┌────────────────────────────────────────┐                          │
            │  │ sim.forward() — refresh kinematics     │                          │
            │  │ command_manager.compute(step_dt)       │                          │
            │  │ event_manager("step"/"interval")       │                          │
            │  │ sim.sense() — read sensors             │                          │
            │  │ observation_manager.compute()          │                          │
            │  │   └─ noise → clip → scale → delay      │                          │
            │  │      → history → concat → NaN check    │                          │
            │  └────────────────────────────────────────┘                          │
            │                                                                      │
            └──────────────────────────────────────────────────────────────────────┘
```

---

## 10. 与 IsaacLab 的关键差异

| 维度 | IsaacLab (PhysX) | mjlab (MuJoCo Warp) |
|------|------------------|---------------------|
| **配置格式** | Hydra YAML + `@configclass` | Python `@dataclass`（无 YAML 文件） |
| **物理引擎** | NVIDIA PhysX GPU | MuJoCo + MJWarp GPU |
| **场景组装** | `InteractiveScene` + 每环境 `clone()` | `Scene` + MJCF `spec` 模板 + `attach()` |
| **实体表示** | `ArticulationView` / `RigidObjectView` | `Entity` + `EntityData` (WarpBridge 张量) |
| **状态访问** | `data.root_pos_w`、`data.joint_pos` | `entity.data.root_link_pose_w`、`entity.data.joint_pos` |
| **仿真 API** | `sim.step(render=False)` | `sim.step()` / `sim.forward()` / `sim.sense()` 显式分离 |
| **动作应用** | 每个 Manager 步进一次 `apply_action()` | 每个 decimation 子步都 `apply_action()` |
| **观察历史** | `history_length` 在 group 级别 | 每个 term 独立 `CircularBuffer` |
| **传感器延迟** | 不原生支持 | `DelayBuffer`（每环境独立延迟、hold probability） |
| **事件系统** | 类似模式 | 四模式：`startup` / `reset` / `interval` / `step` |
| **auto_reset** | 通过 `TerminationsCfg` 控制 | 显式 `auto_reset` 标志 + `_manual_reset_pending` 状态追踪 |
| **RL 算法** | RSL-RL（捆绑） | `instinct_rl`（通过 `InstinctRlVecEnvWrapper` 适配） |
| **分布式训练** | `torchrun`（Hydra 启动器） | `torchrunx.Launcher`（自定义启动器，支持 SLURM） |
| **Warp 缓存** | N/A | 每 rank 独立 `WARP_CACHE_PATH` 避免多进程 kernel JIT 竞争 |
| **Forward 调用位置** | 隐式（PhysX 内部处理） | 显式（单一 `sim.forward()` 在重置后、观察计算前） |
| **Actor/Observer** | 支持 `DirectRLEnv` | 仅 `ManagerBasedRlEnv`（无 Direct 模式） |

---

## 11. 物理步与策略步的关系

```
┌──────────────────────────────────────────────────────────────┐
│                        时间轴                                 │
│                                                              │
│  策略步 N-1            策略步 N              策略步 N+1       │
│  ───────────|──────────────────────────|──────────────────   │
│             │                          │                     │
│             ├── physics_step 1         │                     │
│             ├── physics_step 2         │                     │
│             ├── physics_step 3         │                     │
│             ├── physics_step 4         │                     │
│             │                          │                     │
│   dt_physics = mujoco.timestep   (如 0.005s = 200Hz)         │
│   dt_env     = dt_physics × decimation  (如 0.020s = 50Hz)   │
│                                                              │
│   decimation = 4  → 每个策略步内运行 4 次物理步进               │
└──────────────────────────────────────────────────────────────┘
```

---

## 12. 总结：一张图理解全部

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     配置层（Python Dataclass）                            │
│  ManagerBasedRlEnvCfg ─────► InstinctLabRLEnvCfg (+viewer, +monitors)   │
│       │                         │                                       │
│       │  scene: SceneCfg        │  + monitors: MonitorManager           │
│       │  actions/obs/rewards/   │  + rewards → MultiRewardManager       │
│       │  terminations/commands/ │  + scene → InstinctScene              │
│       │  events/curriculum      │                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                     环境层（Gymnasium Vector Env）                        │
│                                                                         │
│  reset()                         step(action)                           │
│    │                                │                                   │
│    ├─ _reset_idx()                  ├─ action_mgr.process_action()      │
│    │  ├─ sim.reset()                ├─ [decimation ×]                   │
│    │  ├─ scene.reset()              │  ├─ action_mgr.apply_action()     │
│    │  ├─ events("reset")            │  ├─ scene.write_data_to_sim()     │
│    │  └─ managers.reset()           │  ├─ sim.step()                    │
│    │                                │  └─ scene.update(physics_dt)      │
│    ├─ scene.write_data_to_sim()     │                                   │
│    ├─ sim.forward()                 ├─ termination_mgr.compute()        │
│    ├─ command_mgr.compute(0)        ├─ reward_mgr.compute(step_dt)      │
│    ├─ sim.sense()                   ├─ [auto-reset] _reset_idx()        │
│    └─ obs_mgr.compute()            ├─ sim.forward()                    │
│                                     ├─ command_mgr.compute(step_dt)     │
│                                     ├─ events("step"/"interval")        │
│                                     ├─ sim.sense()                      │
│                                     └─ obs_mgr.compute()                │
│                                                                         │
│  RETURN (obs_dict, extras)          RETURN (obs, reward, term, trunc,   │
│                                             extras)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                     适配层                                               │
│  InstinctRlVecEnvWrapper: 展平 obs_dict → (policy_tensor, critic_tensor) │
│  多奖励 → 堆叠为 (num_envs, num_rewards)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                     训练层                                               │
│  instinct_rl OnPolicyRunner: PPO rollout → GAE → update → checkpoint     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 13. 关键文件索引

| 文件 | 内容 |
|------|------|
| [`src/instinct_mj/envs/manager_based_rl_env.py`](../src/instinct_mj/envs/manager_based_rl_env.py) | `InstinctRlEnv` — 主环境类 |
| [`src/instinct_mj/envs/manager_based_rl_env_cfg.py`](../src/instinct_mj/envs/manager_based_rl_env_cfg.py) | `InstinctLabRLEnvCfg` — 环境配置 |
| [`src/instinct_mj/envs/scene.py`](../src/instinct_mj/envs/scene.py) | `InstinctScene` — 自定义场景类 |
| [`src/instinct_mj/envs/mdp/`](../src/instinct_mj/envs/mdp/) | 自定义 MDP 组件（动作、观察、奖励、终止、事件、指令、课程） |
| [`src/instinct_mj/managers/`](../src/instinct_mj/managers/) | 自定义 Manager（MultiRewardManager 等） |
| [`src/instinct_mj/monitors/`](../src/instinct_mj/monitors/) | 自定义监控系统 |
| [`src/instinct_mj/rl/vecenv_wrapper.py`](../src/instinct_mj/rl/vecenv_wrapper.py) | `InstinctRlVecEnvWrapper` — instinct_rl 适配器 |
| [`src/instinct_mj/rl/config.py`](../src/instinct_mj/rl/config.py) | PPO 算法配置类 |
| [`src/instinct_mj/rl/module_cfg.py`](../src/instinct_mj/rl/module_cfg.py) | 编码器模块配置（MLP、Conv2d、Transformer） |
| [`src/instinct_mj/tasks/registry.py`](../src/instinct_mj/tasks/registry.py) | 任务注册表 |
| [`src/instinct_mj/tasks/locomotion/config/g1/flat_env_cfg.py`](../src/instinct_mj/tasks/locomotion/config/g1/flat_env_cfg.py) | G1 Flat Locomotion 任务配置示例 |
| [`src/instinct_mj/scripts/instinct_rl/train.py`](../src/instinct_mj/scripts/instinct_rl/train.py) | 训练入口脚本 |
| [`src/instinct_mj/scripts/instinct_rl/play.py`](../src/instinct_mj/scripts/instinct_rl/play.py) | 推理回放入口脚本 |
| [`src/instinct_mj/assets/unitree_g1.py`](../src/instinct_mj/assets/unitree_g1.py) | Unitree G1 机器人资产定义 |
| [`src/instinct_mj/sensors/`](../src/instinct_mj/sensors/) | 自定义传感器（GroupedRayCaster、NoisyCamera、VolumePoints） |
| [`src/instinct_mj/terrains/`](../src/instinct_mj/terrains/) | 地形生成与导入 |
| [`src/instinct_mj/motion_reference/`](../src/instinct_mj/motion_reference/) | 运动参考系统（Motion Buffer、AMASS 加载器等） |

---

## 14. mjlab 核心框架文件（依赖库）

mjlab 安装在 `/tmp/mjlab/` 下，核心文件位于 [`/tmp/mjlab/src/mjlab/`](/tmp/mjlab/src/mjlab/)：

| 文件 | 内容 |
|------|------|
| `envs/manager_based_rl_env.py` | `ManagerBasedRlEnv` + `ManagerBasedRlEnvCfg` — 核心基类 |
| `envs/types.py` | `VecEnvObs`、`VecEnvStepReturn` 类型别名 |
| `sim/sim.py` | `Simulation` + `SimulationCfg` + `MujocoCfg` — GPU 加速仿真 |
| `scene/scene.py` | `Scene` + `SceneCfg` — MJCF 场景组装 |
| `entity/entity.py` | `Entity` + `EntityCfg` — 物理实体（机器人/物体） |
| `managers/action_manager.py` | `ActionManager` + `ActionTerm` — 动作处理 |
| `managers/observation_manager.py` | `ObservationManager` + `ObservationGroupCfg` — 观察计算 |
| `managers/reward_manager.py` | `RewardManager` + `RewardTermCfg` — 奖励聚合 |
| `managers/termination_manager.py` | `TerminationManager` — 终止条件 |
| `managers/command_manager.py` | `CommandManager` + `CommandTerm` — 指令生成 |
| `managers/event_manager.py` | `EventManager` + `EventTermCfg` — 事件/域随机化 |
| `managers/curriculum_manager.py` | `CurriculumManager` — 课程学习 |
| `managers/metrics_manager.py` | `MetricsManager` — 指标记录 |
| `managers/recorder_manager.py` | `RecorderManager` — 数据录制 |
| `managers/manager_base.py` | `ManagerBase` + `ManagerTermBase` — Manager 基类 |
| `managers/scene_entity_config.py` | `SceneEntityCfg` — 命名实体引用 |
