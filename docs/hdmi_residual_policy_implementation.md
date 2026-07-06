# HDMI-Style Residual Joint-Position Policy Implementation Guide

## 1. Goal

Implement the residual policy used in HDMI-style humanoid motion imitation / interaction training.

The key idea is:

> The policy should **not** directly output the absolute joint position target. Instead, it should output a **normalized residual action** around the reference motion joint pose.

The core formula is:

```text
q_target = q_ref + action_scale * delta_action
```

where:

```text
q_ref         : reference joint position from the motion dataset, in radians
q_target      : final PD joint-position target, in radians
action_scale  : per-joint action scaling, in radians
delta_action  : residual action predicted by the policy, normalized / unitless
```

---

## 2. Important Variable Definitions

There are three different spaces that must not be confused.

### 2.1 Raw joint-position space

```python
q_ref      # reference joint position, unit: rad
q_default  # default robot joint position, unit: rad
q_target   # final PD joint target, unit: rad
```

### 2.2 Normalized action space

```python
ref_action     # normalized reference action, unitless
delta_action   # normalized residual action predicted by policy, unitless
final_action   # normalized final action sent to the action manager, unitless
```

### 2.3 Scaling factor

```python
action_scale   # per-joint action scale, unit: rad
```

The action manager converts normalized actions into joint-position targets using:

```python
q_target = q_default + action_scale * final_action
```

---

## 3. Core Mathematical Form

First convert the reference joint position into normalized action space:

```python
ref_action = (q_ref - q_default) / action_scale
```

Then let the policy output a normalized residual:

```python
delta_action = policy(obs)
```

Compose the final normalized action:

```python
final_action = ref_action + delta_action
```

The action manager converts it back to joint-position target:

```python
q_target = q_default + action_scale * final_action
```

Substitute `final_action`:

```python
q_target = q_default + action_scale * (ref_action + delta_action)
```

Because:

```python
ref_action = (q_ref - q_default) / action_scale
```

we get:

```python
q_target = q_ref + action_scale * delta_action
```

This is the HDMI-style residual policy.

---

## 4. Minimal Pseudocode

```python
class ResidualPolicyWrapper:
    def __init__(self, policy, q_default, action_scale):
        self.policy = policy
        self.q_default = q_default          # shape: [num_joints]
        self.action_scale = action_scale    # shape: [num_joints]

    def compute_ref_action(self, q_ref):
        """
        Convert reference joint position into normalized action space.

        Args:
            q_ref: reference joint position, shape [B, num_joints], unit: rad

        Returns:
            ref_action: normalized reference action, shape [B, num_joints]
        """
        ref_action = (q_ref - self.q_default) / self.action_scale
        return ref_action

    def act(self, obs, q_ref):
        """
        Args:
            obs: policy observation
            q_ref: current reference joint position from motion data

        Returns:
            final_action: normalized joint-position command
        """
        ref_action = self.compute_ref_action(q_ref)

        # Policy only predicts residual offset in normalized action space.
        delta_action = self.policy(obs)

        # Residual action composition.
        final_action = ref_action + delta_action

        return final_action
```

The environment then executes:

```python
class JointPositionActionManager:
    def __init__(self, robot, q_default, action_scale):
        self.robot = robot
        self.q_default = q_default
        self.action_scale = action_scale

    def apply_action(self, final_action):
        """
        Args:
            final_action: normalized joint-position command
        """
        q_target = self.q_default + final_action * self.action_scale
        self.robot.set_joint_position_target(q_target)
```

---

## 5. Implementation Closer to HDMI Code Structure

In HDMI, the residual addition is placed after the actor outputs the Gaussian action mean.

The actor predicts the residual action distribution:

```python
class ActorNetwork:
    def __init__(self, obs_dim, action_dim):
        self.mlp = MLP(obs_dim, hidden_dims=[512, 256, 256])
        self.mean_head = Linear(256, action_dim)
        self.log_std = Parameter(shape=[action_dim])

    def forward(self, obs):
        feature = self.mlp(obs)
        delta_action_mean = self.mean_head(feature)
        action_std = exp(self.log_std)
        return delta_action_mean, action_std
```

Then a residual module adds the normalized reference action:

```python
class ResidualActionModule:
    def forward(self, ref_action, delta_action):
        """
        Args:
            ref_action: normalized reference action
            delta_action: normalized residual action predicted by policy

        Returns:
            final_action: normalized final action
        """
        final_action = ref_action + delta_action
        return final_action
```

The residual Gaussian policy becomes:

```python
class ResidualGaussianPolicy:
    def __init__(self, actor, residual_module):
        self.actor = actor
        self.residual_module = residual_module

    def get_dist(self, obs, ref_action):
        delta_mean, std = self.actor(obs)

        # Add residual at the distribution mean level.
        final_mean = self.residual_module(ref_action, delta_mean)

        dist = Normal(mean=final_mean, std=std)
        return dist

    def sample_action(self, obs, ref_action):
        dist = self.get_dist(obs, ref_action)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
```

Important:

```text
The residual should be added to the Gaussian mean / loc before sampling.
```

Do not first sample `delta_action` and then add `ref_action` outside the distribution unless the log probability is computed consistently.

---

## 6. Full Training-Step Pseudocode

```python
for each simulation step:

    # 1. Get current reference joint position from motion dataset.
    q_ref = motion_dataset.get_current_ref_joint_pos(env_ids)

    # 2. Get default joint position and per-joint action scale.
    q_default = action_manager.default_joint_pos
    action_scale = action_manager.action_scale

    # 3. Convert q_ref to normalized reference action.
    ref_action = (q_ref - q_default) / action_scale

    # 4. Build observation.
    obs = build_policy_obs(
        robot_state=robot_state,
        object_state=object_state,
        command=command,
        previous_actions=previous_actions,
    )

    # 5. Actor predicts residual action distribution.
    delta_action_mean, action_std = actor(obs)

    # 6. Residual composition in normalized action space.
    final_action_mean = ref_action + delta_action_mean

    # 7. Sample final normalized action.
    dist = Normal(final_action_mean, action_std)
    final_action = dist.sample()
    log_prob = dist.log_prob(final_action)

    # 8. Convert normalized action to PD joint target.
    q_target = q_default + action_scale * final_action

    # Equivalent interpretation:
    # q_target = q_ref + action_scale * delta_action

    # 9. Apply joint-position target.
    robot.set_joint_position_target(q_target)

    # 10. Step simulator.
    sim.step()

    # 11. Compute rewards and store transition.
    reward = compute_reward()
    ppo_buffer.add(obs, final_action, reward, log_prob, value)
```

---

## 7. How to Connect with an Existing PD Controller

If the current controller is:

```python
torque = kp * (q_target - q_current) - kd * qd_current
```

then the HDMI residual policy should produce `q_target` as:

```python
delta_action = policy(obs)
q_target = q_ref + action_scale * delta_action

torque = kp * (q_target - q_current) - kd * qd_current
```

The residual is added at the **joint-position target level**, not the torque level.

Do not implement:

```python
torque = torque_ref + policy(obs)
```

Do not implement:

```python
q_target = policy(obs)
```

The correct implementation is:

```python
q_target = reference_joint_position + learned_residual
```

---

## 8. Teacher-Student Distillation Version

HDMI also supports residual action distillation.

During training:

- The teacher uses privileged observations and residual action composition.
- The student uses deployable observations.
- The student learns to match the teacher's **final action mean**, not only the residual.

### 8.1 Teacher

```python
teacher_delta_mean, teacher_std = teacher_actor(priv_obs)
teacher_final_mean = ref_action + teacher_delta_mean
```

### 8.2 Student

```python
student_final_mean, student_std = student_actor(student_obs)
```

### 8.3 Distillation loss

```python
distill_loss = mse(student_final_mean, teacher_final_mean)

optimizer_student.zero_grad()
distill_loss.backward()
optimizer_student.step()
```

Important:

```python
teacher_final_mean = ref_action + teacher_delta_mean
```

not:

```python
teacher_final_mean = teacher_delta_mean
```

The student should imitate the teacher's final action distribution mean after residual composition.

---

## 9. Direct Instruction for a Coding Agent

```text
Implement HDMI-style residual joint-position policy.

The policy should not output absolute joint position targets. It should output a normalized residual action delta_action.

At every control step:

1. Read the current reference joint position q_ref from the motion dataset for the actuated joints.
2. Read q_default and action_scale from the action manager.
3. Convert q_ref into normalized action space:

   ref_action = (q_ref - q_default) / action_scale

4. Run actor(obs) to predict delta_action_mean and action_std.
5. Add the reference action to the predicted residual mean:

   final_action_mean = ref_action + delta_action_mean

6. Sample final_action from Normal(final_action_mean, action_std).
7. Convert final_action to PD target:

   q_target = q_default + action_scale * final_action

   This is equivalent to:

   q_target = q_ref + action_scale * delta_action

8. Send q_target to the joint-position controller.

Do not add the residual at the torque level.
Do not treat q_ref as an observation only; it must be added to the actor output in normalized action space.
Do not add raw q_ref directly to normalized action. Always convert q_ref to ref_action first.
```

---

## 10. Common Implementation Mistakes

### Mistake 1: Directly adding raw `q_ref` to normalized action

Wrong:

```python
final_action = q_ref + delta_action
```

Reason:

```text
q_ref is in radians, while delta_action is normalized / unitless.
```

Correct:

```python
ref_action = (q_ref - q_default) / action_scale
final_action = ref_action + delta_action
```

---

### Mistake 2: Letting policy output absolute joint target

Wrong:

```python
q_target = policy(obs)
```

Correct:

```python
delta_action = policy(obs)
q_target = q_ref + action_scale * delta_action
```

---

### Mistake 3: Adding residual at torque level

Wrong:

```python
torque = torque_ref + policy(obs)
```

Correct:

```python
delta_action = policy(obs)
q_target = q_ref + action_scale * delta_action
torque = kp * (q_target - q_current) - kd * qd_current
```

---

### Mistake 4: Distilling only the residual

Wrong:

```python
loss = mse(student_action, teacher_delta_action)
```

Correct:

```python
teacher_final_action = ref_action + teacher_delta_action
loss = mse(student_action, teacher_final_action)
```

---

## 11. Minimal Formula to Remember

```python
# Reference action in normalized action space.
a_ref = (q_ref - q_default) / action_scale

# Residual predicted by policy.
delta_a = policy(obs)

# Final normalized action.
a = a_ref + delta_a

# PD target.
q_target = q_default + action_scale * a
```

Equivalent:

```python
q_target = q_ref + action_scale * delta_a
```

This is the complete core of HDMI-style residual joint-position policy.

---

## 12. Practical Debug Signals

When implementing this, log the following values:

```python
mean_abs_ref_action = ref_action.abs().mean()
mean_abs_delta_action = delta_action.abs().mean()
mean_abs_final_action = final_action.abs().mean()
mean_abs_q_ref = q_ref.abs().mean()
mean_abs_q_target = q_target.abs().mean()
mean_abs_q_target_minus_q_ref = (q_target - q_ref).abs().mean()
```

Useful sanity checks:

```text
1. At initialization, delta_action should be relatively small.
2. q_target should be close to q_ref if delta_action is small.
3. final_action should be in a reasonable normalized action range.
4. q_target - q_ref should approximately equal action_scale * delta_action.
5. If q_target jumps violently, check whether raw q_ref was added directly to normalized action.
```

---

## 13. Recommended Integration Checklist

```text
[ ] Extract q_ref for the same joint order as the action manager.
[ ] Ensure q_ref shape matches action dimension: [num_envs, action_dim].
[ ] Ensure q_default is indexed by the same joint IDs as the action dimension.
[ ] Ensure action_scale is indexed by the same joint IDs.
[ ] Convert q_ref to ref_action before residual addition.
[ ] Add residual to actor mean / loc before sampling.
[ ] Compute log_prob under the distribution with residualized final mean.
[ ] Convert sampled final_action back to q_target through the action manager.
[ ] Send q_target to PD position controller.
[ ] Log q_target - q_ref to verify residual magnitude.
```

