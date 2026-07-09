"""Visualize Part2Link motion reference, object replay, and contact vectors.

This is a data/debug visualization utility.  It reuses the Part2Link PLAY task
configuration, enables the existing Part2Link contact-vector debug markers, and
runs a zero-action policy so no training checkpoint is required.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")

import mjlab
import tyro
import yaml


DEFAULT_TASK_ID = "Instinct-Interaction-Sitting-Part2Link-Transformer-G1-v0"


@dataclass(frozen=True)
class Part2LinkMotionContactVisCfg:
    task_id: str = DEFAULT_TASK_ID
    motion_file: str | None = None
    num_envs: int = 1
    device: str | None = None
    viewer: Literal["auto", "native", "viser", "none"] = "auto"
    max_steps: int | None = None
    no_terminations: bool = True
    show_contact_points: bool = True
    show_part_centers: bool = True
    ignore_contact_phase: bool = False
    debug_max_envs: int = 1
    point_radius: float = 0.025
    center_radius: float = 0.035
    line_radius: float = 0.008
    collision_visual_mode: Literal["mesh", "collision"] = "mesh"
    video: bool = False
    video_length: int = 400
    video_dir: str | None = None
    video_height: int | None = None
    video_width: int | None = None


def _make_single_motion_metadata(motion_file: str) -> str:
    motion_path = Path(motion_file).expanduser().resolve()
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
    if motion_path.suffix != ".npz":
        raise ValueError(f"Part2Link motion file must be an .npz file: {motion_path}")

    temp = tempfile.NamedTemporaryFile("w", suffix="_part2link_motion.yaml", delete=False)
    with temp:
        yaml.safe_dump(
            {
                "motion_files": [
                    {
                        "motion_file": str(motion_path),
                        "weight": 1.0,
                    }
                ]
            },
            temp,
        )
    return temp.name


def _configure_single_motion(env_cfg, motion_file: str) -> None:
    metadata_yaml = _make_single_motion_metadata(motion_file)
    motion_reference_cfg = next(sensor for sensor in env_cfg.scene.sensors if sensor.name == "motion_reference")
    motion_buffer = next(iter(motion_reference_cfg.motion_buffers.values()))
    motion_buffer.path = "/"
    motion_buffer.metadata_yaml = metadata_yaml
    motion_buffer.motion_start_from_middle_range = [0.0, 0.0]
    motion_buffer.motion_bin_length_s = None
    motion_buffer.env_starting_stub_sampling_strategy = "independent"
    print(f"[INFO] Visualizing single Part2Link motion: {Path(motion_file).expanduser().resolve()}")


def _enable_part2link_contact_debug(env_cfg, cfg: Part2LinkMotionContactVisCfg) -> None:
    reward_cfg = env_cfg.rewards.get("part2link_vector_guidance_gauss")
    if reward_cfg is None:
        raise RuntimeError("Task does not define reward term 'part2link_vector_guidance_gauss'.")
    reward_cfg.params["debug_vis"] = True
    reward_cfg.params["debug_vis_max_envs"] = cfg.debug_max_envs
    reward_cfg.params["debug_vis_show_contact_points"] = cfg.show_contact_points
    reward_cfg.params["debug_vis_show_part_centers"] = cfg.show_part_centers
    reward_cfg.params["debug_vis_ignore_contact_phase"] = cfg.ignore_contact_phase
    reward_cfg.params["debug_vis_point_radius"] = cfg.point_radius
    reward_cfg.params["debug_vis_center_radius"] = cfg.center_radius
    reward_cfg.params["debug_vis_line_radius"] = cfg.line_radius


def run_visualization(cfg: Part2LinkMotionContactVisCfg) -> None:
    if cfg.collision_visual_mode != "mesh":
        os.environ["INSTINCT_PART2LINK_COLLISION_VISUAL_MODE"] = cfg.collision_visual_mode

    from instinct_mj.envs import InstinctRlEnv
    from instinct_mj.rl import InstinctRlVecEnvWrapper
    from instinct_mj.scripts.instinct_rl.play import (
        _ViewerEnvAdapter,
        _build_dummy_policy,
        _disable_headless_debug_visualization,
        _force_viewer_realtime_1x,
        _patch_world_free_camera,
        _resolve_device,
        _resolve_viewer_backend,
        _run_headless_rollout,
    )
    from instinct_mj.tasks.registry import load_env_cfg, load_instinct_rl_cfg
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.utils.wrappers import VideoRecorder
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    configure_torch_backends()
    viewer_backend = _resolve_viewer_backend(cfg.viewer)
    if viewer_backend == "native":
        os.environ["MUJOCO_GL"] = "glfw"
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")

    env_cfg = load_env_cfg(cfg.task_id, play=True)
    agent_cfg = load_instinct_rl_cfg(cfg.task_id)
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if viewer_backend != "native" or not has_display:
        _disable_headless_debug_visualization(env_cfg)
        if hasattr(env_cfg, "headless"):
            env_cfg.headless = True
    env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width
    if cfg.no_terminations:
        env_cfg.terminations = {}
    if cfg.motion_file is not None:
        _configure_single_motion(env_cfg, cfg.motion_file)
    _enable_part2link_contact_debug(env_cfg, cfg)

    device = _resolve_device(cfg.device)
    env = InstinctRlEnv(
        cfg=env_cfg,
        device=device,
        render_mode="rgb_array" if cfg.video else None,
    )
    if cfg.video:
        video_dir = cfg.video_dir
        if video_dir is None:
            video_dir = str(Path("logs") / "part2link_motion_contact_visualization" / "videos")
        env = VideoRecorder(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            disable_logger=True,
            name_prefix=cfg.task_id.replace("/", "_"),
        )
        print(f"[INFO] Recording visualization video to: {video_dir}")

    vec_env = InstinctRlVecEnvWrapper(
        env,
        policy_group=agent_cfg.policy_observation_group,
        critic_group=agent_cfg.critic_observation_group,
    )
    viewer_env = _ViewerEnvAdapter(vec_env)
    policy = _build_dummy_policy("zero", (vec_env.num_envs, vec_env.num_actions), device)

    try:
        if viewer_backend == "native":
            viewer = NativeMujocoViewer(viewer_env, policy)
            _patch_world_free_camera(viewer)
            _force_viewer_realtime_1x(viewer)
            viewer.run(num_steps=cfg.max_steps)
        elif viewer_backend == "viser":
            viewer = ViserPlayViewer(viewer_env, policy)
            _force_viewer_realtime_1x(viewer)
            viewer.run(num_steps=cfg.max_steps)
        elif viewer_backend == "none":
            rollout_steps = cfg.max_steps if cfg.max_steps is not None else cfg.video_length if cfg.video else 300
            _run_headless_rollout(viewer_env, policy, num_steps=rollout_steps)
        else:
            raise RuntimeError(f"Unsupported viewer backend: {viewer_backend}")
    finally:
        viewer_env.close()


def main() -> None:
    cfg = tyro.cli(
        Part2LinkMotionContactVisCfg,
        default=Part2LinkMotionContactVisCfg(),
        prog=sys.argv[0],
        config=mjlab.TYRO_FLAGS,
    )
    run_visualization(cfg)


if __name__ == "__main__":
    main()
