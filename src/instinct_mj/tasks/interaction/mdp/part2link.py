from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers import ManagerTermBase, SceneEntityCfg
from mjlab.utils.lab_api import math as math_utils

from instinct_mj.monitors.monitor_manager import MonitorTerm
import instinct_mj.utils.math as instinct_math

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedEnv, ManagerBasedRlEnv
    from mjlab.managers import CurriculumTermCfg

    from instinct_mj.motion_reference import Part2LinkMotionReferenceData
    from instinct_mj.motion_reference.motion_reference_manager import MotionReferenceManager


@dataclass(frozen=True)
class ObjectVariant:
    name: str
    chair_name: str
    alpha: float
    mesh_path: Path
    contact_points_path: Path
    alpha_index: int


@dataclass
class ObjectVariantCatalog:
    variants: list[ObjectVariant]
    centers_local: torch.Tensor
    points_local: torch.Tensor
    point_valid_mask: torch.Tensor
    part_names: list[str]
    chair_names: list[str]
    variant_names: list[str]


@dataclass
class ObjectVariantRuntimeState:
    catalog: ObjectVariantCatalog
    active_variant_ids: torch.Tensor
    active_alpha: torch.Tensor
    active_scale: torch.Tensor
    active_precision_scale: torch.Tensor
    active_centers_local: torch.Tensor
    active_points_local: torch.Tensor
    active_point_valid_mask: torch.Tensor
    reference_position_offsets: torch.Tensor


_CATALOG_CACHE: dict[tuple[str, str | None, tuple[str, ...] | None, tuple[float, ...]], ObjectVariantCatalog] = {}
_INACTIVE_OBJECT_SPACING = 20.0
_BODY_ALIAS = {
    "left_rubber_hand_link": "left_rubber_hand",
    "right_rubber_hand_link": "right_rubber_hand",
}
_PART_ALIAS = {
    "arm_left": "armrest_left",
    "arm_right": "armrest_right",
}


def _alpha_to_token(alpha: float) -> str:
    return f"alpha_{float(alpha):.2f}".replace(".", "p")


def _alpha_to_glb_name(alpha: float) -> str:
    return f"alpha_{float(alpha):.2f}.glb"


def _sanitize_variant_name(chair_name: str, alpha: float) -> str:
    return f"{chair_name}_{_alpha_to_token(alpha)}".replace("-", "_").replace(".", "p")


def _parse_chair_names(value: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        names = tuple(item.strip() for item in value.split(",") if item.strip())
        return names or None
    names = tuple(str(item) for item in value if str(item))
    return names or None


def _parse_alpha_values(value: str | Sequence[float] | None, default: Sequence[float]) -> tuple[float, ...]:
    if value is None:
        return tuple(float(alpha) for alpha in default)
    if isinstance(value, str):
        alphas = tuple(float(item.strip()) for item in value.split(",") if item.strip())
        return alphas or tuple(float(alpha) for alpha in default)
    return tuple(float(alpha) for alpha in value)


def _resolve_chair_dirs(extension_root: str | Path, chair_names: Sequence[str] | None) -> list[Path]:
    root = Path(extension_root).expanduser().resolve()
    if chair_names is None or len(chair_names) == 0:
        candidates = sorted(path for path in root.iterdir() if path.is_dir())
    else:
        candidates = [(root / str(name)).resolve() for name in chair_names]
    chair_dirs = [
        path
        for path in candidates
        if (path / "ffd_bbox_coarse" / "morph_path").is_dir()
        and (path / "contact_point_transfer" / "contact_points_local.npz").exists()
    ]
    if not chair_dirs:
        raise FileNotFoundError(f"No valid Part2Link chair variants found under extension_root={root}")
    return chair_dirs


def _resolve_mesh_path(chair_dir: Path, alpha: float, asset_cache: str | Path | None) -> Path:
    if asset_cache is not None:
        cache_dir = Path(asset_cache).expanduser().resolve() / chair_dir.name
        cache_candidates = [
            cache_dir / f"{_alpha_to_token(alpha)}.obj",
            cache_dir / f"{_alpha_to_glb_name(alpha)}.obj",
            cache_dir / "mesh.obj",
        ]
        for candidate in cache_candidates:
            if candidate.exists():
                return candidate

    morph_dir = chair_dir / "ffd_bbox_coarse" / "morph_path"
    glb_path = morph_dir / _alpha_to_glb_name(alpha)
    if glb_path.exists():
        return glb_path.resolve()

    usd_path = morph_dir / f"{_alpha_to_token(alpha)}.usd"
    if usd_path.exists():
        return usd_path.resolve()

    raise FileNotFoundError(f"No mesh found for chair={chair_dir.name}, alpha={alpha:.2f} under {morph_dir}")


def _read_part_names(contact_data: np.lib.npyio.NpzFile, num_parts: int) -> list[str]:
    for key in ("points_names", "part_names", "parts_names"):
        if key in contact_data:
            return [str(item) for item in contact_data[key].tolist()]
    return [f"part_{idx}" for idx in range(num_parts)]


def load_object_variant_catalog(
    extension_root: str | Path,
    chair_names: Sequence[str] | str | None = None,
    alpha_values: Sequence[float] | str | None = None,
    asset_cache: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> ObjectVariantCatalog:
    parsed_chairs = _parse_chair_names(chair_names)
    parsed_alphas = _parse_alpha_values(alpha_values, default=(1.0,))
    cache_key = (
        str(Path(extension_root).expanduser().resolve()),
        str(Path(asset_cache).expanduser().resolve()) if asset_cache is not None else None,
        parsed_chairs,
        parsed_alphas,
    )
    if cache_key in _CATALOG_CACHE:
        catalog = _CATALOG_CACHE[cache_key]
        if catalog.centers_local.device == torch.device(device):
            return catalog

    variants: list[ObjectVariant] = []
    centers: list[np.ndarray] = []
    points: list[np.ndarray] = []
    point_masks: list[np.ndarray] = []
    chair_name_values: list[str] = []
    variant_names: list[str] = []
    part_names: list[str] | None = None

    for chair_dir in _resolve_chair_dirs(extension_root, parsed_chairs):
        contact_path = chair_dir / "contact_point_transfer" / "contact_points_local.npz"
        contact_data = np.load(contact_path, allow_pickle=True)
        contact_alphas = np.asarray(contact_data["alpha_values"], dtype=np.float32).reshape(-1)
        points_local = np.asarray(contact_data["points_local"], dtype=np.float32)
        centers_local = np.asarray(contact_data["center_local"], dtype=np.float32)
        if points_local.ndim != 4 or centers_local.ndim != 3:
            raise ValueError(
                f"Expected alpha-indexed contact points in {contact_path}, "
                f"got points={points_local.shape}, centers={centers_local.shape}"
            )
        if part_names is None:
            part_names = _read_part_names(contact_data, centers_local.shape[1])

        for target_alpha in parsed_alphas:
            alpha_idx = int(np.argmin(np.abs(contact_alphas - float(target_alpha))))
            alpha = float(contact_alphas[alpha_idx])
            if abs(alpha - float(target_alpha)) > 1.0e-4:
                warnings.warn(
                    f"Requested alpha={target_alpha:.2f} for {chair_dir.name}, using nearest alpha={alpha:.2f}.",
                    stacklevel=2,
                )
            mesh_path = _resolve_mesh_path(chair_dir, alpha, asset_cache)
            variant_name = _sanitize_variant_name(chair_dir.name, alpha)
            variants.append(
                ObjectVariant(
                    name=variant_name,
                    chair_name=chair_dir.name,
                    alpha=alpha,
                    mesh_path=mesh_path,
                    contact_points_path=contact_path.resolve(),
                    alpha_index=alpha_idx,
                )
            )
            centers.append(np.nan_to_num(centers_local[alpha_idx]).astype(np.float32))
            points.append(np.nan_to_num(points_local[alpha_idx]).astype(np.float32))
            point_masks.append(np.isfinite(points_local[alpha_idx]).all(axis=-1))
            chair_name_values.append(chair_dir.name)
            variant_names.append(variant_name)

    if not variants:
        raise FileNotFoundError(f"No Part2Link alpha variants found under extension_root={extension_root}")

    catalog = ObjectVariantCatalog(
        variants=variants,
        centers_local=torch.as_tensor(np.stack(centers), dtype=torch.float32, device=device),
        points_local=torch.as_tensor(np.stack(points), dtype=torch.float32, device=device),
        point_valid_mask=torch.as_tensor(np.stack(point_masks), dtype=torch.bool, device=device),
        part_names=part_names or [],
        chair_names=chair_name_values,
        variant_names=variant_names,
    )
    _CATALOG_CACHE[cache_key] = catalog
    return catalog


def _get_or_create_variant_state(
    env: ManagerBasedEnv,
    extension_root: str | Path,
    chair_names: Sequence[str] | str | None = None,
    alpha_values: Sequence[float] | str | None = None,
    asset_cache: str | Path | None = None,
) -> ObjectVariantRuntimeState:
    state = getattr(env, "_part2link_object_variant_state", None)
    if state is not None:
        return state

    catalog = load_object_variant_catalog(
        extension_root,
        chair_names=chair_names,
        alpha_values=alpha_values,
        asset_cache=asset_cache,
        device=env.device,
    )
    num_envs = env.num_envs
    num_parts = catalog.centers_local.shape[1]
    num_points = catalog.points_local.shape[2]
    state = ObjectVariantRuntimeState(
        catalog=catalog,
        active_variant_ids=torch.zeros(num_envs, dtype=torch.long, device=env.device),
        active_alpha=torch.zeros(num_envs, dtype=torch.float32, device=env.device),
        active_scale=torch.ones(num_envs, dtype=torch.float32, device=env.device),
        active_precision_scale=torch.ones(num_envs, dtype=torch.float32, device=env.device),
        active_centers_local=torch.zeros(num_envs, num_parts, 3, dtype=torch.float32, device=env.device),
        active_points_local=torch.zeros(num_envs, num_parts, num_points, 3, dtype=torch.float32, device=env.device),
        active_point_valid_mask=torch.zeros(num_envs, num_parts, num_points, dtype=torch.bool, device=env.device),
        reference_position_offsets=torch.zeros(num_envs, 3, dtype=torch.float32, device=env.device),
    )
    env._part2link_object_variant_state = state
    env._interaction_object_variant_state = state
    return state


def _safe_unit_quat(quat: torch.Tensor) -> torch.Tensor:
    quat = torch.nan_to_num(quat, nan=0.0, posinf=0.0, neginf=0.0)
    norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    normalized = quat / torch.clamp(norm, min=1.0e-6)
    identity = torch.zeros_like(normalized)
    identity[..., 0] = 1.0
    return torch.where((norm > 1.0e-6).expand_as(normalized), normalized, identity)


def _root_pos(asset) -> torch.Tensor:
    data = asset.data
    for name in ("root_link_pos_w", "root_pos_w"):
        if hasattr(data, name):
            return torch.nan_to_num(getattr(data, name), nan=0.0, posinf=0.0, neginf=0.0)
    raise AttributeError(f"Entity '{getattr(asset, 'name', '<unknown>')}' has no root position tensor.")


def _root_quat(asset) -> torch.Tensor:
    data = asset.data
    for name in ("root_link_quat_w", "root_quat_w"):
        if hasattr(data, name):
            return _safe_unit_quat(getattr(data, name))
    raise AttributeError(f"Entity '{getattr(asset, 'name', '<unknown>')}' has no root orientation tensor.")


def _root_lin_vel(asset) -> torch.Tensor:
    data = asset.data
    for name in ("root_link_lin_vel_w", "root_lin_vel_w"):
        if hasattr(data, name):
            return torch.nan_to_num(getattr(data, name), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.zeros_like(_root_pos(asset))


def _root_ang_vel(asset) -> torch.Tensor:
    data = asset.data
    for name in ("root_link_ang_vel_w", "root_ang_vel_w"):
        if hasattr(data, name):
            return torch.nan_to_num(getattr(data, name), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.zeros_like(_root_pos(asset))


def _write_pose_velocity(asset, root_pose: torch.Tensor, root_velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
    if getattr(asset, "is_fixed_base", False):
        if not getattr(asset, "is_mocap", False):
            raise ValueError("Part2Link object entities must be mocap or free bodies.")
        asset.write_mocap_pose_to_sim(root_pose, env_ids=env_ids)
        return
    asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(root_velocity, env_ids=env_ids)


def _scene_object_slot(motion_ref: MotionReferenceManager, object_name: str) -> int:
    scene_object_names = getattr(motion_ref.cfg, "scene_object_names", None)
    if not scene_object_names:
        return 0
    if object_name in scene_object_names:
        return int(scene_object_names.index(object_name))
    warnings.warn(
        f"Object name '{object_name}' is not in motion_ref.scene_object_names={scene_object_names}; using slot 0.",
        stacklevel=2,
    )
    return 0


def _inactive_pose(
    env_ids: torch.Tensor,
    variant_index: int,
    env_origins: torch.Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    row = variant_index // 32
    col = variant_index % 32
    pose = torch.zeros(env_ids.numel(), 7, dtype=dtype, device=device)
    if env_origins is not None:
        pose[:, 0] = env_origins[env_ids, 0].to(device=device, dtype=dtype)
        pose[:, 1] = env_origins[env_ids, 1].to(device=device, dtype=dtype)
        pose[:, 2] = env_origins[env_ids, 2].to(device=device, dtype=dtype)
    pose[:, 0] += float(col) * _INACTIVE_OBJECT_SPACING
    pose[:, 1] += float(row) * _INACTIVE_OBJECT_SPACING
    pose[:, 2] -= 100.0
    pose[:, 3] = 1.0
    return pose


def _select_variants_for_reset(state: ObjectVariantRuntimeState, num_reset: int, device: torch.device) -> torch.Tensor:
    chair_order = sorted(set(state.catalog.chair_names))
    chair_ids = torch.randint(len(chair_order), (num_reset,), device=device)
    selected: list[int] = []
    for chair_id in chair_ids.detach().cpu().tolist():
        chair_name = chair_order[int(chair_id)]
        candidates = [idx for idx, variant in enumerate(state.catalog.variants) if variant.chair_name == chair_name]
        if not candidates:
            raise RuntimeError(f"No Part2Link variant for chair '{chair_name}'.")
        selected.append(candidates[0])
    return torch.tensor(selected, dtype=torch.long, device=device)


def _update_active_variant_state(
    state: ObjectVariantRuntimeState,
    env_ids: torch.Tensor,
    variant_ids: torch.Tensor,
    scales: torch.Tensor,
    precision_scales: torch.Tensor,
    reference_position_offsets: torch.Tensor,
) -> None:
    state.active_variant_ids[env_ids] = variant_ids
    state.active_alpha[env_ids] = torch.tensor(
        [state.catalog.variants[int(idx)].alpha for idx in variant_ids.detach().cpu().tolist()],
        dtype=torch.float32,
        device=env_ids.device,
    )
    state.active_scale[env_ids] = scales
    state.active_precision_scale[env_ids] = precision_scales
    state.active_centers_local[env_ids] = state.catalog.centers_local[variant_ids] * scales[:, None, None]
    state.active_points_local[env_ids] = state.catalog.points_local[variant_ids] * scales[:, None, None, None]
    state.active_point_valid_mask[env_ids] = state.catalog.point_valid_mask[variant_ids]
    state.reference_position_offsets[env_ids] = reference_position_offsets


def _invalidate_dynamic_mesh_sensors(env: ManagerBasedEnv) -> None:
    for sensor in getattr(env.scene, "sensors", {}).values():
        if hasattr(sensor, "invalidate_mesh_buffers"):
            sensor.invalidate_mesh_buffers()


def reset_object_variant_by_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_entity_names: Sequence[str],
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    object_name: str = "box",
    extension_root: str | Path = "",
    chair_names: Sequence[str] | str | None = None,
    alpha_values: Sequence[float] | str | None = None,
    asset_cache: str | Path | None = None,
    scale_distribution_params: tuple[float, float] = (1.0, 1.0),
    precision_scale_range: tuple[float, float] = (0.8, 1.4),
    base_lin_vel_ratio: float = 1.0,
    base_ang_vel_ratio: float = 1.0,
    apply_env_spacing: bool = False,
) -> None:
    motion_ref: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    state = _get_or_create_variant_state(env, extension_root, chair_names, alpha_values, asset_cache)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    num_reset = int(env_ids.numel())
    if num_reset == 0:
        return

    init_state = motion_ref.get_init_reference_state(env_ids)
    object_slot = _scene_object_slot(motion_ref, object_name)
    object_pos = init_state.object_pos_w[:, object_slot].clone()
    object_quat = _safe_unit_quat(init_state.object_quat_w[:, object_slot])
    lin_vel = init_state.object_lin_vel_w[:, object_slot] * float(base_lin_vel_ratio)
    ang_vel = init_state.object_ang_vel_w[:, object_slot] * float(base_ang_vel_ratio)
    reference_offsets = torch.zeros_like(object_pos)
    env_origins = getattr(env.scene, "env_origins", None)
    if apply_env_spacing and env_origins is not None:
        offsets = env_origins[env_ids].to(device=env.device, dtype=object_pos.dtype)
        object_pos = object_pos + offsets
        reference_offsets = offsets

    variant_ids = _select_variants_for_reset(state, num_reset, torch.device(env.device))
    scale_min, scale_max = scale_distribution_params
    if scale_max > scale_min:
        scales = torch.empty(num_reset, dtype=torch.float32, device=env.device).uniform_(scale_min, scale_max)
    else:
        scales = torch.ones(num_reset, dtype=torch.float32, device=env.device) * float(scale_min)
    precision_scales = torch.empty(num_reset, dtype=torch.float32, device=env.device).uniform_(*precision_scale_range)

    root_pose = torch.cat([object_pos, object_quat], dim=-1)
    root_velocity = torch.cat([lin_vel, ang_vel], dim=-1)
    zero_velocity = torch.zeros_like(root_velocity)
    for variant_index, entity_name in enumerate(object_entity_names):
        asset = env.scene[entity_name]
        active_mask = variant_ids == variant_index
        if active_mask.any():
            _write_pose_velocity(asset, root_pose[active_mask], root_velocity[active_mask], env_ids[active_mask])
        inactive_mask = ~active_mask
        if inactive_mask.any():
            inactive_pose = _inactive_pose(
                env_ids[inactive_mask],
                variant_index,
                env_origins,
                root_pose.dtype,
                root_pose.device,
            )
            _write_pose_velocity(asset, inactive_pose, zero_velocity[inactive_mask], env_ids[inactive_mask])

    _update_active_variant_state(
        state,
        env_ids=env_ids,
        variant_ids=variant_ids,
        scales=scales,
        precision_scales=precision_scales,
        reference_position_offsets=reference_offsets,
    )
    _invalidate_dynamic_mesh_sensors(env)


def update_object_variant_by_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_entity_names: Sequence[str],
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    object_name: str = "box",
    invalid_object_pos: tuple[float, float, float] = (0.0, 0.0, -100.0),
) -> None:
    del env_ids
    motion_ref: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    data: Part2LinkMotionReferenceData = motion_ref.data
    state = getattr(env, "_part2link_object_variant_state", None)
    if state is None:
        return

    object_slot = _scene_object_slot(motion_ref, object_name)
    all_env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    object_pos = data.object_pos_w[:, 0, object_slot].clone()
    object_pos = object_pos + state.reference_position_offsets.to(device=object_pos.device, dtype=object_pos.dtype)
    object_quat = _safe_unit_quat(data.object_quat_w[:, 0, object_slot])
    lin_vel = data.object_lin_vel_w[:, 0, object_slot]
    ang_vel = data.object_ang_vel_w[:, 0, object_slot]
    object_validity = data.object_validity[:, 0, object_slot].to(torch.bool)
    root_pose = torch.cat([object_pos, object_quat], dim=-1)
    root_velocity = torch.cat([lin_vel, ang_vel], dim=-1)
    invalid_pose = torch.zeros_like(root_pose)
    invalid_pose[:, :3] = torch.tensor(invalid_object_pos, dtype=root_pose.dtype, device=root_pose.device)
    invalid_pose[:, 3] = 1.0
    zero_velocity = torch.zeros_like(root_velocity)

    for variant_index, entity_name in enumerate(object_entity_names):
        asset = env.scene[entity_name]
        active_mask = (state.active_variant_ids == variant_index) & object_validity
        if active_mask.any():
            _write_pose_velocity(asset, root_pose[active_mask], root_velocity[active_mask], all_env_ids[active_mask])
        inactive_mask = ~active_mask
        if inactive_mask.any():
            hidden_pose = invalid_pose[inactive_mask].clone()
            hidden_pose[:, 0] += float(variant_index % 32) * _INACTIVE_OBJECT_SPACING
            hidden_pose[:, 1] += float(variant_index // 32) * _INACTIVE_OBJECT_SPACING
            _write_pose_velocity(asset, hidden_pose, zero_velocity[inactive_mask], all_env_ids[inactive_mask])
    motion_ref.sync_reference_entity_state()


def _active_object_state_w(env: ManagerBasedEnv, object_entity_names: Sequence[str]) -> tuple[torch.Tensor, ...]:
    state = getattr(env, "_part2link_object_variant_state", None)
    if state is None:
        first_asset = env.scene[object_entity_names[0]]
        return _root_pos(first_asset), _root_quat(first_asset), _root_lin_vel(first_asset), _root_ang_vel(first_asset)

    pos_list = []
    quat_list = []
    lin_vel_list = []
    ang_vel_list = []
    for entity_name in object_entity_names:
        asset = env.scene[entity_name]
        pos_list.append(_root_pos(asset))
        quat_list.append(_root_quat(asset))
        lin_vel_list.append(_root_lin_vel(asset))
        ang_vel_list.append(_root_ang_vel(asset))
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    variant_ids = state.active_variant_ids
    pos = torch.stack(pos_list, dim=1)[env_ids, variant_ids]
    quat = torch.stack(quat_list, dim=1)[env_ids, variant_ids]
    lin_vel = torch.stack(lin_vel_list, dim=1)[env_ids, variant_ids]
    ang_vel = torch.stack(ang_vel_list, dim=1)[env_ids, variant_ids]
    return pos, quat, lin_vel, ang_vel


def _object_reference_state_w(
    env: ManagerBasedEnv,
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    object_name: str = "box",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    motion_ref: MotionReferenceManager = env.scene[reference_cfg.name]
    data: Part2LinkMotionReferenceData = motion_ref.data
    object_slot = _scene_object_slot(motion_ref, object_name)
    pos = data.object_pos_w[:, 0, object_slot].clone()
    state = getattr(env, "_part2link_object_variant_state", None)
    if state is not None:
        pos = pos + state.reference_position_offsets.to(device=pos.device, dtype=pos.dtype)
    quat = _safe_unit_quat(data.object_quat_w[:, 0, object_slot])
    lin_vel = torch.nan_to_num(data.object_lin_vel_w[:, 0, object_slot], nan=0.0, posinf=0.0, neginf=0.0)
    ang_vel = torch.nan_to_num(data.object_ang_vel_w[:, 0, object_slot], nan=0.0, posinf=0.0, neginf=0.0)
    validity = data.object_validity[:, 0, object_slot].reshape(-1).to(pos.dtype)
    return pos, quat, lin_vel, ang_vel, validity


def object_position(
    env: ManagerBasedEnv,
    object_entity_names: Sequence[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    in_base_frame: bool = True,
) -> torch.Tensor:
    object_pos_w, _, _, _ = _active_object_state_w(env, object_entity_names)
    if not in_base_frame:
        return object_pos_w
    robot = env.scene[robot_cfg.name]
    return math_utils.quat_apply_inverse(_root_quat(robot), object_pos_w - _root_pos(robot))


def object_orientation_tannorm(
    env: ManagerBasedEnv,
    object_entity_names: Sequence[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    in_base_frame: bool = True,
) -> torch.Tensor:
    _, object_quat_w, _, _ = _active_object_state_w(env, object_entity_names)
    if not in_base_frame:
        return instinct_math.quat_to_tan_norm(object_quat_w)
    robot = env.scene[robot_cfg.name]
    object_quat_b = math_utils.quat_mul(math_utils.quat_inv(_root_quat(robot)), object_quat_w)
    return instinct_math.quat_to_tan_norm(object_quat_b)


def object_reference_position(
    env: ManagerBasedEnv,
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_name: str = "box",
    in_base_frame: bool = True,
) -> torch.Tensor:
    pos_ref_w, _, _, _, validity = _object_reference_state_w(env, reference_cfg, object_name)
    if not in_base_frame:
        return pos_ref_w * validity.unsqueeze(-1)
    robot = env.scene[robot_cfg.name]
    pos_ref_b = math_utils.quat_apply_inverse(_root_quat(robot), pos_ref_w - _root_pos(robot))
    return pos_ref_b * validity.unsqueeze(-1)


def object_reference_orientation_tannorm(
    env: ManagerBasedEnv,
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_name: str = "box",
    in_base_frame: bool = True,
) -> torch.Tensor:
    _, quat_ref_w, _, _, validity = _object_reference_state_w(env, reference_cfg, object_name)
    if not in_base_frame:
        return instinct_math.quat_to_tan_norm(quat_ref_w) * validity.unsqueeze(-1)
    robot = env.scene[robot_cfg.name]
    quat_ref_b = math_utils.quat_mul(math_utils.quat_inv(_root_quat(robot)), quat_ref_w)
    return instinct_math.quat_to_tan_norm(quat_ref_b) * validity.unsqueeze(-1)


def object_position_error(
    env: ManagerBasedEnv,
    object_entity_names: Sequence[str],
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_name: str = "box",
    in_base_frame: bool = True,
) -> torch.Tensor:
    object_pos_w, _, _, _ = _active_object_state_w(env, object_entity_names)
    pos_ref_w, _, _, _, validity = _object_reference_state_w(env, reference_cfg, object_name)
    error = object_pos_w - pos_ref_w
    if not in_base_frame:
        return error * validity.unsqueeze(-1)
    robot = env.scene[robot_cfg.name]
    return math_utils.quat_apply_inverse(_root_quat(robot), error) * validity.unsqueeze(-1)


def object_linear_velocity(
    env: ManagerBasedEnv,
    object_entity_names: Sequence[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    in_base_frame: bool = True,
) -> torch.Tensor:
    _, _, lin_vel_w, _ = _active_object_state_w(env, object_entity_names)
    if not in_base_frame:
        return lin_vel_w
    robot = env.scene[robot_cfg.name]
    return math_utils.quat_apply_inverse(_root_quat(robot), lin_vel_w)


def object_angular_velocity(
    env: ManagerBasedEnv,
    object_entity_names: Sequence[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    in_base_frame: bool = True,
) -> torch.Tensor:
    _, _, _, ang_vel_w = _active_object_state_w(env, object_entity_names)
    if not in_base_frame:
        return ang_vel_w
    robot = env.scene[robot_cfg.name]
    return math_utils.quat_apply_inverse(_root_quat(robot), ang_vel_w)


def _transform_local_points_to_world(
    object_pos_w: torch.Tensor,
    object_quat_w: torch.Tensor,
    points_local: torch.Tensor,
) -> torch.Tensor:
    flat_points = points_local.reshape(object_pos_w.shape[0], -1, 3)
    quat = object_quat_w[:, None, :].expand(-1, flat_points.shape[1], -1)
    points_w = object_pos_w[:, None, :] + math_utils.quat_apply(
        quat.reshape(-1, 4),
        flat_points.reshape(-1, 3),
    ).reshape_as(flat_points)
    return points_w.reshape_as(points_local)


def _metadata_body_names(data: Part2LinkMotionReferenceData) -> list[str]:
    names = list(getattr(data, "sparse_contact_robot_body_names", []) or [])
    if not names:
        raise ValueError("Part2Link NPZ metadata must contain `sparse_contact_robot_body_names`.")
    return names


def _resolve_metadata_body_ids(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg, data: Part2LinkMotionReferenceData):
    robot = env.scene[robot_cfg.name]
    body_names_raw = _metadata_body_names(data)
    resolved_names = [_BODY_ALIAS.get(name, name) for name in body_names_raw]
    cache_key = (robot_cfg.name, tuple(resolved_names))
    cache = getattr(env, "_part2link_body_id_cache", {})
    if cache_key in cache:
        return cache[cache_key]
    body_ids, matched_names = robot.find_bodies(resolved_names, preserve_order=True)
    if len(body_ids) != len(resolved_names):
        raise RuntimeError(
            "Failed to resolve all Part2Link sparse-contact robot bodies from NPZ metadata: "
            f"raw={body_names_raw}, resolved={resolved_names}, matched={matched_names}"
        )
    cache[cache_key] = (body_ids, matched_names, body_names_raw)
    env._part2link_body_id_cache = cache
    return cache[cache_key]


def validate_part2link_metadata(
    env: ManagerBasedEnv,
    metadata_root: str | Path | None = None,
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    object_name: str = "sofa",
) -> None:
    motion_ref: MotionReferenceManager = env.scene[reference_cfg.name]
    data: Part2LinkMotionReferenceData = motion_ref.data
    cache_key = (str(metadata_root), object_name)
    warned = getattr(env, "_part2link_metadata_validation_keys", set())
    if cache_key in warned:
        return
    warned.add(cache_key)
    env._part2link_metadata_validation_keys = warned

    _metadata_body_names(data)
    if metadata_root is None:
        return
    json_path = Path(metadata_root).expanduser() / f"{object_name}.json"
    if not json_path.exists():
        warnings.warn(f"Part2Link metadata JSON not found at {json_path}; NPZ metadata remains authoritative.")
        return

    with open(json_path) as f:
        json_data = json.load(f)
    npz_parts = [str(item) for item in getattr(data, "sparse_contact_part_names", [])]
    json_parts = [str(item) for item in json_data.get("object_parts_order", [])]
    json_parts_aliased = [_PART_ALIAS.get(name, name) for name in json_parts]
    if npz_parts and json_parts_aliased and npz_parts != json_parts_aliased:
        warnings.warn(
            "Part2Link JSON part order differs from NPZ sparse metadata; "
            f"NPZ is authoritative. json={json_parts}, json_alias={json_parts_aliased}, npz={npz_parts}",
            stacklevel=2,
        )
    relation = torch.as_tensor(json_data.get("relation", []), dtype=torch.float32)
    npz_relation = data.sparse_contact_relation_matrix[0, 0].detach().cpu()
    if relation.numel() > 0 and relation.shape == npz_relation.shape and not torch.equal(relation, npz_relation):
        warnings.warn("Part2Link JSON relation differs from NPZ relation; NPZ is authoritative.", stacklevel=2)


def _motion_frame_data(
    env: ManagerBasedEnv,
    reference_cfg: SceneEntityCfg,
    attr_name: str,
) -> torch.Tensor | None:
    motion_ref: MotionReferenceManager = env.scene[reference_cfg.name]
    data = motion_ref.data
    if not hasattr(data, attr_name):
        return None
    value = getattr(data, attr_name)
    if not isinstance(value, torch.Tensor):
        return None
    frame_ids = motion_ref.aiming_frame_idx.clamp(min=0)
    env_ids = motion_ref.ALL_INDICES
    return value[env_ids, frame_ids]


def _motion_validity(env: ManagerBasedEnv, reference_cfg: SceneEntityCfg) -> torch.Tensor:
    motion_ref: MotionReferenceManager = env.scene[reference_cfg.name]
    frame_ids = motion_ref.aiming_frame_idx.clamp(min=0)
    validity = motion_ref.data.validity[motion_ref.ALL_INDICES, frame_ids].to(torch.bool)
    validity = validity & (motion_ref.aiming_frame_idx >= 0)
    return validity


def _part2link_entity_names(env: ManagerBasedEnv, state: ObjectVariantRuntimeState) -> list[str]:
    names = [variant.name for variant in state.catalog.variants]
    missing = [name for name in names if name not in env.scene.entities]
    if missing:
        raise RuntimeError(f"Part2Link scene is missing object entities declared by the catalog: {missing}")
    return names


def _compute_part2link_vectors_and_mask(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg,
    reference_cfg: SceneEntityCfg,
    metadata_root: str | Path | None,
    object_name: str,
    debug_vis: bool = False,
    debug_vis_ignore_contact_phase: bool = False,
    debug_vis_max_envs: int = 1,
    debug_vis_show_contact_points: bool = True,
    debug_vis_show_part_centers: bool = True,
    debug_vis_point_radius: float = 0.025,
    debug_vis_center_radius: float = 0.035,
    debug_vis_line_radius: float = 0.008,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    motion_ref: MotionReferenceManager = env.scene[reference_cfg.name]
    data: Part2LinkMotionReferenceData = motion_ref.data
    validate_part2link_metadata(env, metadata_root=metadata_root, reference_cfg=reference_cfg, object_name=object_name)
    state = getattr(env, "_part2link_object_variant_state", None)
    if state is None:
        num_envs = env.num_envs
        num_links = len(_metadata_body_names(data))
        num_parts = len(getattr(data, "sparse_contact_part_names", []) or [0, 1, 2, 3])
        zeros = torch.zeros(num_envs, num_links, num_parts, 3, device=env.device)
        mask = torch.zeros(num_envs, num_links, num_parts, dtype=torch.bool, device=env.device)
        return zeros, zeros, mask

    robot = env.scene[robot_cfg.name]
    body_ids, _, body_aliases = _resolve_metadata_body_ids(env, robot_cfg, data)
    link_pos_w = robot.data.body_link_pos_w[:, body_ids, :]
    object_pos_w, object_quat_w, _, _ = _active_object_state_w(env, _part2link_entity_names(env, state))

    expanded_quat = object_quat_w[:, None, :].expand(-1, link_pos_w.shape[1], -1)
    link_pos_local = math_utils.quat_apply_inverse(
        expanded_quat.reshape(-1, 4),
        (link_pos_w - object_pos_w[:, None, :]).reshape(-1, 3),
    ).reshape_as(link_pos_w)
    current_vector = link_pos_local[:, :, None, :] - state.active_centers_local[:, None, :, :]

    gt_vector = _motion_frame_data(env, reference_cfg, "sparse_contact_link_part_center_vector_w")
    if gt_vector is None:
        gt_vector = torch.zeros_like(current_vector)
        valid_mask = torch.zeros(current_vector.shape[:-1], dtype=torch.bool, device=env.device)
        if debug_vis:
            env._part2link_debug_cache = None
        return current_vector, gt_vector, valid_mask
    gt_vector = gt_vector[:, : current_vector.shape[1], : current_vector.shape[2], :] * state.active_scale[
        :, None, None, None
    ]

    relation = _motion_frame_data(env, reference_cfg, "sparse_contact_relation_matrix")
    if relation is None:
        relation_mask = torch.ones(current_vector.shape[:-1], dtype=torch.bool, device=env.device)
    else:
        relation_mask = relation[:, : current_vector.shape[1], : current_vector.shape[2]] == 1

    point_phase = _motion_frame_data(env, reference_cfg, "sparse_contact_link_part_point_proximity")
    link_part_phase = _motion_frame_data(env, reference_cfg, "sparse_contact_link_part_proximity")
    if point_phase is not None and point_phase.numel() > 0 and torch.any(point_phase):
        phase_mask = point_phase[:, : current_vector.shape[1], : current_vector.shape[2]].any(dim=-1)
    elif link_part_phase is not None and link_part_phase.numel() > 0 and torch.any(link_part_phase):
        phase_mask = link_part_phase[:, : current_vector.shape[1], : current_vector.shape[2]] > 0
    else:
        phase_mask = torch.ones_like(relation_mask)

    valid_mask = relation_mask & phase_mask & _motion_validity(env, reference_cfg)[:, None, None]

    if debug_vis:
        debug_valid_mask = valid_mask.clone()
        if debug_vis_ignore_contact_phase:
            debug_valid_mask = relation_mask.clone()
            if torch.norm(gt_vector, dim=-1).sum() < 1e-6:
                debug_valid_mask = torch.where(
                    torch.norm(current_vector, dim=-1) > 1e-6,
                    relation_mask,
                    debug_valid_mask,
                )
        env._part2link_debug_cache = {
            "object_pos_w": object_pos_w.detach(),
            "object_quat_w": object_quat_w.detach(),
            "link_pos_w": link_pos_w.detach(),
            "part_centers_local": state.active_centers_local.detach(),
            "part_points_local": state.active_points_local.detach(),
            "point_valid_mask": state.active_point_valid_mask.detach(),
            "gt_vector_local": gt_vector.detach(),
            "current_vector": current_vector.detach(),
            "valid_mask": debug_valid_mask.detach(),
            "body_aliases": body_aliases,
            "part_names": state.catalog.part_names,
            "max_envs": debug_vis_max_envs,
            "show_contact_points": debug_vis_show_contact_points,
            "show_part_centers": debug_vis_show_part_centers,
            "point_radius": debug_vis_point_radius,
            "center_radius": debug_vis_center_radius,
            "line_radius": debug_vis_line_radius,
        }

    return current_vector, gt_vector, valid_mask


def part2link_vector_guidance_observation(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    object_name: str = "sofa",
    metadata_root: str | Path | None = None,
    tracking_sigma: float = 0.25,
    tracking_tolerance: float = 0.08,
    precision_scale_range: tuple[float, float] = (0.8, 1.4),
    debug_vis: bool = False,
) -> torch.Tensor:
    del tracking_sigma, tracking_tolerance, precision_scale_range, debug_vis
    current_vector, gt_vector, valid_mask = _compute_part2link_vectors_and_mask(
        env=env,
        robot_cfg=robot_cfg,
        reference_cfg=reference_cfg,
        metadata_root=metadata_root,
        object_name=object_name,
        debug_vis=False,
    )
    current_vector = torch.where(valid_mask[..., None], current_vector, torch.zeros_like(current_vector))
    gt_vector = torch.where(valid_mask[..., None], gt_vector, torch.zeros_like(gt_vector))
    error_norm = torch.linalg.vector_norm(current_vector - gt_vector, dim=-1)
    return torch.cat(
        [
            current_vector.reshape(env.num_envs, -1),
            gt_vector.reshape(env.num_envs, -1),
            error_norm.reshape(env.num_envs, -1),
            valid_mask.to(current_vector.dtype).reshape(env.num_envs, -1),
        ],
        dim=-1,
    )


def part2link_vector_guidance_gauss(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    object_name: str = "sofa",
    metadata_root: str | Path | None = None,
    tracking_sigma: float = 0.25,
    tracking_tolerance: float = 0.08,
    precision_scale_range: tuple[float, float] = (0.8, 1.4),
    debug_vis: bool = False,
    debug_vis_max_envs: int = 1,
    debug_vis_show_contact_points: bool = True,
    debug_vis_show_part_centers: bool = True,
    debug_vis_point_radius: float = 0.025,
    debug_vis_center_radius: float = 0.035,
    debug_vis_line_radius: float = 0.008,
    debug_vis_ignore_contact_phase: bool = False,
) -> torch.Tensor:
    del precision_scale_range
    state = getattr(env, "_part2link_object_variant_state", None)
    current_vector, gt_vector, valid_mask = _compute_part2link_vectors_and_mask(
        env=env,
        robot_cfg=robot_cfg,
        reference_cfg=reference_cfg,
        metadata_root=metadata_root,
        object_name=object_name,
        debug_vis=debug_vis,
        debug_vis_ignore_contact_phase=debug_vis_ignore_contact_phase,
        debug_vis_max_envs=debug_vis_max_envs,
        debug_vis_show_contact_points=debug_vis_show_contact_points,
        debug_vis_show_part_centers=debug_vis_show_part_centers,
        debug_vis_point_radius=debug_vis_point_radius,
        debug_vis_center_radius=debug_vis_center_radius,
        debug_vis_line_radius=debug_vis_line_radius,
    )
    if state is None or not torch.any(valid_mask):
        return torch.zeros(env.num_envs, dtype=current_vector.dtype, device=env.device)

    error = torch.linalg.vector_norm(current_vector - gt_vector, dim=-1)
    precision = state.active_precision_scale
    sigma = float(tracking_sigma) * precision
    tolerance = float(tracking_tolerance) * precision
    error = torch.clamp(error - tolerance[:, None, None], min=0.0)
    reward_pair = torch.exp(-torch.square(error) / torch.clamp(torch.square(sigma[:, None, None]), min=1.0e-6))
    reward_pair = torch.where(valid_mask, reward_pair, torch.zeros_like(reward_pair))
    denom = valid_mask.float().sum(dim=(1, 2)).clamp(min=1.0)
    return reward_pair.sum(dim=(1, 2)) / denom


def part2link_forbidden_contact_penalty(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    object_name: str = "sofa",
    metadata_root: str | Path | None = None,
    forbidden_distance_threshold: float = 0.10,
) -> torch.Tensor:
    motion_ref: MotionReferenceManager = env.scene[reference_cfg.name]
    data: Part2LinkMotionReferenceData = motion_ref.data
    validate_part2link_metadata(env, metadata_root=metadata_root, reference_cfg=reference_cfg, object_name=object_name)
    state = getattr(env, "_part2link_object_variant_state", None)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)

    relation = _motion_frame_data(env, reference_cfg, "sparse_contact_relation_matrix")
    if relation is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene[robot_cfg.name]
    body_ids, _, _ = _resolve_metadata_body_ids(env, robot_cfg, data)
    link_pos_w = robot.data.body_link_pos_w[:, body_ids, :]
    object_pos_w, object_quat_w, _, _ = _active_object_state_w(env, _part2link_entity_names(env, state))
    part_points_w = _transform_local_points_to_world(object_pos_w, object_quat_w, state.active_points_local)

    num_links = min(link_pos_w.shape[1], relation.shape[1])
    num_parts = min(part_points_w.shape[1], relation.shape[2])
    link_pos_w = link_pos_w[:, :num_links, :]
    part_points_w = part_points_w[:, :num_parts, :, :]
    point_valid_mask = state.active_point_valid_mask[:, :num_parts, :]
    forbidden_mask = relation[:, :num_links, :num_parts] == -1

    distances = torch.linalg.vector_norm(
        link_pos_w[:, :, None, None, :] - part_points_w[:, None, :, :, :],
        dim=-1,
    )
    distances = torch.where(point_valid_mask[:, None, :, :], distances, torch.full_like(distances, float("inf")))
    nearest_distance = distances.amin(dim=-1)

    threshold = max(float(forbidden_distance_threshold), 1.0e-6)
    violation = torch.clamp(threshold - nearest_distance, min=0.0)
    pair_penalty = torch.square(violation / threshold)
    pair_penalty = torch.where(forbidden_mask, pair_penalty, torch.zeros_like(pair_penalty))
    pair_penalty = torch.where(
        _motion_validity(env, reference_cfg)[:, None, None],
        pair_penalty,
        torch.zeros_like(pair_penalty),
    )
    denom = forbidden_mask.float().sum(dim=(1, 2)).clamp(min=1.0)
    return pair_penalty.sum(dim=(1, 2)) / denom


def seat_object_contact(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_names: Sequence[str] = ("pelvis_contour_link", "left_hip_roll_link", "right_hip_roll_link"),
    distance_threshold: float = 0.12,
) -> torch.Tensor:
    state = getattr(env, "_part2link_object_variant_state", None)
    if state is None:
        return torch.zeros(env.num_envs, len(body_names), device=env.device)
    if "seat" not in state.catalog.part_names:
        return torch.zeros(env.num_envs, len(body_names), device=env.device)
    seat_idx = state.catalog.part_names.index("seat")
    robot = env.scene[robot_cfg.name]
    body_ids, _ = robot.find_bodies(list(body_names), preserve_order=True)
    body_pos_w = robot.data.body_link_pos_w[:, body_ids, :]
    object_pos_w, object_quat_w, _, _ = _active_object_state_w(env, _part2link_entity_names(env, state))
    seat_points_w = _transform_local_points_to_world(object_pos_w, object_quat_w, state.active_points_local)[:, seat_idx]
    seat_mask = state.active_point_valid_mask[:, seat_idx]
    distances = torch.linalg.vector_norm(body_pos_w[:, :, None, :] - seat_points_w[:, None, :, :], dim=-1)
    distances = torch.where(seat_mask[:, None, :], distances, torch.full_like(distances, float("inf")))
    return (distances.amin(dim=-1) < float(distance_threshold)).float()


def object_contact_reference_phase(
    env: ManagerBasedRlEnv,
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    object_name: str = "box",
    normalize: bool = True,
    threshold: float = 0.12,
    penetration_penalty_scale: float = 1.0,
    print_reason: bool = False,
    debug_label: str = "seat_object_contact",
) -> torch.Tensor:
    del object_name, print_reason, debug_label
    current_contact = seat_object_contact(env, distance_threshold=threshold)
    motion_ref: MotionReferenceManager = env.scene[reference_cfg.name]
    data: Part2LinkMotionReferenceData = motion_ref.data
    validity = _motion_validity(env, reference_cfg).to(current_contact.dtype)
    reference_contact = validity

    part_names = [str(name) for name in getattr(data, "sparse_contact_part_names", [])]
    relation = _motion_frame_data(env, reference_cfg, "sparse_contact_relation_matrix")
    if part_names and relation is not None:
        seat_indices = [idx for idx, name in enumerate(part_names[: relation.shape[2]]) if name == "seat"]
        if seat_indices:
            seat_idx = torch.as_tensor(seat_indices, dtype=torch.long, device=env.device)
            relation_seat = relation.index_select(2, seat_idx) == 1
            point_phase = _motion_frame_data(env, reference_cfg, "sparse_contact_link_part_point_proximity")
            link_part_phase = _motion_frame_data(env, reference_cfg, "sparse_contact_link_part_proximity")
            if point_phase is not None and point_phase.shape[2] >= relation.shape[2]:
                phase_seat = point_phase.index_select(2, seat_idx).any(dim=-1)
            elif link_part_phase is not None and link_part_phase.shape[2] >= relation.shape[2]:
                phase_seat = link_part_phase.index_select(2, seat_idx) > 0
            else:
                phase_seat = torch.ones_like(relation_seat, dtype=torch.bool)
            reference_contact = (relation_seat & phase_seat).any(dim=(1, 2)).to(current_contact.dtype) * validity

    contact_score = current_contact.sum(dim=-1)
    reward = reference_contact * contact_score - (1.0 - reference_contact) * penetration_penalty_scale * contact_score
    if normalize and current_contact.shape[-1] > 0:
        reward = reward / current_contact.shape[-1]
    return reward * validity


def any_object_filtered_contact(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_names: Sequence[str] = (
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ),
    distance_threshold: float = 0.08,
    print_reason: bool = False,
) -> torch.Tensor:
    del print_reason
    state = getattr(env, "_part2link_object_variant_state", None)
    if state is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    nonseat_indices = [idx for idx, name in enumerate(state.catalog.part_names) if name != "seat"]
    if not nonseat_indices:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    robot = env.scene[robot_cfg.name]
    body_ids, _ = robot.find_bodies(list(body_names), preserve_order=True)
    body_pos_w = robot.data.body_link_pos_w[:, body_ids, :]
    object_pos_w, object_quat_w, _, _ = _active_object_state_w(env, _part2link_entity_names(env, state))
    points_w = _transform_local_points_to_world(object_pos_w, object_quat_w, state.active_points_local[:, nonseat_indices])
    point_mask = state.active_point_valid_mask[:, nonseat_indices]
    distances = torch.linalg.vector_norm(body_pos_w[:, :, None, None, :] - points_w[:, None, :, :, :], dim=-1)
    distances = torch.where(point_mask[:, None, :, :], distances, torch.full_like(distances, float("inf")))
    return distances.amin(dim=(1, 2, 3)) < float(distance_threshold)


class TrackingSigmaCurriculum(ManagerTermBase):
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self.reward_group_name = cfg.params.get("reward_group_name", None)
        self.param_name = cfg.params.get("param_name", "tracking_sigma")
        self.term_names = list(cfg.params.get("term_names", []))
        if not self.term_names:
            raise ValueError("TrackingSigmaCurriculum requires `term_names`.")
        self.initial_sigmas = self._expand_param(cfg.params.get("initial_sigmas", None), "initial_sigmas")
        self.final_sigmas = self._expand_param(cfg.params.get("final_sigmas", None), "final_sigmas")

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: Sequence[int],
        start_step: int = 0,
        end_step: int = 1_000_000,
        reward_group_name: str | None = None,
        term_names: Sequence[str] | None = None,
        param_name: str = "tracking_sigma",
        initial_sigmas: float | Sequence[float] | None = None,
        final_sigmas: float | Sequence[float] | None = None,
        min_sigma: float = 1.0e-3,
    ) -> dict[str, float]:
        del env_ids, term_names, param_name, initial_sigmas, final_sigmas
        reward_group_name = self.reward_group_name if reward_group_name is None else reward_group_name
        if end_step <= start_step:
            progress = 1.0 if env.common_step_counter >= start_step else 0.0
        else:
            progress = (float(env.common_step_counter) - float(start_step)) / float(end_step - start_step)
            progress = min(max(progress, 0.0), 1.0)
        log_dict = {"tracking_sigma_progress": progress}
        for term_name, initial_sigma, final_sigma in zip(self.term_names, self.initial_sigmas, self.final_sigmas):
            current_sigma = max(initial_sigma + progress * (final_sigma - initial_sigma), min_sigma)
            term_cfg = env.reward_manager.get_term_cfg(term_name, group_name=reward_group_name)
            term_cfg.params[self.param_name] = float(current_sigma)
            log_dict[f"{term_name}_{self.param_name}"] = float(current_sigma)
        return log_dict

    def _expand_param(self, value: float | Sequence[float] | None, name: str) -> list[float]:
        if value is None:
            if name != "initial_sigmas":
                raise ValueError(f"TrackingSigmaCurriculum requires `{name}`.")
            values = []
            for term_name in self.term_names:
                term_cfg = self._env.reward_manager.get_term_cfg(term_name, group_name=self.reward_group_name)
                values.append(float(term_cfg.params[self.param_name]))
            return values
        if isinstance(value, (int, float)):
            return [float(value)] * len(self.term_names)
        values = [float(item) for item in value]
        if len(values) != len(self.term_names):
            raise ValueError(f"`{name}` length must match term_names length.")
        return values


class ObjectAlphaCurriculum(ManagerTermBase):
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self.initial_alpha = float(cfg.params.get("initial_alpha", 1.0))
        self.final_alpha = float(cfg.params.get("final_alpha", 0.0))

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: Sequence[int],
        initial_alpha: float = 1.0,
        final_alpha: float = 0.0,
        start_step: int = 0,
        end_step: int = 3_000_000,
    ) -> dict[str, float]:
        del env_ids
        initial_alpha = self.initial_alpha if initial_alpha == 1.0 else float(initial_alpha)
        final_alpha = self.final_alpha if final_alpha == 0.0 else float(final_alpha)
        if end_step <= start_step:
            progress = 1.0 if env.common_step_counter >= start_step else 0.0
        else:
            progress = (float(env.common_step_counter) - float(start_step)) / float(end_step - start_step)
            progress = min(max(progress, 0.0), 1.0)
        alpha = initial_alpha + progress * (final_alpha - initial_alpha)
        env._part2link_object_current_alpha = float(alpha)
        return {"object_alpha_progress": progress, "object_alpha": float(alpha)}


# ---------------------------------------------------------------------------
# Part2Link debug visualization
# ---------------------------------------------------------------------------

_CONTACT_POINT_COLOR = (1.0, 0.85, 0.1, 1.0)
_PART_CENTER_COLOR = (1.0, 0.35, 0.05, 1.0)
_REALTIME_VECTOR_COLOR = (1.0, 0.05, 0.05, 1.0)
_GT_VECTOR_COLOR = (0.05, 0.35, 1.0, 1.0)


def _draw_part2link_debug(env, visualizer) -> None:
    cache = getattr(env, "_part2link_debug_cache", None)
    if cache is None:
        return

    num_envs = env.num_envs
    env_ids = list(visualizer.get_env_indices(num_envs))
    if not env_ids:
        return
    max_envs = min(cache["max_envs"], len(env_ids))
    selected_envs = env_ids[:max_envs]

    object_pos_w = cache["object_pos_w"]
    object_quat_w = cache["object_quat_w"]
    link_pos_w = cache["link_pos_w"]
    part_centers_local = cache["part_centers_local"]
    part_points_local = cache["part_points_local"]
    point_valid_mask = cache["point_valid_mask"]
    gt_vector_local = cache["gt_vector_local"]
    current_vector = cache["current_vector"]
    valid_mask = cache["valid_mask"]

    show_contact_points = cache["show_contact_points"]
    show_part_centers = cache["show_part_centers"]
    point_radius = cache["point_radius"]
    center_radius = cache["center_radius"]
    line_radius = cache["line_radius"]

    for env_idx in selected_envs:
        obj_pos = object_pos_w[env_idx]  # [3]
        obj_quat = object_quat_w[env_idx]  # [4]

        # Transform part centers and points to world frame
        part_centers_w = math_utils.quat_apply(
            obj_quat.expand(part_centers_local[env_idx].shape[0], -1).reshape(-1, 4),
            part_centers_local[env_idx].reshape(-1, 3),
        ).reshape_as(part_centers_local[env_idx]) + obj_pos

        part_points_w = math_utils.quat_apply(
            obj_quat.expand(part_points_local[env_idx].shape[0] * part_points_local[env_idx].shape[1], -1).reshape(-1, 4),
            part_points_local[env_idx].reshape(-1, 3),
        ).reshape_as(part_points_local[env_idx]) + obj_pos

        num_links = valid_mask.shape[1]
        num_parts = valid_mask.shape[2]

        for link_idx in range(min(num_links, link_pos_w.shape[1])):
            link_pos = link_pos_w[env_idx, link_idx]  # [3]

            for part_idx in range(min(num_parts, part_centers_w.shape[0])):
                if not valid_mask[env_idx, link_idx, part_idx]:
                    continue

                part_center_w = part_centers_w[part_idx]  # [3]

                # GT vector: part_center_w → gt_endpoint_w
                gt_vec_local = gt_vector_local[env_idx, link_idx, part_idx]  # [3]
                gt_endpoint_w = math_utils.quat_apply(obj_quat.unsqueeze(0), (part_centers_local[env_idx, part_idx] + gt_vec_local).unsqueeze(0))[0] + obj_pos

                # Realtime vector: part_center_w → link_pos
                visualizer.add_cylinder(
                    start=(float(part_center_w[0]), float(part_center_w[1]), float(part_center_w[2])),
                    end=(float(link_pos[0]), float(link_pos[1]), float(link_pos[2])),
                    radius=line_radius,
                    color=_REALTIME_VECTOR_COLOR,
                )

                # GT vector: part_center_w → gt_endpoint_w
                visualizer.add_cylinder(
                    start=(float(part_center_w[0]), float(part_center_w[1]), float(part_center_w[2])),
                    end=(float(gt_endpoint_w[0]), float(gt_endpoint_w[1]), float(gt_endpoint_w[2])),
                    radius=line_radius,
                    color=_GT_VECTOR_COLOR,
                )

                # Contact points
                if show_contact_points:
                    for pt_idx in range(part_points_w.shape[1]):
                        if not point_valid_mask[env_idx, part_idx, pt_idx]:
                            continue
                        pt_w = part_points_w[part_idx, pt_idx]
                        visualizer.add_sphere(
                            center=(float(pt_w[0]), float(pt_w[1]), float(pt_w[2])),
                            radius=point_radius,
                            color=_CONTACT_POINT_COLOR,
                        )

        # Part centers
        if show_part_centers:
            for part_idx in range(num_parts):
                if not valid_mask[env_idx, :, part_idx].any():
                    continue
                pc_w = part_centers_w[part_idx]
                visualizer.add_sphere(
                    center=(float(pc_w[0]), float(pc_w[1]), float(pc_w[2])),
                    radius=center_radius,
                    color=_PART_CENTER_COLOR,
                )


class Part2LinkDebugVisualizer(MonitorTerm):
    """Monitor-compatible debug visualizer for Part2Link reward markers.

    Reads ``env._part2link_debug_cache`` populated by
    :func:`part2link_vector_guidance_gauss` when ``debug_vis=True``, and draws
    spheres and cylinders via mjlab :class:`~mjlab.viewer.debug_visualizer.DebugVisualizer`.

    Registration example::

        from instinct_mj.monitors import MonitorTermCfg
        monitors["part2link_debug"] = MonitorTermCfg(
            func=Part2LinkDebugVisualizer,
        )
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def get_log(self, is_episode=False) -> dict[str, float]:
        del is_episode
        return {}

    def debug_vis(self, visualizer) -> None:
        _draw_part2link_debug(self._env, visualizer)
