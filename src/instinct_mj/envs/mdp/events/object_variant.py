"""Generalized object variant management for domain randomization.

This module extracts and generalizes the variant catalog, runtime state, and
variant-switching event functions from :mod:`instinct_mj.tasks.interaction.mdp.part2link`
so they can be reused across tasks (HOI, Part2Link, etc.).

.. note::
    ``chair_name`` / ``chair_names`` fields are retained for backward
    compatibility with the Part2Link interaction task.  New code should
    prefer the ``object_type`` / ``object_types`` aliases.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedEnv

    from instinct_mj.motion_reference.motion_reference_manager import MotionReferenceManager

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectVariant:
    """A single variant of a scene object (one mesh + optional contact data).

    Retained for backward-compatibility:
        ``chair_name`` — same as ``object_type``.
    """

    name: str
    """Entity name in the scene, e.g. ``"chair_14_alpha_1p00"``."""

    chair_name: str
    """Backward-compatible alias for ``object_type``."""

    alpha: float
    """Morph / canonical-shape parameter (1.0 = reference, 0.0 = hardest)."""

    mesh_path: Path
    """Absolute path to the mesh file (``.glb`` / ``.usd`` / ``.obj``)."""

    contact_points_path: Path
    """Absolute path to ``contact_points_local.npz``."""

    alpha_index: int
    """Index of this variant's alpha in the NPZ ``alpha_values`` array."""

    variant_index: int = 0
    """Zero-based index within the type's variant list."""

    metadata: dict = field(default_factory=dict)
    """Arbitrary per-variant metadata (scale hints, part counts, …)."""

    @property
    def object_type(self) -> str:
        """Semantic object type (same as ``chair_name``)."""
        return self.chair_name


@dataclass
class ObjectVariantCatalog:
    """Catalog of all object variants loaded for a scene.

    Retained for backward-compatibility:
        ``chair_names`` — same as ``variant_names`` element-per-variant list.
    """

    variants: list[ObjectVariant]
    """All discovered variants (order is deterministic)."""

    variant_names: list[str]
    """Entity name per variant."""

    chair_names: list[str]
    """Chair / object-type name per variant (backward-compatible)."""

    part_names: list[str]
    """Part names (e.g. ``["seat", "backrest", …]``).  Empty if no contact data."""

    centers_local: torch.Tensor | None = None
    """``[num_variants, num_parts, 3]`` part centres in object-local frame."""

    points_local: torch.Tensor | None = None
    """``[num_variants, num_parts, num_points, 3]`` contact points in object-local frame."""

    point_valid_mask: torch.Tensor | None = None
    """``[num_variants, num_parts, num_points]`` boolean mask (``True`` = valid point)."""

    @property
    def object_types(self) -> list[str]:
        """Unique object-type names in catalog insertion order.

        .. warning::
           Order is **not** alphabetical — it reflects the sequence in which
           variants were appended to the catalog.  This is load-bearing for
           motion-reference slot indexing.
        """
        return list(dict.fromkeys(self.chair_names))

    @property
    def type_to_variant_indices(self) -> dict[str, list[int]]:
        """Map ``object_type`` → list of variant indices in catalog order."""
        mapping: dict[str, list[int]] = {}
        for idx, variant in enumerate(self.variants):
            mapping.setdefault(variant.object_type, []).append(idx)
        return mapping


@dataclass
class ObjectVariantRuntimeState:
    """Per-environment, per-object-type active-variant tracking.

    All ``active_*`` tensors have shape ``[num_envs, num_object_types]``
    (or ``[num_envs, num_object_types, …]`` for contact data).  When the
    catalog contains only one object type the type dimension collapses to 1,
    preserving Part2Link backward compatibility.

    Created lazily on first reset and stored as
    ``env._object_variant_state``.
    """

    catalog: ObjectVariantCatalog
    active_variant_ids: torch.Tensor  # [num_envs, num_object_types] long
    active_alpha: torch.Tensor  # [num_envs, num_object_types] float
    active_scale: torch.Tensor  # [num_envs, num_object_types] float
    active_precision_scale: torch.Tensor  # [num_envs, num_object_types] float
    active_centers_local: torch.Tensor | None  # [num_envs, num_object_types, P, 3]
    active_points_local: torch.Tensor | None  # [num_envs, num_object_types, P, N, 3]
    active_point_valid_mask: torch.Tensor | None  # [num_envs, num_object_types, P, N]
    reference_position_offsets: torch.Tensor  # [num_envs, 3]


# ---------------------------------------------------------------------------
# Constants & utilities
# ---------------------------------------------------------------------------

_CATALOG_CACHE: dict[tuple, ObjectVariantCatalog] = {}
_INACTIVE_OBJECT_SPACING = 20.0


def _alpha_to_token(alpha: float) -> str:
    return f"alpha_{float(alpha):.2f}".replace(".", "p")


def _alpha_to_glb_name(alpha: float) -> str:
    return f"alpha_{float(alpha):.2f}.glb"


def _sanitize_variant_name(chair_name: str, alpha: float) -> str:
    return f"{chair_name}_{_alpha_to_token(alpha)}".replace("-", "_").replace(".", "p")


def _parse_object_type_names(value: str | Sequence[str] | None) -> tuple[str, ...] | None:
    """Parse comma-separated object-type names into a tuple."""
    if value is None:
        return None
    if isinstance(value, str):
        names = tuple(item.strip() for item in value.split(",") if item.strip())
        return names or None
    names = tuple(str(item) for item in value if str(item))
    return names or None


def _parse_alpha_values(
    value: str | Sequence[float] | None, default: Sequence[float]
) -> tuple[float, ...]:
    if value is None:
        return tuple(float(alpha) for alpha in default)
    if isinstance(value, str):
        alphas = tuple(float(item.strip()) for item in value.split(",") if item.strip())
        return alphas or tuple(float(alpha) for alpha in default)
    return tuple(float(alpha) for alpha in value)


def _resolve_object_dirs(
    extension_root: str | Path, object_types: Sequence[str] | None
) -> list[Path]:
    """Discover object-type directories under *extension_root*.

    A directory is valid when it contains:
      - ``ffd_bbox_coarse/morph_path/`` (mesh variants)
      - ``contact_point_transfer/contact_points_local.npz``
    """
    root = Path(extension_root).expanduser().resolve()
    if object_types is None or len(object_types) == 0:
        candidates = sorted(path for path in root.iterdir() if path.is_dir())
    else:
        candidates = [(root / str(name)).resolve() for name in object_types]
    valid_dirs = [
        path
        for path in candidates
        if (path / "ffd_bbox_coarse" / "morph_path").is_dir()
        and (path / "contact_point_transfer" / "contact_points_local.npz").exists()
    ]
    if not valid_dirs:
        raise FileNotFoundError(
            f"No valid object variant directories found under extension_root={root}"
        )
    return valid_dirs


def _resolve_mesh_path(
    object_dir: Path, alpha: float, asset_cache: str | Path | None
) -> Path:
    """Resolve a mesh file for *object_dir* / *alpha*, checking cache first."""
    if asset_cache is not None:
        cache_dir = Path(asset_cache).expanduser().resolve() / object_dir.name
        for candidate in (
            cache_dir / f"{_alpha_to_token(alpha)}.obj",
            cache_dir / f"{_alpha_to_glb_name(alpha)}.obj",
            cache_dir / "mesh.obj",
        ):
            if candidate.exists():
                return candidate

    morph_dir = object_dir / "ffd_bbox_coarse" / "morph_path"
    glb_path = morph_dir / _alpha_to_glb_name(alpha)
    if glb_path.exists():
        return glb_path.resolve()

    usd_path = morph_dir / f"{_alpha_to_token(alpha)}.usd"
    if usd_path.exists():
        return usd_path.resolve()

    raise FileNotFoundError(
        f"No mesh found for object={object_dir.name}, alpha={alpha:.2f} under {morph_dir}"
    )


def _read_part_names(contact_data: np.lib.npyio.NpzFile, num_parts: int) -> list[str]:
    for key in ("points_names", "part_names", "parts_names"):
        if key in contact_data:
            return [str(item) for item in contact_data[key].tolist()]
    return [f"part_{idx}" for idx in range(num_parts)]


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------


def load_object_variant_catalog(
    extension_root: str | Path,
    chair_names: Sequence[str] | str | None = None,
    alpha_values: Sequence[float] | str | None = None,
    asset_cache: str | Path | None = None,
    device: str | torch.device = "cpu",
    single_object_type: str | None = None,
) -> ObjectVariantCatalog:
    """Load (or retrieve from cache) a catalog of object variants.

    Parameters
    ----------
    extension_root:
        Root directory containing per-object-type subdirectories.
    chair_names:
        Optional filter: only load variants for these object types.  ``None``
        discovers all valid directories under *extension_root*.
    alpha_values:
        Morph alpha values to load (default ``(1.0,)``).
    asset_cache:
        Optional directory of pre-converted ``.obj`` meshes.
    device:
        Target device for catalog tensors.
    single_object_type:
        When set, **all** loaded variants are assigned this object-type name,
        collapsing multiple chair directories into variants of a single type.
        This is the canonical mode for Part2Link (``single_object_type="box"``).
    """
    parsed_chairs = _parse_object_type_names(chair_names)
    parsed_alphas = _parse_alpha_values(alpha_values, default=(1.0,))
    cache_key = (
        str(Path(extension_root).expanduser().resolve()),
        str(Path(asset_cache).expanduser().resolve()) if asset_cache is not None else None,
        parsed_chairs,
        parsed_alphas,
        single_object_type,
    )
    if cache_key in _CATALOG_CACHE:
        catalog = _CATALOG_CACHE[cache_key]
        if catalog.centers_local is not None and catalog.centers_local.device == torch.device(device):
            return catalog
        if catalog.centers_local is None:
            return catalog

    variants: list[ObjectVariant] = []
    centers: list[np.ndarray] = []
    points: list[np.ndarray] = []
    point_masks: list[np.ndarray] = []
    chair_name_values: list[str] = []
    variant_names: list[str] = []
    part_names: list[str] | None = None

    for object_dir in _resolve_object_dirs(extension_root, parsed_chairs):
        contact_path = object_dir / "contact_point_transfer" / "contact_points_local.npz"
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

        variant_idx = 0
        for target_alpha in parsed_alphas:
            alpha_idx = int(np.argmin(np.abs(contact_alphas - float(target_alpha))))
            alpha = float(contact_alphas[alpha_idx])
            if abs(alpha - float(target_alpha)) > 1.0e-4:
                warnings.warn(
                    f"Requested alpha={target_alpha:.2f} for {object_dir.name}, "
                    f"using nearest alpha={alpha:.2f}.",
                    stacklevel=2,
                )
            mesh_path = _resolve_mesh_path(object_dir, alpha, asset_cache)
            variant_name = _sanitize_variant_name(object_dir.name, alpha)
            type_name = single_object_type if single_object_type is not None else object_dir.name
            variants.append(
                ObjectVariant(
                    name=variant_name,
                    chair_name=type_name,
                    alpha=alpha,
                    mesh_path=mesh_path,
                    contact_points_path=contact_path.resolve(),
                    alpha_index=alpha_idx,
                    variant_index=variant_idx,
                    metadata={"alpha": alpha},
                )
            )
            variant_idx += 1
            centers.append(np.nan_to_num(centers_local[alpha_idx]).astype(np.float32))
            points.append(np.nan_to_num(points_local[alpha_idx]).astype(np.float32))
            point_masks.append(np.isfinite(points_local[alpha_idx]).all(axis=-1))
            chair_name_values.append(type_name)
            variant_names.append(variant_name)

    if not variants:
        raise FileNotFoundError(
            f"No object variants found under extension_root={extension_root}"
        )

    catalog = ObjectVariantCatalog(
        variants=variants,
        variant_names=variant_names,
        chair_names=chair_name_values,
        part_names=part_names or [],
        centers_local=torch.as_tensor(np.stack(centers), dtype=torch.float32, device=device),
        points_local=torch.as_tensor(np.stack(points), dtype=torch.float32, device=device),
        point_valid_mask=torch.as_tensor(np.stack(point_masks), dtype=torch.bool, device=device),
    )
    _CATALOG_CACHE[cache_key] = catalog
    return catalog


# ---------------------------------------------------------------------------
# Runtime state helpers
# ---------------------------------------------------------------------------


def _get_or_create_variant_state(
    env: ManagerBasedEnv,
    extension_root: str | Path = "",
    chair_names: Sequence[str] | str | None = None,
    alpha_values: Sequence[float] | str | None = None,
    asset_cache: str | Path | None = None,
    catalog: ObjectVariantCatalog | None = None,
) -> ObjectVariantRuntimeState:
    """Lazy-initialise :class:`ObjectVariantRuntimeState` on *env*.

    When *catalog* is provided it is used directly (no disk I/O).
    Otherwise a catalog is loaded from *extension_root*.
    """
    state = getattr(env, "_part2link_object_variant_state", None)
    if state is not None:
        return state

    if catalog is None:
        catalog = load_object_variant_catalog(
            extension_root,
            chair_names=chair_names,
            alpha_values=alpha_values,
            asset_cache=asset_cache,
            device=env.device,
        )
    elif catalog.centers_local is not None and catalog.centers_local.device != torch.device(env.device):
        # Move pre-built catalog tensors to the environment device.
        catalog.centers_local = catalog.centers_local.to(env.device)
        if catalog.points_local is not None:
            catalog.points_local = catalog.points_local.to(env.device)
        if catalog.point_valid_mask is not None:
            catalog.point_valid_mask = catalog.point_valid_mask.to(env.device)
    num_envs = env.num_envs
    num_types = len(catalog.object_types)
    num_parts = catalog.centers_local.shape[1] if catalog.centers_local is not None else 0
    num_points = catalog.points_local.shape[2] if catalog.points_local is not None else 0

    if catalog.centers_local is not None:
        centers = torch.zeros(num_envs, num_types, num_parts, 3,
                              dtype=torch.float32, device=env.device)
    else:
        centers = None
    if catalog.points_local is not None:
        pts = torch.zeros(num_envs, num_types, num_parts, num_points, 3,
                          dtype=torch.float32, device=env.device)
        mask = torch.zeros(num_envs, num_types, num_parts, num_points,
                           dtype=torch.bool, device=env.device)
    else:
        pts = None
        mask = None

    state = ObjectVariantRuntimeState(
        catalog=catalog,
        active_variant_ids=torch.zeros(num_envs, num_types, dtype=torch.long, device=env.device),
        active_alpha=torch.zeros(num_envs, num_types, dtype=torch.float32, device=env.device),
        active_scale=torch.ones(num_envs, num_types, dtype=torch.float32, device=env.device),
        active_precision_scale=torch.ones(num_envs, num_types, dtype=torch.float32, device=env.device),
        active_centers_local=centers,
        active_points_local=pts,
        active_point_valid_mask=mask,
        reference_position_offsets=torch.zeros(num_envs, 3, dtype=torch.float32, device=env.device),
    )
    env._part2link_object_variant_state = state
    env._interaction_object_variant_state = state
    env._object_variant_state = state
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
    raise AttributeError(
        f"Entity '{getattr(asset, 'name', '<unknown>')}' has no root position tensor."
    )


def _root_quat(asset) -> torch.Tensor:
    data = asset.data
    for name in ("root_link_quat_w", "root_quat_w"):
        if hasattr(data, name):
            return _safe_unit_quat(getattr(data, name))
    raise AttributeError(
        f"Entity '{getattr(asset, 'name', '<unknown>')}' has no root orientation tensor."
    )


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


def _write_pose_velocity(
    asset, root_pose: torch.Tensor, root_velocity: torch.Tensor, env_ids: torch.Tensor
) -> None:
    if getattr(asset, "is_fixed_base", False):
        if not getattr(asset, "is_mocap", False):
            raise ValueError("Object entities must be mocap or free bodies.")
        asset.write_mocap_pose_to_sim(root_pose, env_ids=env_ids)
        return
    asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(root_velocity, env_ids=env_ids)


def _scene_object_slot(motion_ref: MotionReferenceManager, object_name: str) -> int:
    scene_object_names = getattr(motion_ref.cfg, "scene_object_names", None)
    if not scene_object_names:
        raise RuntimeError(
            f"motion_ref.cfg.scene_object_names is empty; cannot resolve slot for '{object_name}'."
        )
    if object_name not in scene_object_names:
        raise RuntimeError(
            f"Object name '{object_name}' is not in "
            f"motion_ref.scene_object_names={scene_object_names}. "
            f"Check that scene_object_names matches the catalog object_types."
        )
    return int(scene_object_names.index(object_name))


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


def _select_variants_for_reset(
    env: ManagerBasedEnv,
    state: ObjectVariantRuntimeState, num_reset: int, device: torch.device
) -> torch.Tensor:
    """Per-object-type sampling: for each type, sample one variant per env.

    Curriculum-aware selection:
    - ``env._object_variant_active_count`` limits the number of candidate
      variants per type (early training uses fewer variants).
    - ``env._object_variant_current_alpha`` biases selection toward the
      nearest alpha when variants carry alpha metadata.

    Returns
    -------
    torch.Tensor
        ``[num_reset, num_object_types]`` long — the catalog index of the
        selected variant for each (env, object_type) pair.
    """
    object_types = state.catalog.object_types
    num_types = len(object_types)
    type_to_idx = state.catalog.type_to_variant_indices

    # Curriculum: limit number of candidate variants per type
    active_count = getattr(env, "_object_variant_active_count", None)
    # Curriculum: current alpha target
    current_alpha = getattr(env, "_object_variant_current_alpha", None)

    selected = torch.zeros(num_reset, num_types, dtype=torch.long, device=device)
    for ti, obj_type in enumerate(object_types):
        candidates = list(type_to_idx.get(obj_type, []))
        if not candidates:
            candidates = [
                idx
                for idx, variant in enumerate(state.catalog.variants)
                if variant.object_type == obj_type
            ]
        if not candidates:
            raise RuntimeError(f"No variant for object type '{obj_type}'.")

        # Honour active_count curriculum: only use first N candidates
        if active_count is not None:
            max_count = max(1, int(active_count))
            candidates = candidates[:max_count]

        # Alpha-based selection: pick nearest alpha stage, then randomize among variants at that stage.
        if current_alpha is not None and len(candidates) > 1:
            alphas = [
                float(state.catalog.variants[ci].metadata.get("alpha", 1.0))
                for ci in candidates
            ]
            alpha_diffs = np.asarray([abs(a - float(current_alpha)) for a in alphas], dtype=np.float64)
            min_diff = float(alpha_diffs.min())
            nearest_alpha = max(
                alpha for alpha, diff in zip(alphas, alpha_diffs, strict=False) if diff <= min_diff + 1.0e-8
            )
            nearest_candidates = [
                candidate
                for candidate, alpha in zip(candidates, alphas, strict=False)
                if abs(alpha - nearest_alpha) <= 1.0e-6
            ]
            idx = torch.randint(len(nearest_candidates), (num_reset,), device=device)
            selected[:, ti] = torch.tensor(
                [nearest_candidates[int(i)] for i in idx.cpu()], dtype=torch.long, device=device
            )
        else:
            # Uniform random among candidates for each env
            idx = torch.randint(len(candidates), (num_reset,), device=device)
            selected[:, ti] = torch.tensor(
                [candidates[int(i)] for i in idx.cpu()], dtype=torch.long, device=device
            )
    return selected


def _update_active_variant_state(
    state: ObjectVariantRuntimeState,
    env_ids: torch.Tensor,
    variant_ids: torch.Tensor,
    scales: torch.Tensor,
    precision_scales: torch.Tensor,
    reference_position_offsets: torch.Tensor,
) -> None:
    """Store per-env, per-type active variant metadata into *state*.

    Parameters
    ----------
    variant_ids:
        ``[num_reset, num_object_types]`` long.
    scales / precision_scales:
        ``[num_reset, num_object_types]`` float (or broadcast-compatible).
    """
    state.active_variant_ids[env_ids] = variant_ids
    # Compute active_alpha for each (env, type) pair
    num_types = variant_ids.shape[1]
    alpha_flat = []
    for ti in range(num_types):
        for vid in variant_ids[:, ti].detach().cpu().tolist():
            alpha_flat.append(state.catalog.variants[int(vid)].alpha)
    state.active_alpha[env_ids] = torch.tensor(
        alpha_flat, dtype=torch.float32, device=env_ids.device
    ).reshape(len(env_ids), num_types)
    state.active_scale[env_ids] = scales
    state.active_precision_scale[env_ids] = precision_scales
    if state.catalog.centers_local is not None:
        state.active_centers_local[env_ids] = (
            state.catalog.centers_local[variant_ids] * scales[:, :, None, None]
        )
    if state.catalog.points_local is not None:
        state.active_points_local[env_ids] = (
            state.catalog.points_local[variant_ids] * scales[:, :, None, None, None]
        )
    if state.catalog.point_valid_mask is not None:
        state.active_point_valid_mask[env_ids] = state.catalog.point_valid_mask[variant_ids]
    state.reference_position_offsets[env_ids] = reference_position_offsets


def _invalidate_dynamic_mesh_sensors(env: ManagerBasedEnv) -> None:
    for sensor in getattr(env.scene, "sensors", {}).values():
        if hasattr(sensor, "invalidate_mesh_buffers"):
            sensor.invalidate_mesh_buffers()


# ---------------------------------------------------------------------------
# Public event functions
# ---------------------------------------------------------------------------


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
    catalog: ObjectVariantCatalog | None = None,
    scale_distribution_params: tuple[float, float] = (1.0, 1.0),
    precision_scale_range: tuple[float, float] = (0.8, 1.4),
    base_lin_vel_ratio: float = 1.0,
    base_ang_vel_ratio: float = 1.0,
    apply_env_spacing: bool = False,
) -> None:
    """Reset all variant entities: activate one variant per env, hide the rest.

    On each reset, one variant per object-type is randomly selected.  The
    active variant is placed at the motion-reference trajectory; all inactive
    variants are moved far below the scene.

    Parameters
    ----------
    env:
        The manager-based environment.
    env_ids:
        Indices of environments being reset.
    object_entity_names:
        Scene entity names of *all* variant objects (active + inactive).
    extension_root:
        Root directory passed to :func:`load_object_variant_catalog` (ignored
        when *catalog* is provided).
    catalog:
        Pre-built catalog.  When given, *extension_root* / *chair_names* /
        *alpha_values* / *asset_cache* are ignored during state init.
    scale_distribution_params:
        ``(min, max)`` for uniform active-scale sampling.
    precision_scale_range:
        ``(min, max)`` for uniform precision-scale sampling.
    """
    motion_ref: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    state = _get_or_create_variant_state(
        env=env,
        extension_root=extension_root,
        chair_names=chair_names,
        alpha_values=alpha_values,
        asset_cache=asset_cache,
        catalog=catalog,
    )
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    num_reset = int(env_ids.numel())
    if num_reset == 0:
        return
    num_types = len(state.catalog.object_types)
    env_origins = getattr(env.scene, "env_origins", None)

    # ---- variant selection (per type, curriculum-aware) ----
    variant_ids = _select_variants_for_reset(env, state, num_reset, torch.device(env.device))

    # ---- scale sampling (curriculum-aware) ----
    scale_lo, scale_hi = float(scale_distribution_params[0]), float(scale_distribution_params[1])
    cr_scale = getattr(env, "_object_variant_scale_range", None)
    if cr_scale is not None:
        scale_lo, scale_hi = float(cr_scale[0]), float(cr_scale[1])

    prec_lo, prec_hi = float(precision_scale_range[0]), float(precision_scale_range[1])
    cr_prec = getattr(env, "_object_variant_precision_range", None)
    if cr_prec is not None:
        prec_lo, prec_hi = float(cr_prec[0]), float(cr_prec[1])
    if scale_hi > scale_lo:
        scales = torch.empty(num_reset, num_types, dtype=torch.float32, device=env.device).uniform_(
            scale_lo, scale_hi
        )
    else:
        scales = torch.full((num_reset, num_types), scale_lo, dtype=torch.float32, device=env.device)
    precision_scales = torch.empty(num_reset, num_types, dtype=torch.float32, device=env.device).uniform_(
        prec_lo, prec_hi
    )

    # ---- build entity → (type_idx, variant_idx_within_type) lookup ----
    entity_type_idx: list[int] = []
    entity_local_idx: list[int] = []
    for vi, entity_name in enumerate(object_entity_names):
        variant = state.catalog.variants[vi]
        try:
            ti = state.catalog.object_types.index(variant.object_type)
        except ValueError:
            raise RuntimeError(
                f"Entity '{entity_name}' type '{variant.object_type}' not in catalog object_types "
                f"{state.catalog.object_types}"
            )
        entity_type_idx.append(ti)
        entity_local_idx.append(variant.variant_index)

    # ---- per-object-type motion reference data ----
    init_state = motion_ref.get_init_reference_state(env_ids)
    object_slots: dict[str, int] = {}
    for obj_type in state.catalog.object_types:
        object_slots[obj_type] = _scene_object_slot(motion_ref, obj_type)

    # Pre-extract per-type object poses
    type_pos: dict[str, torch.Tensor] = {}
    type_quat: dict[str, torch.Tensor] = {}
    type_lin_vel: dict[str, torch.Tensor] = {}
    type_ang_vel: dict[str, torch.Tensor] = {}
    for obj_type in state.catalog.object_types:
        slot = object_slots[obj_type]
        type_pos[obj_type] = init_state.object_pos_w[:, slot].clone()
        type_quat[obj_type] = _safe_unit_quat(init_state.object_quat_w[:, slot])
        type_lin_vel[obj_type] = init_state.object_lin_vel_w[:, slot] * float(base_lin_vel_ratio)
        type_ang_vel[obj_type] = init_state.object_ang_vel_w[:, slot] * float(base_ang_vel_ratio)

    reference_offsets = torch.zeros(num_reset, 3, device=env.device)
    if apply_env_spacing and env_origins is not None:
        offsets = env_origins[env_ids].to(device=env.device)
        reference_offsets = offsets
        for obj_type in state.catalog.object_types:
            type_pos[obj_type] = type_pos[obj_type] + offsets

    # ---- write entity poses ----
    zero_velocity = torch.zeros(num_reset, 6, dtype=torch.float32, device=env.device)
    for vi, entity_name in enumerate(object_entity_names):
        asset = env.scene[entity_name]
        ti = entity_type_idx[vi]
        obj_type = state.catalog.object_types[ti]

        # active when the selected variant for this env+type matches this entity's catalog index
        active_mask = variant_ids[:, ti] == vi
        if active_mask.any():
            root_pose = torch.cat(
                [type_pos[obj_type][active_mask], type_quat[obj_type][active_mask]], dim=-1
            )
            root_velocity = torch.cat(
                [type_lin_vel[obj_type][active_mask], type_ang_vel[obj_type][active_mask]], dim=-1
            )
            _write_pose_velocity(
                asset, root_pose, root_velocity, env_ids[active_mask]
            )
        inactive_mask = ~active_mask
        if inactive_mask.any():
            inactive_pose = _inactive_pose(
                env_ids[inactive_mask],
                vi,
                env_origins,
                torch.float32,
                env.device,
            )
            _write_pose_velocity(
                asset, inactive_pose, zero_velocity[inactive_mask], env_ids[inactive_mask]
            )

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
    """Every-step update: track reference trajectory with active variant, hide inactive.

    Reads the live motion-reference buffer and writes poses to the active
    variant entities.  Inactive variants are staggered out of sight.
    """
    del env_ids
    motion_ref: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    data = motion_ref.data
    state = getattr(env, "_part2link_object_variant_state", None)
    if state is None:
        return

    all_env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    # Build entity → type_idx lookup (same as reset)
    entity_type_idx: list[int] = []
    for vi, entity_name in enumerate(object_entity_names):
        variant = state.catalog.variants[vi]
        try:
            ti = state.catalog.object_types.index(variant.object_type)
        except ValueError:
            raise RuntimeError(
                f"Entity '{entity_name}' type not in catalog object_types"
            )
        entity_type_idx.append(ti)

    # Per-type object slots
    object_slots: dict[str, int] = {}
    for obj_type in state.catalog.object_types:
        object_slots[obj_type] = _scene_object_slot(motion_ref, obj_type)

    zero_velocity = torch.zeros(env.num_envs, 6, dtype=torch.float32, device=env.device)
    invalid_pose_template = torch.zeros(env.num_envs, 7, dtype=torch.float32, device=env.device)
    invalid_pose_template[:, :3] = torch.tensor(
        invalid_object_pos, dtype=torch.float32, device=env.device
    )
    invalid_pose_template[:, 3] = 1.0

    for vi, entity_name in enumerate(object_entity_names):
        asset = env.scene[entity_name]
        ti = entity_type_idx[vi]
        obj_type = state.catalog.object_types[ti]
        slot = object_slots[obj_type]

        obj_pos = data.object_pos_w[:, 0, slot].clone()
        obj_pos = obj_pos + state.reference_position_offsets.to(
            device=obj_pos.device, dtype=obj_pos.dtype
        )
        obj_quat = _safe_unit_quat(data.object_quat_w[:, 0, slot])
        obj_lin_vel = data.object_lin_vel_w[:, 0, slot]
        obj_ang_vel = data.object_ang_vel_w[:, 0, slot]
        obj_validity = data.object_validity[:, 0, slot].to(torch.bool)

        root_pose = torch.cat([obj_pos, obj_quat], dim=-1)
        root_velocity = torch.cat([obj_lin_vel, obj_ang_vel], dim=-1)

        active_mask = (state.active_variant_ids[:, ti] == vi) & obj_validity
        if active_mask.any():
            _write_pose_velocity(
                asset, root_pose[active_mask], root_velocity[active_mask], all_env_ids[active_mask]
            )
        inactive_mask = ~active_mask
        if inactive_mask.any():
            hidden_pose = invalid_pose_template[inactive_mask].clone()
            hidden_pose[:, 0] += float(vi % 32) * _INACTIVE_OBJECT_SPACING
            hidden_pose[:, 1] += float(vi // 32) * _INACTIVE_OBJECT_SPACING
            _write_pose_velocity(
                asset, hidden_pose, zero_velocity[inactive_mask], all_env_ids[inactive_mask]
            )
    motion_ref.sync_reference_entity_state()


def _squeeze_type_dim(tensor: torch.Tensor | None, type_idx: int = 0) -> torch.Tensor | None:
    """Squeeze the object-type dimension for backward-compatible access.

    Shared ``ObjectVariantRuntimeState`` stores contact data as
    ``[E, T, …]``.  Code written for the legacy ``[E, …]`` layout
    can call this helper to extract a single type's view.
    """
    if tensor is None:
        return None
    return tensor[:, type_idx]


def _active_object_state_w(
    env: ManagerBasedEnv, object_entity_names: Sequence[str]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return world-frame (pos, quat, lin_vel, ang_vel) of the active variant.

    When the catalog contains a single object type the type dimension is
    squeezed so the return shapes match legacy Part2Link expectations.
    """
    state = getattr(env, "_part2link_object_variant_state", None)
    if state is None:
        first_asset = env.scene[object_entity_names[0]]
        return (
            _root_pos(first_asset),
            _root_quat(first_asset),
            _root_lin_vel(first_asset),
            _root_ang_vel(first_asset),
        )

    pos_list, quat_list, lin_vel_list, ang_vel_list = [], [], [], []
    for entity_name in object_entity_names:
        asset = env.scene[entity_name]
        pos_list.append(_root_pos(asset))
        quat_list.append(_root_quat(asset))
        lin_vel_list.append(_root_lin_vel(asset))
        ang_vel_list.append(_root_ang_vel(asset))
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    # active_variant_ids: [E, T]; for single-type catalogs squeeze to [E]
    variant_ids = state.active_variant_ids
    if variant_ids.shape[1] == 1:
        variant_ids = variant_ids[:, 0]
    pos = torch.stack(pos_list, dim=1)[env_ids, variant_ids]
    quat = torch.stack(quat_list, dim=1)[env_ids, variant_ids]
    lin_vel = torch.stack(lin_vel_list, dim=1)[env_ids, variant_ids]
    ang_vel = torch.stack(ang_vel_list, dim=1)[env_ids, variant_ids]
    return pos, quat, lin_vel, ang_vel
