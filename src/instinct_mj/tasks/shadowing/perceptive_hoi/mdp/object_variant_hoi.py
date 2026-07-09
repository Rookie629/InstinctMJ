"""HOI-specific object variant catalog construction for OMOMO objects.

Bridges the generic :mod:`instinct_mj.envs.mdp.events.object_variant` module
to the six OMOMO object types used in the perceptive HOI task.

Two operating modes
-------------------
1. **Single-mesh** (default, backward-compatible):
   Each OMOMO object type has exactly one variant entity with the original
   hardcoded mesh.  The variant catalog degenerates to one variant per type.
   Object DR events (friction / COM) are still applied.

2. **Multi-variant** (``INSTINCT_HOI_VARIANT_ROOT``):
   Each OMOMO object type can have *N* variant meshes discovered from a
   directory.  Variant-switching and curriculum training are enabled.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from instinct_mj.envs.mdp.events.object_variant import (
    ObjectVariant,
    ObjectVariantCatalog,
)

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Hard-coded OMOMO object registry (migrated from perceptive_shadowing_cfg.py)
# ---------------------------------------------------------------------------

OMOMO_OBJECT_TYPES: tuple[str, ...] = (
    "floorlamp",
    "largebox",
    "whitechair",
    "trashcan",
    "smalltable",
    "suitcase",
)

OMOMO_MESH_FILE_PATHS: dict[str, str] = {
    "floorlamp": "~/Datasets/OMOMO/data/captured_objects/floorlamp_cleaned_simplified.obj",
    "largebox": "~/Datasets/OMOMO/data/captured_objects/largebox_cleaned_simplified.obj",
    "whitechair": "~/Datasets/OMOMO/data/captured_objects/whitechair_cleaned_simplified.obj",
    "trashcan": "~/Datasets/OMOMO/data/captured_objects/trashcan_cleaned_simplified.obj",
    "smalltable": "~/Datasets/OMOMO/data/captured_objects/smalltable_cleaned_simplified.obj",
    "suitcase": "~/Datasets/OMOMO/data/captured_objects/suitcase_cleaned_simplified.obj",
}

OMOMO_MESH_SCALES: dict[str, tuple[float, float, float]] = {
    "floorlamp": (1.55 * 0.3793, 1.55 * 0.3793, 1.55 * 0.3793),
    "largebox": (1.55 * 0.3486, 1.55 * 0.3486, 1.55 * 0.3486),
    "whitechair": (1.55 * 0.3129, 1.55 * 0.3129, 1.55 * 0.3129),
    "trashcan": (1.55 * 0.2326, 1.55 * 0.2326, 1.55 * 0.2326),
    "smalltable": (1.55 * 0.0162, 1.55 * 0.0162, 1.55 * 0.0162),
    "suitcase": (1.55 * 0.3672, 1.55 * 0.3672, 1.55 * 0.3672),
}


# ---------------------------------------------------------------------------
# Mesh-object MjSpec factory (reusable)
# ---------------------------------------------------------------------------


def _make_mesh_object_spec_fn(
    mesh_file_path: str, scale: tuple[float, float, float]
):
    """Return a zero-argument callable that builds a mocap-body ``MjSpec``."""

    def spec_fn() -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        mesh = spec.add_mesh(
            name="object_mesh",
            file=os.path.expanduser(mesh_file_path),
            scale=scale,
        )
        body = spec.worldbody.add_body(name="object", mocap=True)
        body.add_geom(
            name="object_geom",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh.name,
            mass=1.0,
            rgba=(0.0, 0.8, 0.3, 1.0),
            friction=(1.0, 0.005, 0.0001),
        )
        return spec

    return spec_fn


# ---------------------------------------------------------------------------
# Catalog builders
# ---------------------------------------------------------------------------


def build_hoi_single_mesh_catalog(
    device: str | torch.device = "cpu",
) -> ObjectVariantCatalog:
    """Build a degenerate catalog: **one variant per OMOMO object type**.

    This preserves exact backward compatibility with the current hardcoded
    entity layout.  The variant entity names are identical to the object
    type names (e.g. ``"floorlamp"``).
    """
    import torch as _torch

    variants: list[ObjectVariant] = []
    for obj_type in OMOMO_OBJECT_TYPES:
        mesh_path = Path(os.path.expanduser(OMOMO_MESH_FILE_PATHS[obj_type]))
        variants.append(
            ObjectVariant(
                name=obj_type,  # entity name == object type
                chair_name=obj_type,
                alpha=1.0,
                mesh_path=mesh_path,
                contact_points_path=mesh_path,  # not used; placeholder
                alpha_index=0,
                variant_index=0,
                metadata={},
            )
        )

    # Degenerate catalog: no contact-point data
    return ObjectVariantCatalog(
        variants=variants,
        variant_names=[v.name for v in variants],
        chair_names=[v.chair_name for v in variants],
        part_names=[],
        centers_local=None,
        points_local=None,
        point_valid_mask=None,
    )


def build_hoi_multi_variant_catalog(
    extension_root: str | Path,
    object_types: Sequence[str] = OMOMO_OBJECT_TYPES,
    asset_cache: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> ObjectVariantCatalog:
    """Build a multi-variant catalog from a directory of OMOMO object variants.

    Expected directory structure::

        extension_root/
            floorlamp/
                variant_0.obj   (or .glb / .usd)
                variant_1.obj
                contact_points_local.npz  (optional)
            largebox/
                ...

    Each mesh file becomes a separate variant entity in the scene.
    """
    import torch as _torch

    root = Path(extension_root).expanduser().resolve()
    obj_types = list(object_types)

    variants: list[ObjectVariant] = []
    for obj_type in obj_types:
        type_dir = root / obj_type
        if not type_dir.is_dir():
            continue
        mesh_files = sorted(
            list(type_dir.glob("*.obj"))
            + list(type_dir.glob("*.glb"))
            + list(type_dir.glob("*.usd"))
        )
        if not mesh_files:
            # Fall back to the hardcoded single mesh
            mesh_path = Path(os.path.expanduser(OMOMO_MESH_FILE_PATHS.get(obj_type, "")))
            if mesh_path.exists():
                mesh_files = [mesh_path]

        for vi, mesh_file in enumerate(mesh_files):
            if asset_cache is not None:
                cache_dir = Path(asset_cache).expanduser().resolve() / obj_type
                cached = cache_dir / f"variant_{vi}.obj"
                if cached.exists():
                    mesh_file = cached

            entity_name = f"{obj_type}_variant_{vi}" if len(mesh_files) > 1 else obj_type
            variants.append(
                ObjectVariant(
                    name=entity_name,
                    chair_name=obj_type,
                    alpha=1.0,
                    mesh_path=mesh_file,
                    contact_points_path=mesh_file,  # placeholder
                    alpha_index=0,
                    variant_index=vi,
                    metadata={"source": str(mesh_file)},
                )
            )

    if not variants:
        raise FileNotFoundError(
            f"No HOI object variants found under extension_root={root}"
        )

    return ObjectVariantCatalog(
        variants=variants,
        variant_names=[v.name for v in variants],
        chair_names=[v.chair_name for v in variants],
        part_names=[],
        centers_local=None,
        points_local=None,
        point_valid_mask=None,
    )


# ---------------------------------------------------------------------------
# Entity factories
# ---------------------------------------------------------------------------


def make_hoi_variant_entities(
    catalog: ObjectVariantCatalog,
    *,
    include_reference: bool = False,
    robot_cfg=None,
) -> dict[str, EntityCfg]:
    """Build scene entity dict from a catalog.

    Replaces the current hardcoded ``_make_hoi_entities()`` loop.
    """

    entities: dict[str, EntityCfg] = {
        "robot": deepcopy(robot_cfg) if robot_cfg is not None else deepcopy(_default_robot_cfg()),
    }

    if include_reference:
        robot_reference = deepcopy(
            robot_cfg if robot_cfg is not None else _default_robot_cfg()
        )
        robot_reference.collisions = (
            CollisionCfg(
                geom_names_expr=(".*",),
                contype=0,
                conaffinity=0,
            ),
        )
        entities["robot_reference"] = robot_reference

    for variant in catalog.variants:
        obj_type = variant.object_type
        scale = OMOMO_MESH_SCALES.get(
            obj_type,
            OMOMO_MESH_SCALES.get("whitechair", (1.0, 1.0, 1.0)),
        )
        entities[variant.name] = EntityCfg(
            spec_fn=_make_mesh_object_spec_fn(str(variant.mesh_path), scale),
        )

    return entities


def make_hoi_variant_camera_mesh_prim_paths(
    catalog: ObjectVariantCatalog,
    link_names: list[str] | None = None,
) -> list[str]:
    """Build camera ``mesh_prim_paths`` covering all variant entities.

    In single-mesh mode the output is identical to the current
    ``_make_hoi_camera_mesh_prim_paths()``.
    """
    paths = ["/World/ground"]
    if link_names:
        for link_name in link_names:
            paths.append(f"/World/envs/env_.*/Robot/{link_name}")
    for variant_name in catalog.variant_names:
        paths.append(f"/World/envs/env_.*/{variant_name}")
    return paths


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_robot_cfg():
    """Lazy-import G1 config to avoid circular imports."""
    from instinct_mj.assets.unitree_g1 import G1_29DOF_TORSOBASE_POPSICLE_CFG

    return G1_29DOF_TORSOBASE_POPSICLE_CFG
