#!/usr/bin/env python3
"""Prepare variant meshes for HOI object domain randomization.

Converts source meshes (GLB/USD) to MuJoCo-compatible OBJ files in an asset
cache directory.  Optionally validates contact-point files.

Usage::

    python -m instinct_mj.scripts.prepare_hoi_variants \\
        --variant-root ~/Datasets/OMOMO/object_variants \\
        --asset-cache ~/.cache/instinct_mj/hoi_assets \\
        --object-types floorlamp,largebox,whitechair

Environment variables (fallbacks for CLI flags)::

    INSTINCT_HOI_VARIANT_ROOT       Source directory of variant meshes.
    INSTINCT_HOI_VARIANT_ASSET_CACHE Output OBJ cache directory.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Default object types (mirrors OMOMO_OBJECT_TYPES in object_variant_hoi.py)
DEFAULT_OBJECT_TYPES = (
    "floorlamp",
    "largebox",
    "whitechair",
    "trashcan",
    "smalltable",
    "suitcase",
)


def _export_glb_to_obj(glb_path: Path, obj_path: Path) -> None:
    """Convert a GLB mesh to Wavefront OBJ using trimesh."""
    try:
        import trimesh
    except ImportError:
        print("trimesh is required for GLB→OBJ conversion.  Install with: pip install trimesh")
        sys.exit(1)

    scene = trimesh.load(str(glb_path), force="mesh")
    if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in scene.geometry.values() if hasattr(g, "vertices")]
        )
    else:
        mesh = scene
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(obj_path))
    print(f"  Converted: {glb_path.name} → {obj_path}")


def _copy_obj(src: Path, dst: Path) -> None:
    """Copy an existing OBJ file to the cache."""
    import shutil

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    print(f"  Copied: {src.name} → {dst}")


def _validate_contact_points(npz_path: Path) -> bool:
    """Check that *npz_path* contains the required keys."""
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:
        print(f"  WARNING: Cannot read {npz_path}: {exc}")
        return False
    required = {"alpha_values", "points_local", "center_local"}
    missing = required - set(data.keys())
    if missing:
        print(f"  WARNING: {npz_path} missing keys: {missing}")
        return False
    print(f"  OK: {npz_path} — {data['alpha_values'].shape[0]} alpha(s), "
          f"points={data['points_local'].shape}, centers={data['center_local'].shape}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare HOI object variant meshes for MuJoCo",
    )
    parser.add_argument(
        "--variant-root",
        default=os.getenv("INSTINCT_HOI_VARIANT_ROOT"),
        help="Source directory with per-type variant meshes (env: INSTINCT_HOI_VARIANT_ROOT)",
    )
    parser.add_argument(
        "--asset-cache",
        default=os.getenv(
            "INSTINCT_HOI_VARIANT_ASSET_CACHE",
            os.path.expanduser("~/.cache/instinct_mj/hoi_assets"),
        ),
        help="Output OBJ cache directory (env: INSTINCT_HOI_VARIANT_ASSET_CACHE)",
    )
    parser.add_argument(
        "--object-types",
        default=",".join(DEFAULT_OBJECT_TYPES),
        help="Comma-separated object types to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without converting",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate contact-point files, skip mesh conversion",
    )
    args = parser.parse_args()

    if not args.variant_root:
        parser.error(
            "--variant-root is required (or set INSTINCT_HOI_VARIANT_ROOT)."
        )

    variant_root = Path(args.variant_root).expanduser().resolve()
    asset_cache = Path(args.asset_cache).expanduser().resolve()
    object_types = [
        t.strip() for t in args.object_types.split(",") if t.strip()
    ]

    if not variant_root.is_dir():
        print(f"ERROR: variant-root does not exist: {variant_root}")
        sys.exit(1)

    converted = 0
    validated = 0

    for obj_type in object_types:
        type_dir = variant_root / obj_type
        if not type_dir.is_dir():
            print(f"SKIP: {obj_type} — directory not found at {type_dir}")
            continue

        print(f"\nProcessing: {obj_type}")

        # --- Contact-point validation ---
        contact_npz = type_dir / "contact_points_local.npz"
        if contact_npz.exists():
            if _validate_contact_points(contact_npz):
                validated += 1

        if args.validate_only:
            continue

        # --- Mesh conversion ---
        mesh_files = sorted(
            list(type_dir.glob("*.glb"))
            + list(type_dir.glob("*.usd"))
            + list(type_dir.glob("*.obj"))
        )
        if not mesh_files:
            print(f"  WARNING: No mesh files found for {obj_type}")
            continue

        for idx, mesh_file in enumerate(mesh_files):
            cache_dest = asset_cache / obj_type / f"variant_{idx}.obj"

            if args.dry_run:
                print(f"  [DRY-RUN] {mesh_file.name} → {cache_dest}")
                converted += 1
                continue

            suffix = mesh_file.suffix.lower()
            if suffix in (".glb", ".usd"):
                _export_glb_to_obj(mesh_file, cache_dest)
            elif suffix == ".obj":
                _copy_obj(mesh_file, cache_dest)
            else:
                print(f"  SKIP: unsupported format {suffix}")
                continue
            converted += 1

    print(f"\nDone. Converted: {converted}, Validated contact NPs: {validated}")


if __name__ == "__main__":
    main()
