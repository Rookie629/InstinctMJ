"""Prepare cache meshes for G1 Sitting Part2Link object variants."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from instinct_mj.tasks.interaction.mdp.part2link import _alpha_to_glb_name, _alpha_to_token

DEFAULT_DATASET_ROOT = "/home/yangke/KY/InstinctLab_interact/datasets/interaction/output_npz_29dof_with_object"
DEFAULT_CHAIRS = (
    "chair_14",
    "chair_15",
    "chair_17",
    "chair_18",
    "chair_20",
    "chair_22",
    "chair_28",
    "chair_30",
    "chair_32",
    "chair_33",
    "chair_37",
    "chair_39",
    "chair_40",
    "chair_41",
    "chair_43",
    "chair_44",
    "chair_46",
    "chair_48",
    "chair_51",
    "chair_55",
)


def _parse_csv(raw: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not raw:
        return default
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    return values or default


def _parse_alphas(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("--alphas must contain at least one value")
    return values


def _export_glb_to_obj(src: Path, dst: Path) -> None:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("prepare_part2link_assets requires `trimesh` to convert GLB assets to OBJ cache files.") from exc

    loaded = trimesh.load(src, force="scene")
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(geom for geom in loaded.geometry.values()))
    else:
        mesh = loaded
    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Part2Link GLB variants into MuJoCo-friendly OBJ cache files.")
    parser.add_argument(
        "--dataset-root",
        default=os.getenv("INSTINCT_PART2LINK_DATASET_ROOT", DEFAULT_DATASET_ROOT),
    )
    parser.add_argument(
        "--asset-cache",
        default=os.getenv("INSTINCT_PART2LINK_ASSET_CACHE", "~/.cache/instinct_mj/part2link_assets"),
    )
    parser.add_argument("--chairs", default=os.getenv("SITTING_PART2LINK_CHAIR_NAMES", ",".join(DEFAULT_CHAIRS)))
    parser.add_argument("--alphas", default=os.getenv("SITTING_PART2LINK_ALPHA_VALUES", "1.0"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    extension_root = dataset_root / "sofa_exntend_obj"
    asset_cache = Path(args.asset_cache).expanduser().resolve()
    chair_names = _parse_csv(args.chairs, DEFAULT_CHAIRS)
    alphas = _parse_alphas(args.alphas)

    for chair_name in chair_names:
        morph_dir = extension_root / chair_name / "ffd_bbox_coarse" / "morph_path"
        for alpha in alphas:
            src = morph_dir / _alpha_to_glb_name(alpha)
            if not src.exists():
                raise FileNotFoundError(f"Missing source GLB: {src}")
            dst = asset_cache / chair_name / f"{_alpha_to_token(alpha)}.obj"
            print(f"{src} -> {dst}")
            if not args.dry_run:
                _export_glb_to_obj(src, dst)
    print(f"[done] Set INSTINCT_PART2LINK_ASSET_CACHE={asset_cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

