"""Prepare CoACD convex collision meshes for G1 Sitting Part2Link object variants."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "datasets", "interaction")
DEFAULT_DATASET_ROOT = os.path.join(_DATA_DIR, "output_npz_29dof_with_object")
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
DEFAULT_ALPHAS = "1.0,0.8,0.5,0.0"


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


def _alpha_to_token(alpha: float) -> str:
    return f"alpha_{float(alpha):.2f}".replace(".", "p")


def _alpha_to_glb_name(alpha: float) -> str:
    return f"alpha_{float(alpha):.2f}.glb"


def _sanitize_variant_name(chair_name: str, alpha: float) -> str:
    return f"{chair_name}_{_alpha_to_token(alpha)}".replace("-", "_").replace(".", "p")


def _load_mesh(mesh_path: Path):
    import trimesh

    loaded = trimesh.load(mesh_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        if hasattr(loaded, "to_geometry"):
            mesh = loaded.to_geometry()
        else:
            mesh = trimesh.util.concatenate(tuple(geom for geom in loaded.geometry.values()))
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected {mesh_path} to load as trimesh.Trimesh, got {type(mesh)!r}.")
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def _run_coacd(mesh, args: argparse.Namespace) -> list[tuple[np.ndarray, np.ndarray]]:
    import coacd

    coacd.set_log_level(args.log_level)
    coacd_mesh = coacd.Mesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        indices=np.asarray(mesh.faces, dtype=np.int32),
    )
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=args.threshold,
        max_convex_hull=args.max_convex_hull,
        preprocess_mode=args.preprocess_mode,
        preprocess_resolution=args.preprocess_resolution,
        resolution=args.resolution,
        mcts_nodes=args.mcts_nodes,
        mcts_iterations=args.mcts_iterations,
        mcts_max_depth=args.mcts_max_depth,
        pca=args.pca,
        merge=args.merge,
        decimate=args.decimate,
        max_ch_vertex=args.max_ch_vertex,
        extrude=args.extrude,
        extrude_margin=args.extrude_margin,
        seed=args.seed,
        apx_mode=args.apx_mode,
    )
    return [(np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)) for vertices, faces in parts]


def _export_collision_parts(parts: list[tuple[np.ndarray, np.ndarray]], output_dir: Path) -> list[str]:
    import trimesh

    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []
    for index, (vertices, faces) in enumerate(parts):
        if vertices.size == 0 or faces.size == 0:
            continue
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        filename = f"collision_{index:03d}.obj"
        mesh.export(output_dir / filename)
        exported.append(filename)
    return exported


def _part_volume(part: tuple[np.ndarray, np.ndarray]) -> float:
    import trimesh

    vertices, faces = part
    if vertices.size == 0 or faces.size == 0:
        return 0.0
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return float(abs(mesh.volume))


def _limit_parts(
    parts: list[tuple[np.ndarray, np.ndarray]],
    max_output_hulls: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if max_output_hulls <= 0 or len(parts) <= max_output_hulls:
        return parts
    ranked_indices = sorted(range(len(parts)), key=lambda index: _part_volume(parts[index]), reverse=True)
    kept_indices = sorted(ranked_indices[:max_output_hulls])
    return [parts[index] for index in kept_indices]


def _write_viewer_xml(
    output_dir: Path,
    visual_mesh: str,
    collision_meshes: list[str],
    body_pos: tuple[float, float, float],
) -> None:
    mesh_assets = [f'    <mesh name="visual_mesh" file="{visual_mesh}" scale="1 1 1"/>']
    for index, filename in enumerate(collision_meshes):
        mesh_assets.append(f'    <mesh name="collision_mesh_{index:03d}" file="{filename}" scale="1 1 1"/>')

    geom_lines = [
        (
            '      <geom name="object_visual" type="mesh" mesh="visual_mesh" '
            'group="2" rgba="0.2 0.55 0.85 0.45" contype="0" conaffinity="0" density="0"/>'
        )
    ]
    mass_per_part = 1.0 / max(len(collision_meshes), 1)
    for index, _ in enumerate(collision_meshes):
        geom_lines.append(
            (
                f'      <geom name="object_collision_{index:03d}" type="mesh" mesh="collision_mesh_{index:03d}" '
                f'group="3" rgba="1.0 0.55 0.05 0.35" mass="{mass_per_part:.8f}" '
                'friction="1.0 0.005 0.0001" contype="1" conaffinity="1"/>'
            )
        )

    xml = "\n".join(
        [
            '<mujoco model="part2link_coacd_collision_viewer">',
            '  <compiler angle="radian"/>',
            '  <option timestep="0.002" gravity="0 0 -9.81"/>',
            "  <asset>",
            '    <texture name="checker" type="2d" builtin="checker" width="512" height="512" rgb1="0.22 0.22 0.22" rgb2="0.34 0.34 0.34"/>',
            '    <material name="ground_mat" texture="checker" texrepeat="4 4" reflectance="0.1"/>',
            *mesh_assets,
            "  </asset>",
            "  <worldbody>",
            '    <light name="key" pos="2 -3 4" dir="-2 3 -4" diffuse="0.8 0.8 0.8"/>',
            '    <camera name="front" pos="2.5 -3.0 1.8" xyaxes="0.768 0.640 0 -0.286 0.343 0.895"/>',
            '    <geom name="ground" type="plane" size="2.5 2.5 0.02" material="ground_mat" contype="1" conaffinity="1"/>',
            f'    <body name="object" pos="{body_pos[0]:.8f} {body_pos[1]:.8f} {body_pos[2]:.8f}">',
            '      <freejoint name="object_free"/>',
            *geom_lines,
            "    </body>",
            "  </worldbody>",
            "</mujoco>",
            "",
        ]
    )
    (output_dir / "viewer.xml").write_text(xml, encoding="utf-8")


def _render_progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = int(round(width * min(max(done / total, 0.0), 1.0)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _print_progress(done: int, total: int, current: str, hulls: int | None = None) -> None:
    percent = 100.0 * float(done) / float(total) if total > 0 else 100.0
    suffix = f" current={current}"
    if hulls is not None:
        suffix += f" hulls={hulls}"
    sys.stdout.write(
        f"\r[Part2Link CoACD] {_render_progress_bar(done, total)} "
        f"{done}/{total} ({percent:5.1f}%){suffix}"
    )
    sys.stdout.flush()


def _process_variant(src: Path, output_dir: Path, args: argparse.Namespace) -> int:
    mesh = _load_mesh(src)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("collision_*.obj"):
        stale_path.unlink()
    for stale_name in ("visual.obj", "viewer.xml", "manifest.json"):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    visual_filename = "visual.obj"
    mesh.export(output_dir / visual_filename)
    parts = _run_coacd(mesh, args)
    raw_num_collision_meshes = len(parts)
    parts = _limit_parts(parts, args.max_output_hulls)
    collision_meshes = _export_collision_parts(parts, output_dir)
    if not collision_meshes:
        raise RuntimeError(f"CoACD produced no collision meshes for {src}.")
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center_xy = bounds[:, :2].mean(axis=0)
    viewer_body_pos = (-float(center_xy[0]), -float(center_xy[1]), -float(bounds[0, 2]))
    _write_viewer_xml(output_dir, visual_filename, collision_meshes, viewer_body_pos)

    manifest = {
        "source": str(src),
        "visual_mesh": visual_filename,
        "collision_meshes": collision_meshes,
        "num_collision_meshes": len(collision_meshes),
        "raw_num_collision_meshes": raw_num_collision_meshes,
        "params": {
            "threshold": args.threshold,
            "max_convex_hull": args.max_convex_hull,
            "preprocess_mode": args.preprocess_mode,
            "preprocess_resolution": args.preprocess_resolution,
            "resolution": args.resolution,
            "mcts_nodes": args.mcts_nodes,
            "mcts_iterations": args.mcts_iterations,
            "mcts_max_depth": args.mcts_max_depth,
            "seed": args.seed,
            "merge": args.merge,
            "decimate": args.decimate,
            "max_ch_vertex": args.max_ch_vertex,
            "apx_mode": args.apx_mode,
            "max_output_hulls": args.max_output_hulls,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return len(collision_meshes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create CoACD convex collision mesh cache for Part2Link object variants.")
    parser.add_argument("--dataset-root", default=os.getenv("INSTINCT_PART2LINK_DATASET_ROOT", DEFAULT_DATASET_ROOT))
    parser.add_argument(
        "--collision-cache",
        default=os.getenv("INSTINCT_PART2LINK_COLLISION_CACHE", "~/.cache/instinct_mj/part2link_coacd_collisions"),
    )
    parser.add_argument("--chairs", default=os.getenv("SITTING_PART2LINK_CHAIR_NAMES", ",".join(DEFAULT_CHAIRS)))
    parser.add_argument("--alphas", default=os.getenv("SITTING_PART2LINK_ALPHA_VALUES", DEFAULT_ALPHAS))
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max-convex-hull", type=int, default=32)
    parser.add_argument("--max-output-hulls", type=int, default=64)
    parser.add_argument("--preprocess-mode", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--preprocess-resolution", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=2000)
    parser.add_argument("--mcts-nodes", type=int, default=20)
    parser.add_argument("--mcts-iterations", type=int, default=150)
    parser.add_argument("--mcts-max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="off")
    parser.add_argument("--apx-mode", choices=("ch", "box"), default="ch")
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--decimate", action="store_true")
    parser.add_argument("--max-ch-vertex", type=int, default=256)
    parser.add_argument("--extrude", action="store_true")
    parser.add_argument("--extrude-margin", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    extension_root = dataset_root / "sofa_exntend_obj"
    collision_cache = Path(args.collision_cache).expanduser().resolve()
    chair_names = _parse_csv(args.chairs, DEFAULT_CHAIRS)
    alphas = _parse_alphas(args.alphas)

    jobs: list[tuple[str, Path, Path]] = []
    for chair_name in chair_names:
        morph_dir = extension_root / chair_name / "ffd_bbox_coarse" / "morph_path"
        for alpha in alphas:
            src = morph_dir / _alpha_to_glb_name(alpha)
            if not src.exists():
                raise FileNotFoundError(f"Missing source GLB: {src}")
            variant_name = _sanitize_variant_name(chair_name, alpha)
            output_dir = collision_cache / variant_name
            jobs.append((variant_name, src, output_dir))

    total_jobs = len(jobs)
    _print_progress(0, total_jobs, "starting")
    for index, (variant_name, src, output_dir) in enumerate(jobs):
        _print_progress(index, total_jobs, variant_name)
        hull_count = None
        if not args.dry_run:
            hull_count = _process_variant(src, output_dir, args)
        _print_progress(index + 1, total_jobs, variant_name, hull_count)
    sys.stdout.write("\n")

    print(f"[done] Set INSTINCT_PART2LINK_COLLISION_CACHE={collision_cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
