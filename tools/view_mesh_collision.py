#!/usr/bin/env python3
"""View a GLB/USD/OBJ mesh in MuJoCo viewer with collision geometry.

Usage:
    python view_mesh_collision.py <mesh_path>  [--scale S]

This creates a minimal MJCF scene with the mesh as a mjGEOM_MESH geom,
opens the MuJoCo interactive viewer, and toggles collision-visualisation.

In the viewer:
    1. Double-click the geom in the left panel to select it.
    2. Press **F6** to toggle convex-hull collision rendering (a translucent
       overlay of the actual collision surface).
    3. Press **1** / **2** to switch between rendering modes.
    4. Scroll to zoom, right-drag to rotate.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# ------------------------------------------------
# Workaround: trimesh → OBJ → MuJoCo mesh
# ------------------------------------------------
def _guess_scale(mesh_path: str) -> tuple[float, float, float]:
    """Return a default scale so the mesh is visible at human scale (~1-2 m)."""
    return (1.0, 1.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="View mesh collision in MuJoCo")
    parser.add_argument("mesh_path", help="Path to .glb / .usd / .obj file")
    parser.add_argument("--scale", type=float, nargs=3, default=(1.0, 1.0, 1.0),
                        help="Mesh scale (x y z), default 1 1 1")
    parser.add_argument("--obj", action="store_true",
                        help="Force convert to OBJ before loading (trimesh required)")
    args = parser.parse_args()

    mesh_path = os.path.expanduser(args.mesh_path)
    if not os.path.exists(mesh_path):
        print(f"ERROR: {mesh_path} not found")
        sys.exit(1)

    # If it's a GLB, convert to OBJ first (MuJoCo doesn't read GLB directly)
    load_path = mesh_path
    if mesh_path.endswith('.glb'):
        try:
            import trimesh
        except ImportError:
            print("ERROR: trimesh required for GLB conversion.  pip install trimesh")
            sys.exit(1)

        scene = trimesh.load(mesh_path, force='mesh')
        if isinstance(scene, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [g for g in scene.geometry.values() if hasattr(g, 'vertices')]
            )
        else:
            mesh = scene

        tmp_dir = tempfile.mkdtemp()
        load_path = os.path.join(tmp_dir, 'mesh.obj')
        mesh.export(load_path)
        print(f"Converted GLB → OBJ: {load_path}")
        print(f"  vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}")

        # Print convex hull info
        try:
            hull = mesh.convex_hull
            print(f"  convex hull vertices: {len(hull.vertices)}, faces: {len(hull.faces)}")
            hull_vol = hull.volume if hasattr(hull, 'volume') else 'N/A'
            mesh_vol = mesh.volume if hasattr(mesh, 'volume') else 'N/A'
            print(f"  mesh volume: {mesh_vol}, hull volume: {hull_vol}")
        except Exception:
            print("  (could not compute convex hull)")

    import mujoco
    import mujoco_viewer  # noqa: F401 — registers the viewer backend

    spec = mujoco.MjSpec()
    mesh = spec.add_mesh(
        name="mesh",
        file=load_path,
        scale=[float(args.scale[0]), float(args.scale[1]), float(args.scale[2])],
    )

    # Add a ground plane
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(5, 5, 0.1),
        rgba=(0.3, 0.3, 0.3, 1),
    )

    body = spec.worldbody.add_body(name="object")
    body.add_geom(
        name="object_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="mesh",
        mass=1.0,
        rgba=(0.0, 0.8, 0.3, 1.0),
    )

    # Add a free joint so the object can be moved around in the viewer
    body.add_freejoint()

    model = spec.compile()
    data = mujoco.MjData(model)

    # Use passive viewer via the 'mujoco' Python bindings if available,
    # otherwise fall back to the GLFW-based viewer.
    print("\n=== Viewer Controls ===")
    print("  Mouse: left-drag=rotate, right-drag=translate, scroll=zoom")
    print("  F6   : toggle convex-hull collision overlay (semi-transparent)")
    print("    → This is THE key: watch how the chair concavity gets filled!")
    print("  1/2  : switch rendering modes")
    print("  F1   : help overlay")
    print("========================\n")

    try:
        from mujoco import viewer
        viewer.launch(model, data)
    except Exception:
        # Fallback: launch via glfw
        with mujoco.viewer.launch_passive(model, data) as v:
            while v.is_running():
                mujoco.mj_step(model, data)
                v.sync()


if __name__ == "__main__":
    main()
