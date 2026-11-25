#!/usr/bin/env python3
"""Turn a thin, holey splat mesh into a solid, watertight surface.

Voxelise the surface (ray parity fills the interior), mesh the occupancy grid
with marching cubes, relax it, and carry the source vertex colour across.
This is the "printable/solid" branch of mesh repair; it trades a little
geometric sharpness for a closed surface with no visible gaps.

    python remesh_solid.py in.ply out.ply --pitch 0.0025 --smooth 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree


def largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if len(mesh.faces) == 0:
        return mesh
    comps = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(mesh.faces))
    )
    if len(comps) <= 1:
        return mesh
    keep = max(comps, key=len)
    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[keep] = True
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()
    return mesh


def transfer_colors(src: trimesh.Trimesh, dst: trimesh.Trimesh) -> np.ndarray:
    colors = getattr(src.visual, "vertex_colors", None)
    if colors is None or len(colors) != len(src.vertices):
        return np.full((len(dst.vertices), 4), 200, dtype=np.uint8)
    tree = cKDTree(np.asarray(src.vertices, dtype=np.float64))
    _, idx = tree.query(np.asarray(dst.vertices, dtype=np.float64), k=1)
    out = np.asarray(colors, dtype=np.uint8)[idx]
    if out.shape[1] == 3:
        out = np.column_stack([out, np.full(len(out), 255, dtype=np.uint8)])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--pitch", type=float, default=0.0025, help="voxel side in scene units")
    ap.add_argument("--smooth", type=int, default=8, help="Taubin iterations, 0 disables")
    ap.add_argument("--close", type=int, default=1, help="dilate/erode passes that bridge split shells")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    src = trimesh.load(args.input, process=False)
    if isinstance(src, trimesh.Scene):
        src = trimesh.util.concatenate(list(src.geometry.values()))
    before = {"vertices": int(len(src.vertices)), "faces": int(len(src.faces)),
              "watertight": bool(src.is_watertight)}

    grid = trimesh.voxel.creation.voxelize(src, pitch=args.pitch, method="subdivide")
    if grid is None:
        raise SystemExit("voxelisation failed")
    occ = np.asarray(grid.matrix, dtype=bool)
    if args.close > 0:
        occ = ndimage.binary_dilation(occ, iterations=args.close)
    occ = ndimage.binary_fill_holes(occ)
    if args.close > 0:
        occ = ndimage.binary_erosion(occ, iterations=args.close)
    grid = trimesh.voxel.VoxelGrid(occ, transform=grid.transform)
    solid = grid.marching_cubes
    solid.apply_transform(grid.transform)
    solid = largest_component(solid)

    if args.smooth > 0:
        trimesh.smoothing.filter_taubin(solid, lamb=0.5, nu=-0.53, iterations=args.smooth)

    solid.visual = trimesh.visual.ColorVisuals(vertex_colors=transfer_colors(src, solid))

    after = {
        "vertices": int(len(solid.vertices)),
        "faces": int(len(solid.faces)),
        "watertight": bool(solid.is_watertight),
        "components": len(trimesh.graph.connected_components(
            solid.face_adjacency, nodes=np.arange(len(solid.faces)))),
        "euler": int(solid.euler_number),
        "volume": round(float(solid.volume), 6),
        "grid_shape": [int(x) for x in grid.shape],
        "pitch": args.pitch,
        "closed": args.close,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    solid.export(args.output)
    report = {"input": args.input, "output": args.output, "before": before, "after": after}
    text = json.dumps(report, indent=2)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
