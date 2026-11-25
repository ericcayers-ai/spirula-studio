#!/usr/bin/env python3
"""Repair a triangle mesh extracted from a splat: close holes, drop junk, fix topology.

External to Spirula core on purpose. Needs numpy + trimesh (+ mapbox_earcut for small loops).

    python repair_mesh.py in.glb out.glb --report report.json

What it does, in order:
  1. load (GLB texture / PLY vertex colour preserved)
  2. drop degenerate and duplicate faces (welds only when there is no UV seam to protect)
  3. keep the largest components (drop floaters)
  4. find every boundary loop and triangulate it (earcut <= 64 edges, centroid fan above)
  5. orient the new faces with the surrounding surface, optionally relax the fill
  6. rebuild the visual layer: UVs for texture, interpolated colour otherwise
  7. write the mesh and a JSON report with before/after counts

It never edits the Spirula source tree, so upstream updates cannot conflict with it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh


# ---------------------------------------------------------------- load / stats

def load_mesh(path: Path) -> trimesh.Trimesh:
    obj = trimesh.load(str(path), process=False)
    if isinstance(obj, trimesh.Scene):
        geoms = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise SystemExit(f"no triangle geometry in {path}")
        mesh = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)
    elif isinstance(obj, trimesh.Trimesh):
        mesh = obj
    else:
        raise SystemExit(f"unsupported geometry in {path}: {type(obj)}")
    mesh = mesh.copy()
    return mesh


def face_components(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    if len(mesh.faces) == 0:
        return []
    edges = mesh.face_adjacency
    nodes = np.arange(len(mesh.faces))
    if len(edges) == 0:
        return [nodes]
    return trimesh.graph.connected_components(edges, nodes=nodes)


def stats(mesh: trimesh.Trimesh) -> dict:
    uniq, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    comps = face_components(mesh)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": len(comps),
        "component_faces": sorted((int(len(c)) for c in comps), reverse=True)[:12],
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "watertight": bool(mesh.is_watertight),
        "euler": int(mesh.euler_number),
    }


# ---------------------------------------------------------------- hole filling

def boundary_loops(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    """Ordered vertex loops of every boundary edge, robust to junction vertices."""
    if len(mesh.faces) == 0:
        return []
    uniq, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    bedges = uniq[counts == 1]
    if len(bedges) == 0:
        return []

    adj: dict[int, list[int]] = defaultdict(list)
    key = {}
    for a, b in bedges:
        a, b = int(a), int(b)
        adj[a].append(b)
        adj[b].append(a)
        key[(min(a, b), max(a, b))] = True

    used: set[tuple[int, int]] = set()
    loops: list[np.ndarray] = []
    for a, b in bedges:
        a, b = int(a), int(b)
        ek = (min(a, b), max(a, b))
        if ek in used:
            continue
        loop = [a, b]
        used.add(ek)
        prev, cur = a, b
        while True:
            nxts = [n for n in adj[cur] if n != prev and (min(cur, n), max(cur, n)) not in used]
            if not nxts:
                break
            nxt = nxts[0]
            ek2 = (min(cur, nxt), max(cur, nxt))
            used.add(ek2)
            if nxt == loop[0]:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3:
            loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def _plane_project(points: np.ndarray):
    c = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - c, full_matrices=False)
    normal = vh[2]
    u = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(normal, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    local = points - c
    return np.stack([local @ u, local @ v], axis=1), normal


def boundary_normals(mesh: trimesh.Trimesh) -> dict[tuple[int, int], np.ndarray]:
    """Map every boundary edge to the normal of its single adjacent face."""
    if len(mesh.faces) == 0:
        return {}
    uniq = np.asarray(mesh.edges_unique, dtype=np.int64)
    inverse = np.asarray(mesh.edges_unique_inverse, dtype=np.int64)
    counts = np.bincount(inverse, minlength=len(uniq))
    _, first = np.unique(inverse, return_index=True)
    face_of_edge = np.asarray(mesh.edges_face, dtype=np.int64)[first]
    mask = counts == 1
    keys = uniq[mask]
    faces = face_of_edge[mask]
    normals = np.asarray(mesh.face_normals, dtype=np.float64)[faces]
    return {(int(a), int(b)): normals[i] for i, (a, b) in enumerate(keys)}


def _adjacent_normal(loop: np.ndarray, normals: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    acc = np.zeros(3, dtype=np.float64)
    for i in range(len(loop)):
        a, b = int(loop[i]), int(loop[(i + 1) % len(loop)])
        n = normals.get((min(a, b), max(a, b)))
        if n is not None:
            acc += n
    ln = np.linalg.norm(acc)
    return acc / ln if ln > 1e-9 else acc


def triangulate_loop(points: np.ndarray, ref_normal: np.ndarray):
    """Return (local_faces, extra_point|None) for one boundary loop.

    Local indices 0..n-1 are the loop vertices; index n (when present) is the
    appended centroid. Faces are already wound to agree with ref_normal.
    """
    n = len(points)
    extra = None
    tris = None

    if n == 3:
        tris = np.array([[0, 1, 2]], dtype=np.int64)
    else:
        # A centroid fan cannot create a diagonal that already exists in the
        # mesh, so each boundary edge gains exactly one face and the result is
        # manifold by construction (earcut can and did create 3-face edges).
        extra = points.mean(axis=0)
        tris = np.array([[n, i, (i + 1) % n] for i in range(n)], dtype=np.int64)

    # orient every new face with the surrounding surface (or the polygon normal)
    v = points if extra is None else np.vstack([points, extra[None, :]])
    for k in range(len(tris)):
        a, b, c = tris[k]
        fn = np.cross(v[b] - v[a], v[c] - v[a])
        if np.dot(fn, ref_normal) < 0:
            tris[k] = [a, c, b]
    return tris, extra


def fill_holes(mesh: trimesh.Trimesh, max_loop_edges: int, smooth_iters: int):
    loops = boundary_loops(mesh)
    if not loops:
        return {"loops": 0, "filled": 0, "added_vertices": 0, "added_faces": 0, "largest": []}

    verts = [np.asarray(mesh.vertices, dtype=np.float64)]
    faces = [np.asarray(mesh.faces, dtype=np.int64)]
    base = len(verts[0])
    extra_verts = []
    extra_faces = []
    filled = 0
    sizes = []
    normals = boundary_normals(mesh)

    for loop in loops:
        if len(loop) > max_loop_edges:
            continue
        pts = verts[0][loop]
        _, normal = _plane_project(pts)
        ref = _adjacent_normal(loop, normals)
        if np.linalg.norm(ref) < 1e-9:
            ref = normal
        tris, extra = triangulate_loop(pts, ref)
        if extra is None:
            global_tris = loop[tris]
        else:
            local_n = len(loop)
            extra_idx = base + len(extra_verts)
            extra_verts.append(np.asarray(extra, dtype=np.float64))
            mapped = np.empty_like(tris)
            mask = tris == local_n
            mapped[~mask] = loop[tris[~mask]]
            mapped[mask] = extra_idx
            global_tris = mapped
        extra_faces.append(global_tris)
        filled += 1
        sizes.append(int(len(loop)))

    if not extra_faces:
        return {"loops": len(loops), "filled": 0, "added_vertices": 0, "added_faces": 0, "largest": []}

    new_v = np.vstack(verts + [np.asarray(extra_verts, dtype=np.float64)]) if extra_verts else verts[0]
    new_f = np.vstack(faces + [np.asarray(np.vstack(extra_faces), dtype=np.int64)])

    # relax only the appended centroid vertices
    if smooth_iters > 0 and extra_verts:
        new_v = _relax(new_v, new_f, base, smooth_iters)

    out = trimesh.Trimesh(vertices=new_v, faces=new_f, process=False)
    return {
        "loops": len(loops),
        "filled": filled,
        "added_vertices": int(len(new_v) - base),
        "added_faces": int(len(new_f) - len(faces[0])),
        "largest": sorted(sizes, reverse=True)[:12],
        "_mesh": out,
    }


def drop_small_components(mesh: trimesh.Trimesh, min_faces: int) -> int:
    comps = face_components(mesh)
    if not comps:
        return 0
    keep = [c for c in comps if len(c) >= min_faces]
    if len(keep) == len(comps):
        return 0
    mask = np.zeros(len(mesh.faces), dtype=bool)
    for c in keep:
        mask[c] = True
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()
    return len(comps) - len(keep)


def remove_nonmanifold_faces(mesh: trimesh.Trimesh, max_rounds: int = 8) -> int:
    """Delete the smallest face at every edge shared by more than two faces."""
    removed = 0
    for _ in range(max_rounds):
        if len(mesh.faces) == 0:
            break
        uniq, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
        bad = uniq[counts > 2]
        if len(bad) == 0:
            break
        face_edges = mesh.edges_sorted
        keep = np.ones(len(mesh.faces), dtype=bool)
        for a, b in bad:
            hit = np.where((face_edges[:, 0] == a) & (face_edges[:, 1] == b))[0]
            if len(hit) <= 2:
                continue
            faces = np.unique(hit // 3)
            areas = mesh.area_faces[faces]
            for f in faces[np.argsort(areas)[:-2]]:
                keep[f] = False
        if keep.all():
            break
        removed += int((~keep).sum())
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
    return removed


def _relax(vertices, faces, first_new, iters):
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    neigh: dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        neigh[int(a)].add(int(b))
        neigh[int(b)].add(int(a))
    v = vertices.copy()
    new_ids = [i for i in range(first_new, len(v))]
    for _ in range(iters):
        upd = v.copy()
        for i in new_ids:
            ns = neigh.get(i)
            if ns:
                upd[i] = 0.5 * v[i] + 0.5 * v[list(ns)].mean(axis=0)
        v = upd
    return v


# ---------------------------------------------------------------- visual layer

def rebuild_visual(src: trimesh.Trimesh, dst: trimesh.Trimesh):
    """Carry texture UVs or vertex colours onto the repaired mesh."""
    kind = getattr(src.visual, "kind", None)
    n_src = len(src.vertices)
    n_dst = len(dst.vertices)
    if n_dst < n_src:
        return dst

    if kind == "texture":
        uv = getattr(src.visual, "uv", None)
        mat = getattr(src.visual, "material", None)
        if uv is not None and mat is not None:
            uv = np.asarray(uv, dtype=np.float64)
            if len(uv) == n_src:
                new_uv = np.zeros((n_dst, 2), dtype=np.float64)
                new_uv[:n_src] = uv
                # appended vertices take the UV of the nearest original vertex
                ref = np.asarray(src.vertices)
                for i in range(n_src, n_dst):
                    j = int(np.argmin(((ref - dst.vertices[i]) ** 2).sum(axis=1)))
                    new_uv[i] = uv[j]
                dst.visual = trimesh.visual.TextureVisuals(uv=new_uv, material=mat)
                return dst

    colors = getattr(src.visual, "vertex_colors", None)
    if colors is not None and len(colors) == n_src:
        colors = np.asarray(colors)
        new_c = np.zeros((n_dst, colors.shape[1]), dtype=colors.dtype)
        new_c[:n_src] = colors
        ref = np.asarray(src.vertices)
        for i in range(n_src, n_dst):
            j = int(np.argmin(((ref - dst.vertices[i]) ** 2).sum(axis=1)))
            new_c[i] = colors[j]
        dst.visual = trimesh.visual.ColorVisuals(vertex_colors=new_c)
    return dst


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="Close holes and clean topology of a splat mesh.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--report", default=None)
    ap.add_argument("--keep-components", type=int, default=1, help="0 keeps every component above the face floor")
    ap.add_argument("--min-component-faces", type=int, default=200)
    ap.add_argument("--max-loop-edges", type=int, default=20000)
    ap.add_argument("--fill-passes", type=int, default=3)
    ap.add_argument("--smooth-fill", type=int, default=2)
    ap.add_argument("--no-fill", action="store_true")
    ap.add_argument("--no-normals", action="store_true")
    args = ap.parse_args()

    inp, outp = Path(args.input), Path(args.output)
    mesh = load_mesh(inp)
    before = stats(mesh)

    # weld only when UV seams cannot be damaged
    if getattr(mesh.visual, "kind", None) != "texture":
        mesh.merge_vertices()

    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    comps = face_components(mesh)
    if comps:
        order = sorted(comps, key=lambda c: len(c), reverse=True)
        keep_n = args.keep_components if args.keep_components > 0 else len(order)
        keep = [c for c in order[:keep_n] if len(c) >= args.min_component_faces]
        if keep and len(keep) < len(order):
            mask = np.zeros(len(mesh.faces), dtype=bool)
            for c in keep:
                mask[c] = True
            mesh.update_faces(mask)
            mesh.remove_unreferenced_vertices()

    fill = {"passes": [], "loops": 0, "filled": 0, "added_vertices": 0, "added_faces": 0, "largest": []}
    if not args.no_fill:
        for _ in range(max(1, args.fill_passes)):
            step = fill_holes(mesh, args.max_loop_edges, args.smooth_fill)
            added = step.pop("_mesh", None)
            if added is not None:
                mesh = rebuild_visual(mesh, added)
            step["dropped_components"] = drop_small_components(mesh, args.min_component_faces)
            step["removed_nonmanifold_faces"] = remove_nonmanifold_faces(mesh)
            fill["passes"].append(step)
            for k in ("loops", "filled", "added_vertices", "added_faces"):
                fill[k] += step[k]
            fill["largest"] = sorted(fill["largest"] + step["largest"], reverse=True)[:12]
            if step["filled"] == 0 and step["removed_nonmanifold_faces"] == 0:
                break

    drop_small_components(mesh, args.min_component_faces)
    if not args.no_normals:
        try:
            trimesh.repair.fix_normals(mesh)
        except Exception as exc:  # noqa: BLE001
            print(f"normal repair skipped: {exc}", file=sys.stderr)

    after = stats(mesh)
    out = outp
    if out.suffix.lower() == ".ply" and getattr(mesh.visual, "kind", None) == "texture":
        # PLY cannot carry the atlas: sample it into vertex colours instead
        mesh.visual = trimesh.visual.ColorVisuals(mesh)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))

    report = {"input": str(inp), "output": str(out), "before": before, "after": after, "fill": fill}
    text = json.dumps(report, indent=2)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
