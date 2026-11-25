#!/usr/bin/env python3
"""Bake the 4K atlas of a textured GLB onto another mesh as vertex colours.

The solid remesh is smooth by construction, so the photographic detail has to
come back as colour. Every target vertex is projected onto the textured source
surface, its barycentric UV is looked up, and the atlas is sampled there.

    python bake_texture.py --target solid.ply --source mesh_textured.glb --out baked.ply
"""
from __future__ import annotations

import argparse
import io
import json
import struct
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def glb_image(path: Path) -> Image.Image:
    raw = path.read_bytes()
    _, _, length = struct.unpack_from("<III", raw, 0)
    off, js, blob = 12, None, None
    while off < length:
        clen, ctype = struct.unpack_from("<II", raw, off)
        data = raw[off + 8:off + 8 + clen]
        if ctype == 0x4E4F534A:
            js = json.loads(data.decode("utf-8").rstrip("\x00 "))
        elif ctype == 0x004E4942:
            blob = data
        off += 8 + clen
    if js is None or blob is None:
        raise SystemExit("not a GLB with a binary chunk")
    view = js["images"][0]["bufferView"]
    bv = js["bufferViews"][view]
    start = bv.get("byteOffset", 0)
    chunk = blob[start:start + bv["byteLength"]]
    return Image.open(io.BytesIO(chunk)).convert("RGB")


def barycentric(points, tri):
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0, v1, v2 = b - a, c - a, points - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-20, 1e-20, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1.0 - v - w
    return np.stack([u, v, w], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--source", required=True, help="textured GLB")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    target = trimesh.load(args.target, process=False)
    if isinstance(target, trimesh.Scene):
        target = trimesh.util.concatenate(list(target.geometry.values()))

    scene = trimesh.load(args.source, process=False)
    src = list(scene.geometry.values())[0] if isinstance(scene, trimesh.Scene) else scene
    uv = np.asarray(src.visual.uv, dtype=np.float64)
    image = glb_image(Path(args.source))
    img = np.asarray(image, dtype=np.uint8)
    h, w = img.shape[:2]

    query = trimesh.proximity.ProximityQuery(src)
    pts, dist, tri = query.on_surface(np.asarray(target.vertices, dtype=np.float64))
    bary = barycentric(pts, np.asarray(src.triangles)[tri])
    tri_uv = uv[np.asarray(src.faces)[tri]]
    out_uv = np.einsum("ij,ijk->ik", bary, tri_uv)

    # trimesh's UVs keep glTF's bottom-left origin; image rows run down
    xs = np.clip((out_uv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(np.int64)
    ys = np.clip((1.0 - out_uv[:, 1] % 1.0) * (h - 1), 0, h - 1).astype(np.int64)
    colors = img[ys, xs]
    rgba = np.column_stack([colors, np.full(len(colors), 255, dtype=np.uint8)])
    target.visual = trimesh.visual.ColorVisuals(vertex_colors=rgba)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    target.export(args.out)
    report = {
        "target": args.target, "source": args.source, "out": args.out,
        "texture": [int(w), int(h)],
        "projected": int(len(pts)),
        "distance_median": round(float(np.median(dist)), 6),
        "distance_p95": round(float(np.percentile(dist, 95)), 6),
        "distance_max": round(float(np.max(dist)), 6),
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
