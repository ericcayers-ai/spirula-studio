#!/usr/bin/env python3
"""Deterministic splat cleanup: drop floaters before any learned fixer runs.

Reads a 3DGS PLY (binary little endian or ascii), scores every Gaussian by
opacity, world-space size and local support, and writes the survivors with the
same properties. No model, no GPU, no licence.

    python prune_splats.py in.ply out.ply --dry-run
    python prune_splats.py in.ply out.ply --min-opacity 0.02 --max-scale-frac 0.04

Rules (all optional):
  opacity  -- sigmoid(opacity) below --min-opacity
  size     -- max(exp(scale)) above --max-scale-frac * scene diagonal
  support  -- fewer than --min-neighbours in the 3x3x3 voxel block
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

DTYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply(path: Path):
    raw = path.read_bytes()
    marker = raw.find(b"end_header")
    if marker < 0:
        raise SystemExit("not a PLY")
    head_end = raw.find(b"\n", marker) + 1
    header = raw[:head_end].decode("ascii", "replace")
    fmt = re.search(r"format\s+(\S+)", header).group(1)
    n = int(re.search(r"element\s+vertex\s+(\d+)", header).group(1))
    props = re.findall(r"property\s+(\w+)\s+(\w+)", header)
    fields = [(name, DTYPES[typ]) for typ, name in props]
    dtype = np.dtype(fields)
    if fmt == "binary_little_endian":
        arr = np.frombuffer(raw[head_end:head_end + n * dtype.itemsize], dtype=dtype, count=n)
    elif fmt == "ascii":
        arr = np.loadtxt(path, skiprows=header.count("\n"), dtype=dtype, max_rows=n)
    else:
        raise SystemExit(f"unsupported PLY format {fmt}")
    return arr, header, fmt


def write_ply(path: Path, arr, header: str, fmt: str) -> None:
    lines = []
    for line in header.splitlines():
        if line.startswith("element vertex"):
            line = f"element vertex {len(arr)}"
        lines.append(line)
    head = ("\n".join(lines) + "\n").encode("ascii")
    body = arr.tobytes() if fmt == "binary_little_endian" else None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(head)
        if body is not None:
            fh.write(body)
        else:
            for row in arr:
                fh.write((" ".join(str(row[k]) for k in arr.dtype.names) + "\n").encode("ascii"))


def voxel_support(xyz: np.ndarray, res: int) -> np.ndarray:
    lo = np.percentile(xyz, 0.5, axis=0)
    hi = np.percentile(xyz, 99.5, axis=0)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    cell = diag / res
    idx = np.floor((xyz - lo) / cell).astype(np.int64)
    uniq, inv = np.unique(idx, axis=0, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.int64)
    base = 1 << 20
    keys = ((uniq[:, 0] + base) | ((uniq[:, 1] + base) << 21) | ((uniq[:, 2] + base) << 42)).astype(np.int64)
    order = np.argsort(keys)
    keys = keys[order]
    counts = counts[order]
    total = np.zeros(len(xyz), dtype=np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                delta = dx + (dy << 21) + (dz << 42)
                q = keys[inv] + delta
                pos = np.searchsorted(keys, q)
                pos = np.clip(pos, 0, len(keys) - 1)
                hit = keys[pos] == q
                total += np.where(hit, counts[pos], 0)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--min-opacity", type=float, default=0.01)
    ap.add_argument("--max-scale-frac", type=float, default=0.05)
    ap.add_argument("--min-neighbours", type=int, default=3)
    ap.add_argument("--voxel-res", type=int, default=128)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    arr, header, fmt = read_ply(Path(args.input))
    names = set(arr.dtype.names)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    lo = np.percentile(xyz, 0.5, axis=0)
    hi = np.percentile(xyz, 99.5, axis=0)
    diag = float(np.linalg.norm(hi - lo)) or 1.0

    keep = np.ones(len(arr), dtype=bool)
    reasons = {}
    if "opacity" in names:
        op = 1.0 / (1.0 + np.exp(-arr["opacity"].astype(np.float64)))
        bad = op < args.min_opacity
        reasons["opacity"] = int(bad.sum())
        keep &= ~bad
    if all(f"scale_{i}" in names for i in range(3)):
        scale = np.stack([arr[f"scale_{i}"] for i in range(3)], axis=1)
        big = np.exp(scale).max(axis=1) > args.max_scale_frac * diag
        reasons["size"] = int(big.sum())
        keep &= ~big
    support = voxel_support(xyz, args.voxel_res)
    lonely = support < args.min_neighbours
    reasons["support"] = int(lonely.sum())
    keep &= ~lonely

    report = {
        "input": args.input, "splats": int(len(arr)), "kept": int(keep.sum()),
        "removed": int((~keep).sum()), "removed_by_rule": reasons,
        "scene_diagonal": round(diag, 4), "voxel_res": args.voxel_res,
        "thresholds": {"min_opacity": args.min_opacity,
                       "max_scale_frac": args.max_scale_frac,
                       "min_neighbours": args.min_neighbours},
    }
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.dry_run:
        write_ply(Path(args.output), arr[keep], header, fmt)
        print(f"wrote {args.output} ({int(keep.sum())} splats)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
