# Mesh repair — measured on Missy

Goal: no holes, manifold topology, and as much real detail as the capture allows.

## What the raw extraction looks like

`mesh_vertexcolor.ply` (opacity, iso 0.35, floor 1000, samples 1):

| metric | value |
|---|---|
| vertices / faces | 161,102 / 323,848 |
| components | 7 |
| boundary edges | **12,414 in 353 loops** |
| non-manifold edges | 0 |
| Spirula `--check 1` | PASS (the checker does not count boundaries as a defect) |

The large loop (9,436 edges, ~10 cm) is the unobserved underside; 681 of the 825 loops found later
are 3–6 edge micro-slits between fur sheets.

## Three repairs, three outcomes

| approach | faces | boundary | self-intersections | check |
|---|---|---|---|---|
| A. cap every loop with a centroid fan | 335,472 | 0 | 72,243 | FAIL |
| B. cull-unseen 0 + binary hole fill | 873,820 | 4,384 | 0 | **PASS** |
| C. B + external caps (826 loops) | 877,678 | 0 | 9,622 | FAIL |
| **D. C solidified (voxel + flood fill)** | **336,630** | **0** | **0** | **PASS** |

Key finding: **capping micro-slits creates self-intersections.** Closing a 3-edge slit inside fur
forces a triangle through neighbouring geometry; the same is true of any learned "hole filler" that
guesses geometry. The binary's own extraction (B) is the clean high-detail mesh; it still has small
openings. D is the clean solid mesh; it loses geometric fur detail but keeps colour.

## D — the solid branch, and the finer pass

The first solid used the 2.5 mm copy mesh:

```
remesh_solid.py capped.ply solid.ply --pitch 0.0025 --smooth 4 --close 1
bake_texture.py --target solid.ply --source mesh_textured.glb --out solid_baked.ply
```

That passed (336,630 faces, 0 boundary, 0 non-manifold, 0 self-intersections, 0 folds) but visibly
softened the fur. Re-solidifying the **873 k detail extraction** at a finer pitch recovers it:

| pitch | faces | boundary | self-int | folds | `--check 1` |
|---|---|---|---|---|---|
| 2.5 mm | 336,630 | 0 | 0 | 0 | PASS (soft) |
| 2.1 mm | 705,852 | 0 | 0 | 0 | PASS |
| 2.0 mm | 808,862 | 0 | 0 | 1 | FAIL |
| **1.9 mm** | **917,700** | **0** | **0** | **0** | **PASS** |
| 1.5 mm | 1,611,846 | 0 | 0 | 1 | FAIL |

The delivered mesh is the 1.9 mm solid: 457,606 vertices / 917,700 faces, 1 component, 0
non-manifold, 0 self-intersections, 0 folds, 4,096² atlas baked (median 4.6 mm). 2.0 and 1.5 mm each
land on a single fold, and extra Taubin adds self-intersections, so screen pitch in 0.1 mm steps at
smooth 2.

The atlas bake uses the 4,096² texture embedded in `mesh_textured.glb`; the UV convention is `1 − v`
(using `v` directly gave MAE 38.6 vs 1.5 against the known vertex colours). `on_surface` refuses
>~1M candidate faces, so `bake_texture.py` projects 20 k vertices per call.

## When to use which

- **Appearance / turntable:** the splat. The mesh is a byproduct of a fur-covered 87-camera capture.
- **Downstream geometry (collision, print, export):** the 1.9 mm solid. Watertight, manifold, PASS,
  and visibly detailed; the 2.5 mm solid is the lighter fallback.
- **Raw maximum detail, closure not required:** B (873 k faces, 4,384 small openings).

## Traps

- `trimesh.Trimesh.split()` can `MemoryError` on large textured meshes; use `graph.connected_components`
  instead.
- `mesh.edges_sorted` is **per face** (3F rows); `mesh.edges_unique`/`edges_unique_inverse`/`edges_face`
  are the unique-edge views. Mixing them silently mis-indexes.
- A centroid fan is manifold by construction; ear clipping is not, because a diagonal may already
  exist as a mesh edge.
- Ray-parity voxelisation of overlapping shells produces internal tunnels (watertight, volume 1.66 L,
  Euler −23,324). Surface voxelisation + `binary_fill_holes` is the correct solidification.
- Machining the remesh at 2.5 mm and 4 Taubin iterations preserves the silhouette without melting the
  legs; more smoothing blobs the toes.
