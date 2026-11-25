# Integrations — ArtiFixer-class splat repair and mesh repair

Two separate problems, deliberately kept apart:

| track | question | runs where |
|---|---|---|
| `splat_repair/` | the **rendered view** has artifacts (floaters, under-observed blur) | render-time or post-process |
| `mesh_repair/` | the **extracted surface** has holes, bad topology, missing detail | offline, CPU |

Nothing here edits a Spirula source file, so `git pull` / fast-forward never conflicts. Work on the
`integrations` branch; `master` stays identical to upstream.

## Decision matrix — splat side (measured 2026-11-25)

Hardware: RTX 4060, **8 GB**, Windows, torch 2.13+cu126, diffusers 0.40.

| system | method | stated hardware | licence | verdict |
|---|---|---|---|---|
| **ArtiFixer** (nv-tlabs) | autoregressive video diffusion + 3DGRUT distillation | 24–80 GB, CUDA Docker, ~160 GB disk | Apache-2.0 code; checkpoints separate | **remote only** — cannot run here |
| **Difix3D+ / Difix** (nv-tlabs) | single-step image diffusion, distill + online enhancer | 0.9 B UNet; 8 GB plausible at reduced res | **non-commercial** (NVIDIA + SD-Turbo) | **optional local**, licence-gated |
| GSFixer | CogVideoX-I2V 5 B + VGGT/DINOv2 guidance | tested on H20/H100 | research | no |
| 3DGS-Enhancer | Stable Video Diffusion priors | 16–24 GB class | research | no |
| SplatFormer | PTv3 point transformer | ~8 GB but flash-attn/CUDA build | research; synthetic Objaverse training | no (domain + Windows build) |
| **prune_splats.py** (this repo) | opacity / scale / local-support filter | CPU | MIT | **yes** — always-on baseline |

Conclusion: **ArtiFixer is not a local runtime on this machine.** Its idea (clean the render, then
distill back) is represented by the Difix bridge, which is the only class-compatible model; the
deterministic pruner is what actually runs by default. Difix is not auto-downloaded.

## Decision matrix — mesh side

| tool | licence | verdict |
|---|---|---|
| trimesh + scipy (+ rtree) | MIT / BSD | **used** for welding, components, caps, solid remesh, colour transfer |
| PyMeshLab / MeshFix | GPL-3 | not used (licence), and not needed |
| Blender headless | GPL, 300 MB+ | not installed; not required |
| Open3D | MIT | not required; Poisson is an option if a 3.12 env is added |
| Spirula `mesh --method tsdf` | in-repo | tested at 3 mm: 173 parts, worse than opacity |
| voxel remesh + marching cubes | scipy/skimage | **used** for the solid deliverable |

## Commands

```bat
python integrations\splat_repair\doctor.py
python integrations\splat_repair\prune_splats.py run.ply pruned.ply --dry-run
python integrations\splat_repair\prune_splats.py run.ply pruned.ply

python integrations\mesh_repair\repair_mesh.py raw.ply capped.ply --keep-components 0
python integrations\mesh_repair\remesh_solid.py capped.ply solid.ply --pitch 0.0025
python integrations\mesh_repair\bake_texture.py --target solid.ply --source mesh_textured.glb --out baked.ply
```

`repair_mesh.py` closes boundary loops; `remesh_solid.py` produces the printable surface;
`bake_texture.py` projects the 4K atlas back on as colour. See `mesh_repair/README.md` for the
measured Missy comparison and the trade-off that made the solid branch the deliverable.

## Rules

- No non-commercial weights in the repository or in a package. `difix_fix.py` refuses without
  `SPIRULA_DIFIX_LICENSE=noncommercial`.
- No edits to upstream files. Integration code lives only under `integrations/`.
- Every mesh claim gets `spirula mesh <file> --check 1` and a rendered screenshot.
