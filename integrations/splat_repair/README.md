# Splat repair — what can run on 8 GB

## Decision

ArtiFixer, GSFixer, 3DGS-Enhancer and SplatFormer are all out of reach on this machine (24–80 GB,
H100-class, or a Windows-hostile CUDA build). Difix3D+'s Difix is the only single-step model in the
class that fits 8 GB, and its weights are **non-commercial**. So:

1. **default:** `prune_splats.py` — deterministic floater/size/support cleanup, CPU, MIT.
2. **optional:** `difix_fix.py` — licence-gated Difix post-process for stills.
3. **remote only:** ArtiFixer, as a documented recipe (see `integrations/README.md`).

## prune_splats.py

```
python prune_splats.py run.ply pruned.ply --dry-run
```

Missy measured (2,980,586 splats, 0.705 m scene diagonal):

| rule | removed |
|---|---|
| opacity < 0.01 | 42,709 |
| max scale > 5 % of diagonal | 7 |
| support < 3 in the 3×3×3 voxel block | 4,840 |
| **total** | **47,484 (1.6 %)** |

Conservative on purpose: a pruner that removes >5 % of a delivered splat is hiding a problem, not
fixing one.

## difix_fix.py

Prerequisites, none of which are automatic:

1. Difix weights (`nvidia/difix`, 0.9 B, non-commercial).
2. A Difix pipeline class. **diffusers 0.40 does not ship one**; copy Difix3D's into
   `vendor/difix_pipeline.py`, or install their pinned environment.
3. `SPIRULA_DIFIX_LICENSE=noncommercial`.

At 576×1024 on an A100 the paper reports 0.355 s/frame; an 8 GB Ada card will be several times
slower and may need `--max-side 768`. Treat it as an offline enhancer, not real-time.

## doctor.py

`doctor.py` is the gate: it prints GPU/VRAM, torch, diffusers, whether a Difix class exists, whether
the weights are cached, and the licence state. Current output on this machine: prune READY,
Difix NOT READY, ArtiFixer REMOTE ONLY.
