#!/usr/bin/env python3
"""Report which splat-repair path this machine can actually run.

Run it before promising a fixer. It checks the GPU, torch, diffusers, the
Difix pipeline class, the Hugging Face cache and the licence gate. Exits 0
when at least the deterministic path works (it always does).
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


def main() -> int:
    out = {"deterministic_prune": True, "artifixer": {}, "difix": {}}
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            out["gpu"] = p.name
            out["vram_gb"] = round(p.total_memory / 2**30, 1)
    except Exception as exc:  # noqa: BLE001
        out["torch"] = f"missing ({exc})"
        out["cuda"] = False

    try:
        import diffusers
        out["diffusers"] = diffusers.__version__
        out["difix"]["pipeline_class"] = hasattr(diffusers, "DifixPipeline") or hasattr(diffusers, "DiffixPipeline")
    except Exception as exc:  # noqa: BLE001
        out["diffusers"] = f"missing ({exc})"
        out["difix"]["pipeline_class"] = False

    cache = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub" / "models--nvidia--difix"
    out["difix"]["weights_cached"] = cache.is_dir()
    out["difix"]["licence_env"] = os.environ.get("SPIRULA_DIFIX_LICENSE", "")
    out["difix"]["runnable"] = bool(out["difix"]["pipeline_class"] and out["difix"]["weights_cached"]
                                    and out["difix"]["licence_env"] == "noncommercial")
    out["difix"]["note"] = ("single-step Difix fits 8 GB only at reduced resolution; "
                            "weights are NVIDIA non-commercial")

    vram = out.get("vram_gb", 0)
    out["artifixer"]["vram_ok"] = vram >= 24
    out["artifixer"]["note"] = ("ArtiFixer needs 24-80 GB VRAM and a CUDA Docker image; "
                                "not a local option on this machine")

    print(json.dumps(out, indent=2))
    print("\nverdict:")
    print("  prune_splats.py  READY (no GPU, no licence)")
    print(f"  difix            {'READY' if out['difix']['runnable'] else 'NOT READY'}")
    print("  artifixer        REMOTE ONLY (VRAM floor 24 GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
