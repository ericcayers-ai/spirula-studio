#!/usr/bin/env python3
"""Optional Difix post-process for rendered splat frames (licence-gated).

Difix is NVIDIA's single-step diffusion fixer for under-constrained novel
views. It is the only fixer in ArtiFixer's class that can plausibly run on an
8 GB card, and its weights are **non-commercial**. Nothing is downloaded until
you opt in:

    set SPIRULA_DIFIX_LICENSE=noncommercial
    python difix_fix.py --in frames/ --out fixed/

If the installed diffusers has no Difix pipeline (0.40 does not), drop the
pipeline file from nv-tlabs/Difix3D into ./vendor/difix_pipeline.py or install
their pinned environment. The script refuses to guess.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def load_pipeline_cls():
    for name in ("DifixPipeline", "DiffixPipeline"):
        try:
            import diffusers

            cls = getattr(diffusers, name, None)
            if cls is not None:
                return cls, f"diffusers.{name}"
        except Exception:  # noqa: BLE001
            pass
    vendor = Path(__file__).with_name("vendor") / "difix_pipeline.py"
    if vendor.exists():
        spec = importlib.util.spec_from_file_location("difix_pipeline", vendor)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in ("DifixPipeline", "DiffixPipeline"):
            cls = getattr(mod, name, None)
            if cls is not None:
                return cls, str(vendor)
    return None, None


def run_one(pipe, image, prompt, size):
    from PIL import Image

    if size and max(image.size) > size:
        scale = size / max(image.size)
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                             Image.LANCZOS)
    for kwargs in ({"num_inference_steps": 1, "timesteps": [999], "guidance_scale": 0.0},
                   {"num_inference_steps": 1, "guidance_scale": 0.0}):
        try:
            return pipe(prompt=prompt, image=image, **kwargs).images[0]
        except TypeError:
            continue
    raise RuntimeError("pipeline call signature not recognised")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="image or folder of renders")
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--model", default="nvidia/difix")
    ap.add_argument("--prompt", default="remove degradation")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--accept-noncommercial", action="store_true")
    args = ap.parse_args()

    if os.environ.get("SPIRULA_DIFIX_LICENSE") != "noncommercial" and not args.accept_noncommercial:
        print("refused: Difix weights are non-commercial (NVIDIA License + SD-Turbo terms).\n"
              "set SPIRULA_DIFIX_LICENSE=noncommercial or pass --accept-noncommercial",
              file=sys.stderr)
        return 2

    cls, origin = load_pipeline_cls()
    if cls is None:
        print("Difix pipeline not available.\n"
              "  diffusers 0.40 has no DifixPipeline.\n"
              "  copy Difix3D's pipeline into integrations/splat_repair/vendor/difix_pipeline.py\n"
              "  or install nv-tlabs/Difix3D's environment. See README.md.", file=sys.stderr)
        return 2

    import torch
    from PIL import Image

    pipe = cls.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe.to("cuda")
    for method in ("enable_attention_slicing", "enable_vae_slicing"):
        if hasattr(pipe, method):
            getattr(pipe, method)()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXT)
    for path in files:
        image = Image.open(path).convert("RGB")
        fixed = run_one(pipe, image, args.prompt, args.max_side)
        out = dst / f"{path.stem}_difix.png"
        fixed.save(out)
        print(f"{path} -> {out}")
    print(f"pipeline: {origin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
