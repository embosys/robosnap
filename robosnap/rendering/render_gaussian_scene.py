#!/usr/bin/env python3
"""Render one Gaussian PLY with gsplat, without Genesis, Nyx, or foreground meshes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .render_layered_scene import load_camera, load_gaussians, render_gaussians


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background-ply", type=Path, required=True)
    parser.add_argument(
        "--camera-npz",
        type=Path,
        required=True,
        help="Lyra-2 cameras.npz containing w2c_render and render intrinsics",
    )
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--status-json", type=Path)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--device", default=os.environ.get("ROBOSNAP_DEVICE", "cuda:0"))
    parser.add_argument("--background-color", nargs=3, type=float, default=[0.96, 0.96, 0.96])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import numpy as np
    from PIL import Image

    w2c, intrinsics, width, height, frame_idx, intrinsic_key = load_camera(
        args.camera_npz,
        args.frame_index,
        args.image_width,
        args.image_height,
    )
    gaussians = load_gaussians(args.background_ply, args.device)
    rgb, depth, alpha = render_gaussians(
        gaussians,
        w2c,
        intrinsics,
        width,
        height,
        args.background_color,
    )

    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255.0).round().astype(np.uint8)).save(args.output_image)
    status_path = args.status_json or args.output_image.with_suffix(".json")
    status = {
        "status": "ok",
        "renderer": "gsplat",
        "background_ply": str(args.background_ply.resolve()),
        "camera_npz": str(args.camera_npz.resolve()),
        "camera_frame": frame_idx,
        "intrinsic_key": intrinsic_key,
        "intrinsics": intrinsics.tolist(),
        "w2c": w2c.tolist(),
        "image_size_wh": [width, height],
        "gaussians": int(len(gaussians["means_np"])),
        "rgb_std": float(rgb.std()),
        "alpha_mean": float(alpha.mean()),
        "alpha_coverage_002": float((alpha > 0.02).mean()),
        "valid_depth_coverage": float((depth > 0.0).mean()),
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(f"[render-gaussian] wrote {args.output_image}")
    print(f"[render-gaussian] wrote {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
