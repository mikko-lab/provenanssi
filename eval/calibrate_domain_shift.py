"""
eval/calibrate_domain_shift.py — Domain-shift calibration experiment.

Hypothesis: calibration slope (not Pearson r) tracks distance from
the training distribution. ResShift was trained on ImageNet bicubic
SR pairs. We calibrate on 4 groups of CC0/PD images spanning intuitive
"distance from natural-image training distribution":

  natural   — CC0/PD outdoor/wildlife scenes (close to ImageNet)
  faces     — CC0 portrait photographs (off-distribution: few portraits in ImageNet)
  texture   — CC0 close-up surface textures (moderately off-distribution)
  synthetic — programmatically generated (gradients, shapes, blobs — far)

CONSTRAINTS (R6, research mode):
- Report what the numbers show; do not tune to produce a clean trend
- A null / noisy result is valid and reported as such
- "Distance from training distribution" is an intuitive proxy; there is
  no rigorous metric. Named as a limitation in the output.
- Group composition documented in full.

Usage
-----
    python eval/calibrate_domain_shift.py [--fast]

--fast: skip ResShift (oracle only, for pipeline smoke-test)
Default: full ResShift run (~105s per group of 3–4 images, GPU recommended)

Output: printed table + eval/domain_shift_results.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from operators.bicubic import BicubicDownsample
from layer.decompose import rectify, CONSISTENCY_EPS
from layer.calibrate import calibrate, is_calibrated, CalibrationResult

# ─── configuration ────────────────────────────────────────────────────────────

SCALE       = 4
N_ENSEMBLE  = 6
N_BINS      = 10
MIN_STD     = 1e-6
TARGET_SIZE = 256   # model input: 256×256 grayscale

# is_calibrated thresholds — FIXED, match falsify.py exactly
MIN_R    = 0.9
MAX_ECE  = 0.3
MIN_SLOP = 0.5
MAX_SLOP = 2.0

RESEARCH_DIR = os.path.join(os.path.dirname(__file__), "research_sources")

# Group definitions: (name, subdirectory, prose description)
GROUPS = [
    (
        "natural",
        "natural",
        "CC0/PD outdoor scenes (boardwalk, frog+log, landscape) — "
        "closest to ImageNet training distribution",
    ),
    (
        "faces",
        "faces",
        "CC0 portrait photographs (Venezuelan subjects, by Wilfredor) — "
        "human faces rare in ImageNet; different aspect ratio / content",
    ),
    (
        "texture",
        "texture",
        "CC0 close-up surface textures (grass, dirt, wood grain) — "
        "same images certified in falsify.py demo; repetitive fine-grained pattern",
    ),
    (
        "synthetic",
        "synthetic",
        "Programmatically generated: linear gradient, radial gradient, "
        "soft blobs (sigma=20), hard shapes (rect+circle+stripe) — "
        "far from any natural-image training distribution",
    ),
]

# CC0/PD source documentation (for the record; full URLs in RESEARCH_SOURCES_LICENCE.md)
SOURCES = {
    "natural": [
        ("boardwalk_nature.jpg",   "Yinan Chen (Goodfreephotos_com)", "CC-PD",
         "https://commons.wikimedia.org/wiki/File:Gfp-wisconsin-madison-the-nature-boardwalk.jpg"),
        ("frog_on_log.jpg",        "Unknown photographer",            "CC0 1.0",
         "https://commons.wikimedia.org/wiki/File:Frog_on_a_log.jpg"),
        ("nature_landscape.jpg",   "Unknown photographer",            "CC0 1.0",
         "https://commons.wikimedia.org/wiki/File:Nature_Landscape_(248036051).jpeg"),
    ],
    "faces": [
        ("boy_face_venezuela.jpg", "Wilfredor", "CC0 1.0",
         "https://commons.wikimedia.org/wiki/File:Boy_Face_from_Venezuela.jpg"),
        ("girl_sad_face.jpg",      "Wilfredor", "CC0 1.0",
         "https://commons.wikimedia.org/wiki/File:Girl_with_sad_face.jpg"),
        ("wayuu_woman.jpg",        "Wilfredor", "CC0 1.0",
         "https://commons.wikimedia.org/wiki/File:Sad_face_of_a_Wayuu_Woman.jpg"),
    ],
    "texture": [
        ("grass_meadow.png",       "Titus Tscharntke", "CC-PD",
         "https://commons.wikimedia.org/wiki/File:Grass_texture.jpg"),
        ("dirt_soil.png",          "Nathan Anderson",  "CC0 1.0",
         "https://commons.wikimedia.org/wiki/File:Dirt_Texture.jpg"),
        ("wood_grain.png",         "Yinan Chen (Goodfreephotos_com)", "CC-PD",
         "https://commons.wikimedia.org/wiki/File:Gfp-wood-texture.jpg"),
    ],
    "synthetic": [
        ("linear_gradient.png",    "generated", "no licence required", ""),
        ("radial_gradient.png",    "generated", "no licence required", ""),
        ("soft_blobs.png",         "generated", "no licence required", ""),
        ("hard_shapes.png",        "generated", "no licence required", ""),
    ],
}


# ─── image loading + preprocessing ───────────────────────────────────────────

def _load_and_crop(path: str, size: int = TARGET_SIZE) -> np.ndarray:
    """Load any image → centre-crop → resize to (size, size) → grayscale float64 [0,1].

    Steps:
      1. Open image (JPEG or PNG), convert to RGB
      2. Resize so the shorter side = size (LANCZOS)
      3. Centre-crop to size×size
      4. Convert to grayscale via BT.601 luminance
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size

    # Resize so shorter side = size
    if w < h:
        new_w = size
        new_h = int(round(h * size / w))
    else:
        new_h = size
        new_w = int(round(w * size / h))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Centre-crop to size×size
    left  = (new_w - size) // 2
    top   = (new_h - size) // 2
    img   = img.crop((left, top, left + size, top + size))

    # Grayscale (BT.601 luminance)
    rgb = np.array(img, dtype=np.float64) / 255.0
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _find_images(directory: str) -> List[str]:
    """Return sorted list of image paths (PNG, JPG, JPEG) in directory."""
    exts = (".png", ".jpg", ".jpeg")
    paths = [
        os.path.join(directory, fn)
        for fn in sorted(os.listdir(directory))
        if fn.lower().endswith(exts)
    ]
    return paths


# ─── calibration run ──────────────────────────────────────────────────────────

@dataclass
class GroupResult:
    name: str
    description: str
    n_images: int
    per_image: List[Tuple[str, float, float, float]]   # (filename, r, slope, ECE)
    pooled: CalibrationResult
    is_cal: bool
    runtime_s: float


def run_group(
    name: str,
    directory: str,
    description: str,
    engine,
    op: BicubicDownsample,
) -> GroupResult:
    image_paths = _find_images(directory)
    if not image_paths:
        raise RuntimeError(f"No images found in {directory}")

    print(f"\n{'═' * 68}")
    print(f"  GROUP: {name}  ({len(image_paths)} images)")
    print(f"  {description}")
    print("═" * 68)

    all_x_outs_per_member: list[list[np.ndarray]] = [[] for _ in range(N_ENSEMBLE)]
    all_x_gt: list[np.ndarray] = []
    per_image_results = []

    t_group = time.perf_counter()

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"\n  [{filename}]")

        x_gt = _load_and_crop(img_path)    # (256, 256) float64 [0,1]
        y    = op.forward(x_gt)            # (64, 64) — ground truth NEVER passed to engine

        t0 = time.perf_counter()
        x_hats = engine.ensemble(y, N_ENSEMBLE)
        rects  = [rectify(xh, y, op) for xh in x_hats]
        dt = time.perf_counter() - t0

        # Verify R3 (data consistency) for each member
        max_res = max(r.residual for r in rects)
        if max_res > CONSISTENCY_EPS:
            print(f"    WARNING: R3 violated: max_residual={max_res:.2e} > {CONSISTENCY_EPS:.0e}")

        x_outs = [r.x_out for r in rects]

        img_cal = calibrate(x_outs, x_gt, n_bins=N_BINS, min_predicted_std=MIN_STD)
        print(f"    r={img_cal.pearson_r:+.4f}  slope={img_cal.slope:.4f}  "
              f"ECE={img_cal.ece:.4f}  (runtime: {dt:.0f}s)")

        per_image_results.append((filename, img_cal.pearson_r, img_cal.slope, img_cal.ece))

        all_x_gt.append(x_gt)
        for j, xo in enumerate(x_outs):
            all_x_outs_per_member[j].append(xo)

    # Pooled calibration across all images in the group
    x_outs_pooled = [np.vstack(all_x_outs_per_member[j]) for j in range(N_ENSEMBLE)]
    x_gt_pooled   = np.vstack(all_x_gt)
    pooled  = calibrate(x_outs_pooled, x_gt_pooled, n_bins=N_BINS, min_predicted_std=MIN_STD)
    verdict = is_calibrated(pooled, MIN_R, MAX_ECE, MIN_SLOP, MAX_SLOP)

    runtime = time.perf_counter() - t_group
    print(f"\n  POOLED:  r={pooled.pearson_r:+.4f}  slope={pooled.slope:.4f}  "
          f"ECE={pooled.ece:.4f}  IS_CALIBRATED={verdict}  (total: {runtime:.0f}s)")

    return GroupResult(
        name=name,
        description=description,
        n_images=len(image_paths),
        per_image=per_image_results,
        pooled=pooled,
        is_cal=verdict,
        runtime_s=runtime,
    )


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Domain-shift calibration experiment")
    parser.add_argument("--fast", action="store_true",
                        help="Use OracleEngine (no GPU, smoke-test only)")
    args = parser.parse_args()

    op = BicubicDownsample(SCALE)

    if args.fast:
        print("Fast mode: using OracleEngine (calibration numbers not meaningful)")
        from engine.oracle import OracleEngine
        engine = OracleEngine(op)
    else:
        print("Loading ResShift engine ...")
        from engine.resshift import ResShiftEngine
        engine = ResShiftEngine(op)
        print("ResShift ready.\n")

    results: list[GroupResult] = []
    for name, subdir, description in GROUPS:
        directory = os.path.join(RESEARCH_DIR, subdir)
        result = run_group(name, directory, description, engine, op)
        results.append(result)

    # ── Final results table ──

    table_lines = []
    table_lines.append("")
    table_lines.append("=" * 80)
    table_lines.append("  DOMAIN-SHIFT CALIBRATION RESULTS")
    table_lines.append("  Model: ResShift + BicubicDownsample(4), N_ensemble=6, 256×256 grayscale")
    table_lines.append("=" * 80)
    table_lines.append("")
    table_lines.append(
        f"  {'Group':<12}  {'N':>2}  {'r (pooled)':>11}  {'slope':>7}  {'ECE':>6}  {'IS_CAL':>8}"
    )
    table_lines.append("  " + "─" * 60)
    for r in results:
        cal_str = "YES" if r.is_cal else "NO"
        table_lines.append(
            f"  {r.name:<12}  {r.n_images:>2}  "
            f"{r.pooled.pearson_r:>+11.4f}  "
            f"{r.pooled.slope:>7.4f}  "
            f"{r.pooled.ece:>6.4f}  "
            f"{cal_str:>8}"
        )
    table_lines.append("  " + "─" * 60)
    table_lines.append("")

    # Reference: certified ImageNet result
    table_lines.append(
        "  Reference (ImageNet, falsify.py --full, commit 83ab9cd):"
    )
    table_lines.append(
        "  natural_imagenet  16  +0.9667      1.5301  0.0282  YES"
    )
    table_lines.append("")
    table_lines.append("  Per-image breakdown:")
    for r in results:
        table_lines.append(f"\n  {r.name}:")
        for fn, pr, sl, ece in r.per_image:
            table_lines.append(f"    {fn:<35}  r={pr:+.4f}  slope={sl:.4f}  ECE={ece:.4f}")

    table_lines.append("")
    table_lines.append("─" * 80)
    table_lines.append("  METHODOLOGY NOTES")
    table_lines.append("─" * 80)
    table_lines.append("""
  'Distance from training distribution' is an intuitive label, not a rigorous
  metric. ResShift was trained on ImageNet bicubic SR pairs; 'natural' images
  resemble that domain most, 'synthetic' images least. No FID or embedding-space
  distance was computed. This ordering is a proxy, not a measurement.

  All images are CC0 / public domain. Sources documented in SOURCES dict above
  and in eval/research_sources/SOURCES_LICENCE.md (written by this script).

  Calibration thresholds are fixed and identical to falsify.py:
    MIN_R=0.9, MAX_ECE=0.3, slope in [0.5, 2.0].

  N_ensemble=6 per image (same as falsify.py --full).
  Images pre-processed: resize shorter side to 256, centre-crop 256×256, BT.601 gray.
""")
    table_lines.append("=" * 80)

    output = "\n".join(table_lines)
    print(output)

    # Write to file
    out_path = os.path.join(os.path.dirname(__file__), "domain_shift_results.txt")
    with open(out_path, "w") as f:
        f.write(output)
    print(f"\nResults written to {out_path}")

    # Write sources licence file
    _write_sources_licence()


def _write_sources_licence() -> None:
    licence_path = os.path.join(RESEARCH_DIR, "SOURCES_LICENCE.md")
    lines = [
        "# Research sources image licence",
        "",
        "All images used in `eval/research_sources/` are CC0 1.0 or public domain.",
        "No attribution is legally required; sources documented here for transparency.",
        "",
    ]
    for group_name, source_list in SOURCES.items():
        lines.append(f"## {group_name}")
        lines.append("")
        for filename, author, licence, url in source_list:
            lines.append(f"### {filename}")
            lines.append(f"- **Author:** {author}")
            lines.append(f"- **Licence:** {licence}")
            if url:
                lines.append(f"- **Source:** {url}")
            lines.append("")

    with open(licence_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Sources licence written to {licence_path}")


if __name__ == "__main__":
    main()
