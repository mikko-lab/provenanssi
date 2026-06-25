"""
eval/phase1_assemble.py — Phase 1: assemble ~40-image sample, measure ResNet50
distances, check axis coverage. STOP and report for human approval before Phase 2.

Does NOT run any calibration (no ResShift). Only:
  1. Verify + download candidate images via Wikimedia API (with delays)
  2. Compute ResNet50 distances for ALL images (existing + new)
  3. Report distance distribution and flag if bimodal
  4. Compute Phase 2 budget estimate
  5. Append pre-registered analysis plan to pre_registration.md

Usage:
    python eval/phase1_assemble.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.parse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from pathlib import Path
from PIL import Image

from eval.distance_metric_v2 import (
    _load_resnet50, _load_rgb_224, _resnet_features,
    IMAGENET_MEAN, IMAGENET_STD,
)

# ─── paths ────────────────────────────────────────────────────────────────────
REPO      = Path(__file__).parent.parent
RES       = Path(__file__).parent / "research_sources"
OUT_DIR   = RES / "large_sample"
OUT_DIR.mkdir(exist_ok=True)

REF_DIR   = REPO / "vendor" / "ResShift" / "testdata" / "Bicubicx4" / "gt"
RESNET_W  = REPO / "weights" / "resnet50-11ad3fa6.pth"
OUT_REPORT= Path(__file__).parent / "phase1_report.txt"
PRE_REG   = Path(__file__).parent / "pre_registration.md"

IMAGENET_MEAN_NP = IMAGENET_MEAN
IMAGENET_STD_NP  = IMAGENET_STD

# ─── candidate images to download ─────────────────────────────────────────────
# Format: (local_label, wikimedia_filename, category, expected_distance_range)
# Category: faces / natural / texture / painting / architectural / synthetic
# Distance range: rough expected ResNet50 cosine dist to ImageNet centroid

CANDIDATES = [
    # ── FACES (non-Wilfredor) ─────────────────────────────────────────────────
    # Wilfredor's 3 already exist. Need diverse portraits.
    ("face_red_hair",      "Red_Hair_Woman.jpg",                                  "faces",        "0.70-0.85"),
    ("face_algerian",      "Algerian_man_with_turban.jpg",                        "faces",        "0.70-0.85"),
    # Vermeer's Girl with Pearl Earring is a face-like painting — treat separately
    # Additional CC0 faces via well-known filenames:
    ("face_pexels_woman",  "Pexels_woman_1239291.jpg",                            "faces",        "0.70-0.85"),

    # ── TEXTURES ──────────────────────────────────────────────────────────────
    # Existing: wood_grain(0.948), dirt_soil(0.910), grass_meadow(0.914)
    # Need: variety at 0.88–0.96
    ("texture_sand",       "Gfp-grainy-sand-texture.jpg",                         "texture",      "0.88-0.96"),
    ("texture_brick",      "Brick_wall_close-up_view.jpg",                        "texture",      "0.88-0.96"),
    ("texture_stone",      "Square_stone_brick_Texture.jpg",                      "texture",      "0.88-0.96"),
    ("texture_cement",     "Cement_texture.jpg",                                  "texture",      "0.88-0.96"),
    ("texture_asphalt",    "Asphalt_close_800x600.jpg",                           "texture",      "0.88-0.96"),
    ("texture_bark",       "Gfp-tree-bark.jpg",                                   "texture",      "0.88-0.96"),

    # ── NATURAL SCENES ────────────────────────────────────────────────────────
    # Existing: boardwalk(0.786), frog_on_log(0.866), nature_landscape(0.802),
    #           nat_landscape2(0.804), nat_landscape3(0.818)
    # Need: wildlife, snow, ocean, forest close-up
    ("natural_snow_mountain", "Snowy_mountains_in_Valloire_(Unsplash).jpg",       "natural",      "0.79-0.88"),
    ("natural_coral_reef",    "Underwater_photo_of_coral_reef.jpg",               "natural",      "0.79-0.88"),
    ("natural_caribou",       "Barren_ground_caribou_grazing_with_autumn_foliage_in_background.jpg",
                                                                                  "natural",      "0.79-0.88"),
    ("natural_fir_snow",      "Fir_trees_in_the_cold_snow_(Unsplash).jpg",        "natural",      "0.79-0.88"),

    # ── PAINTINGS (public domain — key for filling 0.87-0.91 gap) ────────────
    ("paint_vermeer_pearl", "Johannes_Vermeer_-_Girl_with_a_Pearl_Earring_-_670_-_Mauritshuis.jpg",
                                                                                  "painting",     "0.80-0.92"),
    ("paint_vermeer_milk",  "Johannes_Vermeer_-_Het_melkmeisje_-_Google_Art_Project.jpg",
                                                                                  "painting",     "0.80-0.92"),
    ("paint_rembrandt_self","Rembrandt_van_Rijn_-_Self-Portrait_-_Google_Art_Project.jpg",
                                                                                  "painting",     "0.80-0.92"),
    ("paint_monet_magpie",  "Claude_Monet_-_The_Magpie_-_Google_Art_Project.jpg","painting",     "0.82-0.93"),
    ("paint_monet_lilies",  "Claude_Monet_-_Water_Lilies_-_Google_Art_Project_(462013).jpg",
                                                                                  "painting",     "0.82-0.93"),
    ("paint_renoir_galette","Auguste_Renoir_-_Dance_at_Le_Moulin_de_la_Galette_-_Google_Art_Project.jpg",
                                                                                  "painting",     "0.82-0.93"),
    ("paint_renoir_piano",  "Auguste_Renoir_-_Young_Girls_at_the_Piano_-_Google_Art_Project.jpg",
                                                                                  "painting",     "0.80-0.92"),
    ("paint_vangogh_starry","Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",  "painting",     "0.85-0.95"),

    # ── ARCHITECTURAL / URBAN ─────────────────────────────────────────────────
    ("arch_street_sf",     "Street_View_urban_street_(Unsplash).jpg",             "architectural","0.82-0.92"),
    ("arch_nyc_street",    "People_walking_across_urban_street_(Unsplash).jpg",   "architectural","0.82-0.92"),
    ("arch_bologna",       "Metropolitan_City_of_Bologna,_Italy_(Unsplash).jpg",  "architectural","0.82-0.92"),
    ("arch_interior",      "Living_room_(Unsplash).jpg",                          "architectural","0.82-0.92"),
    ("arch_building",      "Moore_Hall,_Western_Carolina_University,_Cullowhee,_NC_(45725670745).jpg",
                                                                                  "architectural","0.82-0.92"),
]

# ─── existing images (already downloaded) ─────────────────────────────────────
EXISTING = [
    # (local_label, path_relative_to_RES, category, prior_distance)
    ("boardwalk",       "natural/boardwalk_nature.jpg",     "natural",     0.7858),
    ("frog_on_log",     "natural/frog_on_log.jpg",          "natural",     0.8663),
    ("nature_land",     "natural/nature_landscape.jpg",     "natural",     0.8023),
    ("nat_landscape2",  "new_v2/nat_landscape2.jpeg",       "natural",     0.8041),
    ("nat_landscape3",  "new_v2/nat_landscape3.jpeg",       "natural",     0.8177),
    ("boy_face",        "faces/boy_face_venezuela.jpg",     "faces",       0.8018),
    ("girl_sad_face",   "faces/girl_sad_face.jpg",          "faces",       0.7365),
    ("wayuu_woman",     "faces/wayuu_woman.jpg",            "faces",       0.7203),
    ("wood_grain",      "texture/wood_grain.png",           "texture",     0.9483),
    ("dirt_soil",       "texture/dirt_soil.png",            "texture",     0.9099),
    ("grass_meadow",    "texture/grass_meadow.png",         "texture",     0.9140),
    # synthetics kept for sanity only (excluded from main analysis)
    ("soft_blobs",      "synthetic/soft_blobs.png",         "synthetic",   0.8890),
    ("noise_gauss50",   "synthetic/noise_gauss50.png",      "synthetic",   0.9154),
    ("checker32",       "synthetic/checker32.png",          "synthetic",   0.7891),
]

# ─── helpers ─────────────────────────────────────────────────────────────────

def _wikimedia_url(filename: str, width: int = 800) -> str | None:
    """Query Wikimedia Commons API for the thumbnail URL of a file."""
    api = "https://commons.wikimedia.org/w/api.php"
    params = urllib.parse.urlencode({
        "action": "query",
        "titles": f"File:{filename}",
        "prop": "imageinfo",
        "iiprop": "url|thumburl",
        "iiurlwidth": str(width),
        "format": "json",
    })
    url = f"{api}?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "provenanssi-research/1.0 (research project)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if page.get("ns") == -1 or "missing" in page:
                return None
            iinfo = page.get("imageinfo", [{}])
            if iinfo:
                return iinfo[0].get("thumburl") or iinfo[0].get("url")
    except Exception:
        return None
    return None


def _download(url: str, dest: Path, label: str) -> bool:
    """Download a file with User-Agent header. Returns True on success."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "provenanssi-research/1.0 (research project)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"  FAIL {label}: {e}")
        return False


def _compute_distance(path: Path, model, centroid: np.ndarray, device: str) -> float | None:
    """Compute ResNet50 cosine distance to centroid."""
    try:
        rgb = _load_rgb_224(str(path))
        feat = _resnet_features(rgb, model, device)
        feat_n = feat / (np.linalg.norm(feat) + 1e-12)
        return float(1.0 - float(np.dot(feat_n, centroid)))
    except Exception as e:
        print(f"  distance FAIL {path.name}: {e}")
        return None


def _imagenet_centroid(model, device: str) -> np.ndarray:
    """Compute L2-normalized centroid of 16 ILSVRC2012 reference images."""
    ref_files = sorted(REF_DIR.glob("*.png"))[:16]
    if len(ref_files) < 16:
        ref_files = sorted(REF_DIR.glob("*.*"))[:16]
    feats = []
    for f in ref_files:
        try:
            rgb  = _load_rgb_224(str(f))
            feat = _resnet_features(rgb, model, device)
            feats.append(feat / (np.linalg.norm(feat) + 1e-12))
        except Exception:
            pass
    centroid = np.mean(feats, axis=0)
    return centroid / (np.linalg.norm(centroid) + 1e-12)


def _coverage_stats(distances: list[float]) -> dict:
    """Summary stats for coverage check."""
    d = np.array(distances)
    return {
        "n":    len(d),
        "min":  float(d.min()),
        "max":  float(d.max()),
        "mean": float(d.mean()),
        "std":  float(d.std()),
        "q10":  float(np.percentile(d, 10)),
        "q25":  float(np.percentile(d, 25)),
        "q50":  float(np.percentile(d, 50)),
        "q75":  float(np.percentile(d, 75)),
        "q90":  float(np.percentile(d, 90)),
    }


def _is_bimodal(distances: list[float]) -> bool:
    """Rough bimodality check: gap > 2× IQR in the middle of the range."""
    d = sorted(distances)
    n = len(d)
    if n < 6:
        return False
    gaps = [d[i+1] - d[i] for i in range(n-1)]
    iqr  = np.percentile(d, 75) - np.percentile(d, 25)
    q25  = np.percentile(d, 25)
    q75  = np.percentile(d, 75)
    # check for large gaps in the middle quartiles only
    for i in range(n-1):
        if q25 <= d[i] <= q75 and gaps[i] > 1.5 * iqr:
            return True
    return False


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    device = "cpu"   # ResNet50 only, no diffusion

    out("=" * 72)
    out("  PHASE 1 — SAMPLE ASSEMBLY + DISTANCE COVERAGE CHECK")
    out(f"  Date: 2026-06-23  Target: ~40 images, continuous distance axis")
    out("=" * 72)
    out()

    # ── 1. Load ResNet50 + reference centroid ────────────────────────────────
    out("Loading ResNet50 (IMAGENET1K_V2) …")
    model = _load_resnet50(RESNET_W, device)
    out("Computing ImageNet reference centroid (16 images) …")
    centroid = _imagenet_centroid(model, device)
    out(f"  Centroid computed (L2-normalized, dim=2048)")
    out()

    # ── 2. Download new candidate images ─────────────────────────────────────
    out("─" * 72)
    out("  STEP 1: Download candidate images via Wikimedia API")
    out("─" * 72)
    out()

    downloaded: list[tuple[str, Path, str]] = []   # (label, path, category)
    failed: list[str] = []

    for label, filename, category, _ in CANDIDATES:
        dest_ext = Path(filename).suffix.lower()
        dest = OUT_DIR / f"{label}{dest_ext}"

        if dest.exists():
            out(f"  EXISTS  {label}")
            downloaded.append((label, dest, category))
            time.sleep(0.1)
            continue

        out(f"  QUERY   {label}  ({filename[:60]})")
        url = _wikimedia_url(filename, width=1024)
        if url is None:
            out(f"  FAIL    {label}  — not found on Wikimedia Commons")
            failed.append(label)
            time.sleep(1.0)
            continue

        out(f"  FETCH   {label}  → {url[:80]}…")
        ok = _download(url, dest, label)
        if ok:
            size_kb = dest.stat().st_size // 1024
            out(f"  OK      {label}  ({size_kb} KB)")
            downloaded.append((label, dest, category))
        else:
            failed.append(label)
        time.sleep(3.0)   # respectful delay

    out()
    out(f"  Downloaded: {len(downloaded)} / {len(CANDIDATES)}  Failed: {len(failed)}")
    if failed:
        out(f"  Failed: {', '.join(failed)}")
    out()

    # ── 3. Compute distances for all images ──────────────────────────────────
    out("─" * 72)
    out("  STEP 2: ResNet50 distances — existing + new images")
    out("─" * 72)
    out()

    all_rows: list[dict] = []

    # Existing images
    out("  [Existing images]")
    for label, rel_path, category, prior_dist in EXISTING:
        path = RES / rel_path
        if not path.exists():
            out(f"  SKIP  {label}  (file missing)")
            continue
        d = _compute_distance(path, model, centroid, device)
        if d is None:
            continue
        out(f"  {label:<22}  {category:<14}  d={d:.4f}  (prior={prior_dist:.4f})")
        all_rows.append({"label": label, "category": category, "dist": d,
                         "path": str(path), "is_new": False})

    out()
    out("  [New images]")
    for label, path, category in downloaded:
        d = _compute_distance(path, model, centroid, device)
        if d is None:
            continue
        out(f"  {label:<22}  {category:<14}  d={d:.4f}")
        all_rows.append({"label": label, "category": category, "dist": d,
                         "path": str(path), "is_new": True})

    out()
    out(f"  Total measured: {len(all_rows)} images")
    out()

    # ── 4. Distribution table (sorted by distance) ───────────────────────────
    out("─" * 72)
    out("  STEP 3: Distribution table (sorted by distance)")
    out("─" * 72)
    out()
    sorted_rows = sorted(all_rows, key=lambda r: r["dist"])
    out(f"  {'#':>2}  {'label':<22}  {'category':<14}  {'dist':>6}  new?")
    out("  " + "─" * 56)
    for i, r in enumerate(sorted_rows, 1):
        new_flag = "NEW" if r["is_new"] else "   "
        out(f"  {i:>2}  {r['label']:<22}  {r['category']:<14}  {r['dist']:>6.4f}  {new_flag}")
    out()

    # ── 5. Coverage analysis ─────────────────────────────────────────────────
    out("─" * 72)
    out("  STEP 4: Coverage analysis — non-synthetic images only")
    out("─" * 72)
    out()

    non_syn = [r for r in all_rows if r["category"] != "synthetic"]
    non_syn_dists = [r["dist"] for r in non_syn]

    by_cat: dict[str, list[float]] = {}
    for r in non_syn:
        by_cat.setdefault(r["category"], []).append(r["dist"])

    s = _coverage_stats(non_syn_dists) if non_syn_dists else {}
    if s:
        out(f"  Non-synthetic n = {s['n']}")
        out(f"  Range:  [{s['min']:.4f}, {s['max']:.4f}]   span = {s['max']-s['min']:.4f}")
        out(f"  Mean:   {s['mean']:.4f}   Std: {s['std']:.4f}")
        out(f"  Q10–Q90: [{s['q10']:.4f}, {s['q90']:.4f}]")
        out()
        out("  Per-category ranges:")
        for cat, dists in sorted(by_cat.items()):
            out(f"    {cat:<14}  n={len(dists):>2}  [{min(dists):.4f}, {max(dists):.4f}]  mean={np.mean(dists):.4f}")
        out()

        # Gap detection: largest gap between consecutive sorted distances
        d_sorted = sorted(non_syn_dists)
        gaps = [(d_sorted[i+1] - d_sorted[i], d_sorted[i], d_sorted[i+1])
                for i in range(len(d_sorted)-1)]
        top_gaps = sorted(gaps, reverse=True)[:3]
        out("  Largest gaps (top 3):")
        for gap, lo, hi in top_gaps:
            # Find which categories span this gap
            cats_lo = [r["category"] for r in non_syn if abs(r["dist"] - lo) < 0.002]
            cats_hi = [r["category"] for r in non_syn if abs(r["dist"] - hi) < 0.002]
            out(f"    gap={gap:.4f}  between {lo:.4f}({cats_lo}) and {hi:.4f}({cats_hi})")
        out()

        bimodal = _is_bimodal(non_syn_dists)
        out(f"  Bimodality check: {'BIMODAL — fix composition' if bimodal else 'OK (no large mid-range gap detected)'}")

    out()

    # ── 6. Phase 2 budget ────────────────────────────────────────────────────
    out("─" * 72)
    out("  STEP 5: Phase 2 runtime budget")
    out("─" * 72)
    out()

    n_new_images = len([r for r in all_rows if r["is_new"]])
    n_existing_calibrated = 11  # existing non-synthetic with N≥48 calibrations
    n_total_non_syn = len(non_syn)

    # Phase 2 requires per image:
    # - N=48 calibration (if not already done): 48 passes × 1.1s = ~53s
    # - 5 noise-floor windows of N=12: 5×12×1.1s = 66s
    # - 1 coherence window of N=12: 12×1.1s = 13s
    # Total per image: ~132s (53+66+13)  if fresh
    # For already-calibrated: just noise-floor + coherence = 79s

    n_fresh = n_new_images  # new images: need full calibration
    n_recal  = 0            # may need re-calibration for some existing ones
    n_nf_only = n_total_non_syn - n_fresh  # only need noise-floor + coherence

    t_fresh   = n_fresh   * 132   # seconds
    t_nf_only = n_nf_only * 79
    t_total   = t_fresh + t_nf_only

    out(f"  Non-synthetic images total: {n_total_non_syn}")
    out(f"  Fresh calibrations needed (new images): {n_fresh}")
    out(f"  Noise-floor + coherence only (existing N≥48): {n_nf_only}")
    out()
    out(f"  Time per fresh image (N=48 cal + 5×NF×N=12 + 1×coh×N=12):")
    out(f"    N=48 cal:         {48*1.1:.0f}s")
    out(f"    5×NF windows ×12: {5*12*1.1:.0f}s")
    out(f"    1 coherence ×12:  {12*1.1:.0f}s")
    out(f"    Total per fresh:  ~{132}s")
    out()
    out(f"  Time per existing image (NF + coherence only):  ~79s")
    out()
    out(f"  Estimated total: {n_fresh}×132s + {n_nf_only}×79s = {t_total}s")
    out(f"                   = {t_total/60:.0f} min ({t_total/3600:.1f} h)")
    out()

    # Proposed N and trade-offs
    out("  PROPOSED PROTOCOL:")
    out("  ─────────────────")
    out("  N=48 for calibration slope (same as prior results — comparable).")
    out("  5 independent NF windows at N=12 (same as Update 6 — directly comparable).")
    out("  1 coherence run at N=12 (same as Update 7).")
    out()
    out("  Why N=48 for slope, not higher:")
    out("  · At N=48, slope SE ≈ noise_floor/√5 ≈ 0.013–0.023 (for faces group),")
    out("    which is ~10–20% of the slope signal. Adequate for correlations at n≥20.")
    out("  · N=96 halves SE but doubles runtime. Marginal gain at n=30+ images.")
    out("  · N=48 matches the prior CERTIFIED result — keeps everything comparable.")
    out()
    out(f"  Statistical power at n=30 non-synthetic images:")
    out(f"  · For r=0.5 (medium effect): power≈0.82 at α=0.05 (two-tailed)")
    out(f"  · For r=0.4 (smaller effect): power≈0.60 — marginal")
    out(f"  · Minimum detectable r at 80% power, α=0.05: r≈0.48 (n=30)")
    out()
    out("  Tradeoff: if budget allows, ≥40 non-synthetic would give power≈0.89 at r=0.45.")
    out("  Recommendation: proceed with all available non-synthetic images.")
    out()

    # ── 7. Pre-registered analysis plan ──────────────────────────────────────
    out("─" * 72)
    out("  STEP 6: Pre-registered analysis plan (written BEFORE seeing results)")
    out("─" * 72)
    out()
    out("  [Written to pre_registration.md]")

    pre_reg_text = f"""# Pre-registered Analysis Plan
# Phase 2 — Large-sample resolution of three power-limited threads
# Written: 2026-06-23  BEFORE any Phase 2 data is collected.
# This plan is locked before Phase 2 runs. No post-hoc changes.

## Context

Three threads hit the same statistical-power wall at n=5–11:
1. Distance→slope (Update 4): r(ResNet50_dist, slope), n=11 non-synthetic in 3 clusters
2. Slope→noise-floor α (Update 6): power law fit, n=5 images
3. Coherence test (Update 7): r(rho_nn, slope), n=5 images; girl_sad_face anomaly

One larger balanced sample resolves all three.

## Sample

Target: all non-synthetic images available after Phase 1 assembly.
Exclusions:
- `soft_blobs`, `noise_gauss50`, `checker32`, `hard_shapes` (synthetic/generated)
- Any image where N=48 calibration fails to converge (r < 0.90)
- Any image where null_frac_gt < 0.001 (degenerate null space)

Primary set: all remaining non-synthetic images (expected n=25–35).

## Thread 1: distance→slope continuous relationship

### Measurement
- dist_i = ResNet50 cosine distance to 16-image ILSVRC2012 centroid (same metric as Update 4)
- slope_i = OLS calibration slope at N=48 (same protocol as prior measurements)

### Tests
A. Pearson r(dist, slope) + Fisher-z 95% CI + n — ALL non-synthetic images
B. Pearson r(dist, slope) + CI — EXCLUDING paintings (to test sensitivity to new category)
C. Partial r(dist, slope | null_frac_gt) + CI — controls for null energy confound
D. Spearman rho(dist, slope) + p — rank-based (non-parametric, same as Update 4)

### Decision rule
- (a) POSITIVE if: r < 0 (negative expected — higher dist → lower slope is the hypothesis
  based on Update 4), AND CI excludes 0 in tests A AND C.
  NOTE: the hypothesis is that slope is HIGHER for images CLOSER to ImageNet (faces),
  so we expect r(dist, slope) < 0. Recode if needed.
- (a-weak) if: r in predicted direction, CI includes 0, but p < 0.10 (one test)
- (b) NOT SUPPORTED if: r near zero or in wrong direction, CI includes 0

Pre-stated expected direction: r(dist, slope) < 0 (nearer ImageNet → higher slope).
EXCEPTION if paintings turn out near ImageNet but low slope — recheck mechanism.

### Within-group test (new: n≥6 per group)
E. For each group with n≥6: report within-group r(dist, slope)
   Decision: if within-group r is near zero for ALL groups but between-group r is large
   → it's a group-level effect, not continuous. Report explicitly.

## Thread 2: slope→noise-floor α re-estimation

### Measurement
- slope_i at N=48 (as above)
- noise_floor_std_i = std of β̂ across 5 independent N=12 windows (same protocol as Update 6)

### Test
Power law fit: log(noise_floor_std) = log(a) + α·log(slope), OLS
Fit on: non-synthetic images EXCLUDING soft_blobs and any image with slope < 0.2

### Decision rule (α confidence interval)
- Definitive: t-CI for α excludes 0 AND is entirely below 1.0 (α<1 = sub-proportional)
- Provisional: CI excludes 0 but includes 1.0
- Null: CI includes 0

Pre-stated H0: α=0 (OLS scale-invariance under iid pixels)
Pre-stated H1: α>0 (spatial correlation of reconstruction errors)
Pre-stated expected value from Update 6: α≈0.35–0.50 (sub-proportional)

## Thread 3: coherence→slope, coherence→noise-floor mediation

### Measurement
- rho_nn_i = mean nearest-neighbor ACF of normalized null-space deviations (same as Update 7)
  N=12 ensemble per image, 1 run (no repeats for coherence — the ACF is stable per run)

### Tests
F. r(rho_nn, slope) + CI + n
G. r(rho_nn, noise_floor_std) + CI + n
H. Partial r(slope, noise_floor_std | rho_nn) — mediation test

### Decision rule
For F: (a) if r > 0 and CI excludes 0; (a-weak) if r > 0.2 and CI includes 0; (b) if r ≤ 0.
For mediation H: confirmed if |partial_r| < 0.7 × |direct_r(slope, nf_std)|

### Girl_sad_face anomaly
Update 7: girl_sad_face had rho_nn=0.154 despite slope=2.80 (lower than wayuu_woman 0.245,
slope=1.58). With n≥6 faces, test:
- Within-faces: r(rho_nn, slope) + CI
- Decision: if CI still includes 0 within faces → anomaly is noise (n=5 too small)
             if within-faces r < 0 → coherence and slope are decoupled within faces

## Reporting format

All results reported as:
  r = X.XXX  CI [±XX.XX%, df=n-2, t-corrected]  n=XX  verdict

No selective reporting: ALL pre-stated tests reported, even nulls.
A definitive NULL (CI clearly includes 0 with n≥20) is a full result.

## Anti-fishing clause

This plan was written before Phase 2 data collection. If the data suggest
additional interesting tests, those are flagged as POST-HOC EXPLORATORY and
not included in the confirmatory results table. The four pre-stated tests
above are the only ones that count for the confirmatory verdict.

## Statistical thresholds

α = 0.05 (two-tailed) for all confirmatory tests.
CI: Fisher-z with t-critical value at df=n-2 (t-corrected, not z=2).
Partial r: computed from OLS residuals (standard partial correlation).
No multiple-comparison correction (four pre-stated tests on the same dataset
are correlated, and the primary question is pattern, not individual p-values).
"""

    PRE_REG.write_text(pre_reg_text)
    out()
    out(f"  Pre-registration written to: {PRE_REG}")
    out()

    # ── 8. Summary ────────────────────────────────────────────────────────────
    out("=" * 72)
    out("  PHASE 1 SUMMARY — FOR HUMAN REVIEW")
    out("=" * 72)
    out()
    out(f"  Images assembled:      {len(all_rows)} total ({len(non_syn)} non-synthetic)")
    out(f"  New images downloaded: {len([r for r in all_rows if r['is_new']])}")
    out(f"  Download failures:     {len(failed)}")
    out()
    if non_syn_dists:
        bimodal = _is_bimodal(non_syn_dists)
        out(f"  Distance range (non-syn): [{min(non_syn_dists):.4f}, {max(non_syn_dists):.4f}]")
        out(f"  Bimodality:               {'YES — needs more mid-range images' if bimodal else 'NO — axis covered'}")
    out()
    out(f"  Phase 2 estimated runtime: {t_total}s ({t_total/60:.0f} min, {t_total/3600:.1f}h)")
    out(f"  Proposed N: 48 calibration + 5×NF×12 + 1×coherence×12 per image")
    out()
    out("  Pre-registered analysis plan: WRITTEN (eval/pre_registration.md)")
    out()
    out("  AWAITING APPROVAL before Phase 2.")
    out()

    # Save report
    OUT_REPORT.write_text("\n".join(lines))
    print(f"\nFull report written to {OUT_REPORT}")


if __name__ == "__main__":
    main()
