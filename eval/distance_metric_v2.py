#!/usr/bin/env python3
"""
eval/distance_metric_v2.py — Semantic distance metric + slope correlation, attempt 2.

THREAD: FINDINGS.md §11 item 1 (continued from distance_metric.py null result).

PREVIOUS FAILURE (attempt 1):
  VQ-autoencoder cosine distance placed synthetic images *closer* to the ImageNet
  centroid than natural images — the opposite of the intuitive ordering. The metric
  was measuring compressibility/encodability, not semantic distribution membership.

THIS ATTEMPT FIXES TWO PROBLEMS:
  1. METRIC: ResNet50 penultimate-layer features (2048-dim) — an ImageNet-trained
     CLASSIFIER, not a reconstruction encoder. Semantic content drives the features.
  2. SAMPLE SIZE: grow from n=13 to n≈30–35 by adding CC0/PD real images and
     programmatically generated images covering the mid-range of the distance axis.

METRIC (precisely named):
  Cosine distance in ResNet50 (IMAGENET1K_V2 weights, PyTorch official) penultimate-
  layer feature space (2048-dim, global-average-pooled after layer4), to the
  L2-normalized centroid of 16 ILSVRC2012 validation image encodings.

  Preprocessing (all images, both reference and eval):
    RGB, center-crop 224×224, ImageNet normalisation
    (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).

  LIMITATIONS:
    · n_ref=16: centroid is noisy estimate of the ImageNet centroid
    · Cosine distance to centroid assumes the centroid is representative;
      with only 16 reference images, this is a rough proxy
    · ResNet50 features are optimised for classification, not distribution
      distance; relative distances between images may not be meaningful
    · Grayscale calibration slope is predicted from RGB features — correct
      in principle (distribution shift exists in both modalities) but the
      mapping is not guaranteed

SANITY CHECK (gates the analysis):
  Before computing any correlation, confirm that the ResNet50 distance ordering
  is sane: linear_gradient distance > boardwalk_nature distance, i.e. an obvious
  synthetic is measured as FARTHER from ImageNet than an obvious natural photo.
  If this fails, the metric is broken again — report and stop.

SAMPLE: ~30–35 images (n reported at run time).
  Existing 13 + new downloads (~10 CC0/PD) + programmatic synthetics (~8).
  Download failures are handled gracefully (skip and note).
  Programmatic images: noise variants and patterns at different scales/distributions.
  The calibration for each image uses N=48 (or reuses prior N≥48 result).

CONFOUND + EXCLUSION:
  · Without-synthetics correlation reported separately (the key test)
  · Partial correlation controlling null_frac_gt
  · CI width for every r reported; n stated alongside
"""

import os, sys, time, math, hashlib, textwrap
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from operators.bicubic import BicubicDownsample
from engine.resshift import ResShiftEngine
from layer.decompose import rectify, CONSISTENCY_EPS
from layer.calibrate import calibrate

# ── Config ─────────────────────────────────────────────────────────────────────

SCALE   = 4
H, W    = 256, 256
N_BINS  = 10
MIN_STD = 1e-6
R3_TOL  = CONSISTENCY_EPS
N_NEW   = 48

RNG_SEED = 42  # for generated synthetic images

REPO     = Path(__file__).parent.parent
RES      = Path(__file__).parent / "research_sources"
REF_DIR  = REPO / "vendor" / "ResShift" / "testdata" / "Bicubicx4" / "gt"
OUT_FILE = Path(__file__).parent / "distance_metric_v2_results.txt"
SOURCES  = RES / "SOURCES_LICENCE.md"

RESNET_WEIGHTS = REPO / "weights" / "resnet50-11ad3fa6.pth"

# ImageNet normalisation for ResNet50
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Minimal ResNet50 (no torchvision dependency) ────────────────────────────────

class _Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes * 4)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _ResNet50(nn.Module):
    """ResNet50 with .features() method returning 2048-dim penultimate vector.
    Layer and parameter names match the official PyTorch IMAGENET1K_V2 weights exactly.
    """
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1  = self._make_layer(64,  64,  3)
        self.layer2  = self._make_layer(256, 128, 4, stride=2)
        self.layer3  = self._make_layer(512, 256, 6, stride=2)
        self.layer4  = self._make_layer(1024, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(2048, 1000)

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        layers = [_Bottleneck(inplanes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(_Bottleneck(planes * 4, planes))
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 2048-dim penultimate feature vector (before fc)."""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.avgpool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


def _load_resnet50(weights_path: Path, device: str) -> _ResNet50:
    model = _ResNet50()
    ckpt  = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    return model.to(device).eval()


# ── Image loading / preprocessing ──────────────────────────────────────────────

def _load_rgb_224(path: str) -> np.ndarray:
    """Load image → RGB float32 [0,1] at 224×224 (center crop after resize)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    short = 256
    if w < h:
        img = img.resize((short, int(round(h * short / w))), Image.LANCZOS)
    else:
        img = img.resize((int(round(w * short / h)), short), Image.LANCZOS)
    w2, h2 = img.size
    l = (w2 - 224) // 2; t = (h2 - 224) // 2
    img = img.crop((l, t, l + 224, t + 224))
    return np.array(img, dtype=np.float32) / 255.0  # (224, 224, 3)


def _load_gray_256(path: str) -> np.ndarray:
    """Load image → BT.601 grayscale float64 [0,1] at 256×256 (for calibration)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w < h:
        img = img.resize((W, int(round(h * W / w))), Image.LANCZOS)
    else:
        img = img.resize((int(round(w * H / h)), H), Image.LANCZOS)
    w2, h2 = img.size
    l = (w2 - W) // 2; t = (h2 - H) // 2
    img = img.crop((l, t, l + W, t + H))
    rgb = np.array(img, dtype=np.float64) / 255.0
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _resnet_features(rgb: np.ndarray, model: _ResNet50, device: str) -> np.ndarray:
    """Extract 2048-dim ResNet50 features from RGB [0,1] (224×224×3) numpy array."""
    x = (rgb - IMAGENET_MEAN) / IMAGENET_STD    # (224, 224, 3)
    x = torch.from_numpy(x).permute(2, 0, 1).float()  # (3, 224, 224)
    x = x.unsqueeze(0).to(device)               # (1, 3, 224, 224)
    with torch.no_grad():
        feat = model.features(x)                # (1, 2048)
    return feat.squeeze(0).cpu().numpy()         # (2048,)


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (na * nb))


def _null_frac_gt(gray: np.ndarray, op) -> float:
    rp = op.pinv(op.forward(gray))
    null = gray - rp
    return float(np.sum(null**2) / (np.sum(gray**2) + 1e-12))


def _run_calibration(gray: np.ndarray, op, engine, n: int) -> dict:
    y = op.forward(gray)
    x_hats = engine.ensemble(y, n)
    results = [rectify(xh, y, op) for xh in x_hats]
    for r in results:
        assert r.residual <= R3_TOL, f"R3 violated: {r.residual:.2e}"
    x_outs = [r.x_out for r in results]
    cal = calibrate(x_outs, gray, n_bins=N_BINS, min_predicted_std=MIN_STD)
    return {"slope": cal.slope, "r": cal.pearson_r, "n": n}


# ── Statistics ─────────────────────────────────────────────────────────────────

def _pearson_ci(r: float, n: int, alpha: float = 0.05) -> tuple:
    if abs(r) >= 1.0 or n <= 3:
        return float("nan"), float("nan")
    z  = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    zc = scipy_stats.norm.ppf(1 - alpha / 2)
    return math.tanh(z - zc * se), math.tanh(z + zc * se)


def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    rxy = np.corrcoef(x, y)[0, 1]
    rxz = np.corrcoef(x, z)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]
    d   = math.sqrt((1 - rxz**2) * (1 - ryz**2))
    return float((rxy - rxz * ryz) / d) if d > 1e-12 else float("nan")


def _corr_block(label: str, dists: np.ndarray, slopes: np.ndarray,
                nulls: np.ndarray, lines: list) -> None:
    n = len(dists)
    r_p, p_p = scipy_stats.pearsonr(dists, slopes)
    lo, hi   = _pearson_ci(r_p, n)
    r_s, p_s = scipy_stats.spearmanr(dists, slopes)
    r_nu     = np.corrcoef(nulls, slopes)[0, 1]
    r_pa     = _partial_corr(dists, slopes, nulls)
    lo_pa, hi_pa = _pearson_ci(r_pa, n - 1)  # n-1 for partial with 1 control (df=n-4)
    lines += [
        f"  [{label}]  n={n}",
        f"    Pearson r(dist,slope)      = {r_p:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  p={p_p:.3f}",
        f"    Spearman rho(dist,slope)   = {r_s:+.3f}  p={p_s:.3f}",
        f"    Pearson r(null,slope)      = {r_nu:+.3f}  (confounder)",
        f"    Partial r(dist,slope|null) = {r_pa:+.3f}  95% CI [{lo_pa:+.3f},{hi_pa:+.3f}]  (approx)",
    ]


# ── Programmatic image generation ──────────────────────────────────────────────

def _gen_gaussian_noise(std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0.5, std, (H, W)), 0, 1).astype(np.float32)


def _gen_pink_noise(seed: int) -> np.ndarray:
    """1/f power spectrum noise (pink noise) — intermediate between Gaussian and natural."""
    rng = np.random.default_rng(seed)
    n = max(H, W)
    f = np.fft.rfftfreq(n)
    # Avoid divide-by-zero at DC
    psd = np.where(f == 0, 0, f ** (-1.0))
    # 2D version: build amplitude in frequency domain
    fy, fx = np.fft.fftfreq(H), np.fft.rfftfreq(W)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    freq_2d = np.sqrt(FY**2 + FX**2)
    amplitude = np.where(freq_2d == 0, 0.0, freq_2d ** (-1.0))
    phase = rng.uniform(0, 2 * np.pi, amplitude.shape)
    spectrum = amplitude * np.exp(1j * phase)
    img = np.fft.irfft2(spectrum, s=(H, W))
    img -= img.min(); img /= (img.max() - img.min() + 1e-12)
    return img.astype(np.float32)


def _gen_checkerboard(cell_size: int) -> np.ndarray:
    y, x = np.mgrid[:H, :W]
    return ((y // cell_size + x // cell_size) % 2).astype(np.float32)


def _gen_stripes(freq: float, angle_deg: float) -> np.ndarray:
    y, x = np.mgrid[:H, :W]
    a = np.radians(angle_deg)
    s = 0.5 + 0.5 * np.sin(2 * np.pi * freq * (x * np.cos(a) + y * np.sin(a)) / W)
    return s.astype(np.float32)


def _gray_to_rgb_224(gray: np.ndarray) -> np.ndarray:
    """Convert (H,W) float32 [0,1] → (224,224,3) float32 via resize + channel replicate."""
    img = Image.fromarray((gray * 255).clip(0, 255).astype(np.uint8), mode="L")
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.stack([arr, arr, arr], axis=-1)  # (224, 224, 3)


def _gray_to_rgb_256(gray: np.ndarray) -> np.ndarray:
    """Convert (H,W) float32 → save as 256×256 RGB PNG (for calibration loading)."""
    return gray  # calibration uses grayscale directly


# ── CC0/PD download list ────────────────────────────────────────────────────────
# Only images confirmed PD or CC0 via Wikimedia API.

DOWNLOAD_IMAGES = [
    # Natural landscapes (CC0 / Public Domain)
    ("nat_landscape2",  "natural",
     "https://upload.wikimedia.org/wikipedia/commons/c/c9/Nature_Landscape_%28248036121%29.jpeg",
     "CC0 1.0", "Unknown photographer",
     "https://commons.wikimedia.org/wiki/File:Nature_Landscape_(248036121).jpeg"),
    ("nat_landscape3",  "natural",
     "https://upload.wikimedia.org/wikipedia/commons/8/83/Nature_Landscape_%28248036053%29.jpeg",
     "CC0 1.0", "Unknown photographer",
     "https://commons.wikimedia.org/wiki/File:Nature_Landscape_(248036053).jpeg"),
    ("wooded_pond",     "natural",
     "https://upload.wikimedia.org/wikipedia/commons/5/52/Gfp-Sangchris-Lake-State-Park-wooded-pond.jpg",
     "CC-PD", "Yinan Chen (Goodfreephotos_com)",
     "https://commons.wikimedia.org/wiki/File:Gfp-Sangchris-Lake-State-Park-wooded-pond.jpg"),
    ("rain_lake",       "natural",
     "https://upload.wikimedia.org/wikipedia/commons/0/02/Gfp-Rain-on-lake-michigan-at-newport-state-park-wisconsin.jpg",
     "CC-PD", "Yinan Chen (Goodfreephotos_com)",
     "https://commons.wikimedia.org/wiki/File:Gfp-Rain-on-lake-michigan-at-newport-state-park-wisconsin.jpg"),
    ("mountain_trail",  "natural",
     "https://upload.wikimedia.org/wikipedia/commons/c/c4/Gfp-Minnesota-frontenac-state-park-mountaintop-trail.jpg",
     "CC-PD", "Yinan Chen (Goodfreephotos_com)",
     "https://commons.wikimedia.org/wiki/File:Gfp-Minnesota-frontenac-state-park-mountaintop-trail.jpg"),
    ("hilltop_valley",  "natural",
     "https://upload.wikimedia.org/wikipedia/commons/1/13/Gfp-Minnesota-beaver-creek-valley-view-of-hill-top.jpg",
     "CC-PD", "Yinan Chen (Goodfreephotos_com)",
     "https://commons.wikimedia.org/wiki/File:Gfp-Minnesota-beaver-creek-valley-view-of-hill-top.jpg"),
    ("harbor_dusk",     "natural",
     "https://upload.wikimedia.org/wikipedia/commons/d/da/Gfp-Wisconsin-algoma-dusk-in-the-harbor.jpg",
     "CC-PD", "Yinan Chen (Goodfreephotos_com)",
     "https://commons.wikimedia.org/wiki/File:Gfp-Wisconsin-algoma-dusk-in-the-harbor.jpg"),
    ("fishing_shack",   "natural",
     "https://upload.wikimedia.org/wikipedia/commons/5/54/Gfp-Wisconsin-lonely-fishing-shack.jpg",
     "CC-PD", "Yinan Chen (Goodfreephotos_com)",
     "https://commons.wikimedia.org/wiki/File:Gfp-Wisconsin-lonely-fishing-shack.jpg"),
    ("small_cliff",     "natural",
     "https://upload.wikimedia.org/wikipedia/commons/b/b3/Gfp-Wisconsin-sturgeon-bay-a-small-cliff.jpg",
     "CC-PD", "Yinan Chen (Goodfreephotos_com)",
     "https://commons.wikimedia.org/wiki/File:Gfp-Wisconsin-sturgeon-bay-a-small-cliff.jpg"),
    # Mid-range: paintings (semantic content + non-photographic style)
    ("manet_cafe",      "painting",
     "https://upload.wikimedia.org/wikipedia/commons/c/c8/Edouard_Manet_-_At_the_Caf%C3%A9_-_Google_Art_Project.jpg",
     "Public domain", "Édouard Manet (1878) / Google Art Project",
     "https://commons.wikimedia.org/wiki/File:Edouard_Manet_-_At_the_Café_-_Google_Art_Project.jpg"),
    ("vermeer_art",     "painting",
     "https://upload.wikimedia.org/wikipedia/commons/5/5e/Jan_Vermeer_-_The_Art_of_Painting_-_Google_Art_Project.jpg",
     "Public domain", "Johannes Vermeer (~1668) / Google Art Project",
     "https://commons.wikimedia.org/wiki/File:Jan_Vermeer_-_The_Art_of_Painting_-_Google_Art_Project.jpg"),
    ("hdr_fire",        "natural",
     "https://upload.wikimedia.org/wikipedia/commons/5/53/Gfp-HDR-fire.jpg",
     "CC-PD", "Yinan Chen (Goodfreephotos_com)",
     "https://commons.wikimedia.org/wiki/File:Gfp-HDR-fire.jpg"),
]

# Programmatic images (no licence needed — generated)
GENERATED_IMAGES = [
    # (label, group, description)
    ("noise_gauss50",  "synthetic_gen", "Gaussian noise std=0.50"),
    ("noise_gauss10",  "synthetic_gen", "Gaussian noise std=0.10"),
    ("noise_gauss05",  "synthetic_gen", "Gaussian noise std=0.05"),
    ("pink_noise",     "synthetic_gen", "Pink noise (1/f power spectrum)"),
    ("checker8",       "synthetic_gen", "Checkerboard 8×8 px"),
    ("checker32",      "synthetic_gen", "Checkerboard 32×32 px"),
    ("stripes_h",      "synthetic_gen", "Horizontal stripes f=0.05"),
    ("stripes_d",      "synthetic_gen", "Diagonal stripes 45° f=0.03"),
]

# Existing 13 images with prior slopes (from distance_metric.py)
# (label, rel_path, group, prior_slope, prior_N, known_null_frac)
EXISTING = [
    ("boardwalk",    "natural/boardwalk_nature.jpg",   "natural",   0.9510,  48,  0.0046),
    ("frog_on_log",  "natural/frog_on_log.jpg",        "natural",   1.4238,  48,  0.0111),
    ("nature_land",  "natural/nature_landscape.jpg",   "natural",   1.1147,  48,  0.0128),
    ("boy_face",     "faces/boy_face_venezuela.jpg",   "faces",     3.1829,  48,  0.0049),
    ("girl_sad",     "faces/girl_sad_face.jpg",        "faces",     2.7834,  48,  0.0032),
    ("wayuu_woman",  "faces/wayuu_woman.jpg",          "faces",     1.7483,  48,  0.0162),
    ("wood_grain",   "texture/wood_grain.png",         "texture",   0.5969,  192, 0.0159),
    ("dirt_soil",    "texture/dirt_soil.png",          "texture",   0.8536,  48,  0.0088),
    ("grass_meadow", "texture/grass_meadow.png",       "texture",   0.5778,  48,  0.0477),
    ("soft_blobs",   "synthetic/soft_blobs.png",       "synthetic", 4.2917,  48,  0.0014),
    ("hard_shapes",  "synthetic/hard_shapes.png",      "synthetic", 8.0176,  48,  0.0426),
    ("linear_grad",  "synthetic/linear_gradient.png",  "synthetic", 6.5295,  48,  0.0059),
    ("radial_grad",  "synthetic/radial_gradient.png",  "synthetic", 0.6376,  48,  0.0000),
]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    lines: list[str] = []
    def log(s: str = "", end: str = "\n") -> None:
        print(s, end=end, flush=True)
        lines.append(s)

    t_total = time.perf_counter()

    # ── 0. Preamble ────────────────────────────────────────────────────────────
    log("=" * 72)
    log("  DISTANCE METRIC V2 — ResNet50 semantic features + slope correlation")
    log("=" * 72)
    log()
    log("  Metric: ResNet50 (IMAGENET1K_V2) penultimate-layer features (2048-dim)")
    log("  Distance: cosine distance to centroid of 16 ILSVRC2012 reference images")
    log("  Preprocessing: RGB 224×224, ImageNet mean/std normalisation")
    log("  SANITY CHECK gates analysis: confirms synthetics > naturals in distance")
    log()

    # ── 1. Load ResNet50 ───────────────────────────────────────────────────────
    log("─" * 72)
    log("  STEP 1: Load ResNet50 (IMAGENET1K_V2)")
    log("─" * 72)
    if not RESNET_WEIGHTS.exists():
        log(f"  ERROR: ResNet50 weights not found at {RESNET_WEIGHTS}")
        log("  Run: curl -L https://download.pytorch.org/models/resnet50-11ad3fa6.pth -o /tmp/resnet50.pth")
        sys.exit(1)

    device_rn = "cpu"  # ResNet50 runs on CPU (small model, no MPS issues)
    rn50 = _load_resnet50(RESNET_WEIGHTS, device_rn)
    log(f"  ResNet50 loaded from {RESNET_WEIGHTS} on device={device_rn}")
    log()

    # ── 2. Download new CC0/PD images ─────────────────────────────────────────
    log("─" * 72)
    log("  STEP 2: Download CC0/PD images")
    log("─" * 72)
    import urllib.request, shutil

    new_dir = RES / "new_v2"
    new_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[tuple] = []  # (label, path, group, licence, author, source_url)
    for label, group, url, licence, author, wiki_url in DOWNLOAD_IMAGES:
        ext = Path(url.split("?")[0]).suffix or ".jpg"
        dest = new_dir / f"{label}{ext}"
        if dest.exists():
            log(f"  SKIP (exists): {label}")
            downloaded.append((label, str(dest), group, licence, author, wiki_url))
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            with open(dest, "wb") as f:
                f.write(data)
            # Verify it's a valid image
            Image.open(dest).verify()
            log(f"  OK ({len(data)//1024} KB): {label}")
            downloaded.append((label, str(dest), group, licence, author, wiki_url))
        except Exception as e:
            log(f"  FAIL: {label} — {e}")
    log()

    # ── 3. Generate programmatic images ───────────────────────────────────────
    log("─" * 72)
    log("  STEP 3: Generate programmatic synthetic images")
    log("─" * 72)

    syn_dir = RES / "synthetic"
    gen_cache: dict[str, np.ndarray] = {}  # label → (H,W) float32

    gen_specs = [
        ("noise_gauss50", _gen_gaussian_noise(0.50, RNG_SEED + 0)),
        ("noise_gauss10", _gen_gaussian_noise(0.10, RNG_SEED + 1)),
        ("noise_gauss05", _gen_gaussian_noise(0.05, RNG_SEED + 2)),
        ("pink_noise",    _gen_pink_noise(RNG_SEED + 3)),
        ("checker8",      _gen_checkerboard(8)),
        ("checker32",     _gen_checkerboard(32)),
        ("stripes_h",     _gen_stripes(0.05, 0.0)),
        ("stripes_d",     _gen_stripes(0.03, 45.0)),
    ]
    for label, gray in gen_specs:
        dest = syn_dir / f"{label}.png"
        if not dest.exists():
            img = Image.fromarray((gray * 255).clip(0, 255).astype(np.uint8), mode="L")
            img.save(str(dest))
        gen_cache[label] = gray
        log(f"  Generated: {label}  shape={gray.shape}  range=[{gray.min():.3f},{gray.max():.3f}]")
    log()

    # ── 4. Reference centroid (ResNet50 features of 16 ILSVRC2012 images) ─────
    log("─" * 72)
    log("  STEP 4: ResNet50 reference centroid (16 ILSVRC2012 images)")
    log("─" * 72)
    import glob
    ref_paths = sorted(glob.glob(str(REF_DIR / "*.png")))
    assert len(ref_paths) == 16, f"Expected 16, got {len(ref_paths)}"

    ref_feats = []
    for p in ref_paths:
        rgb = _load_rgb_224(p)
        feat = _resnet_features(rgb, rn50, device_rn)
        ref_feats.append(feat)
    ref_feats   = np.stack(ref_feats)       # (16, 2048)
    centroid     = ref_feats.mean(axis=0)   # (2048,) raw mean

    ref_dists = [_cosine_dist(f, centroid) for f in ref_feats]
    log(f"  Reference centroid from {len(ref_paths)} ILSVRC2012 images")
    log(f"  Within-ref cosine dist: mean={np.mean(ref_dists):.4f}  "
        f"min={np.min(ref_dists):.4f}  max={np.max(ref_dists):.4f}")
    log()

    # ── 5. SANITY CHECK ────────────────────────────────────────────────────────
    log("─" * 72)
    log("  STEP 5: SANITY CHECK — ResNet50 distance ordering")
    log("─" * 72)
    log("  Must confirm: d(linear_gradient) > d(boardwalk) > d(hard_shapes???)")
    log("  If natural < synthetic in distance, metric is wrong — stop here.")
    log()

    def _dist_for(path_or_gray, is_gray=False):
        if is_gray:
            rgb = _gray_to_rgb_224(path_or_gray.astype(np.float32))
        else:
            rgb = _load_rgb_224(path_or_gray)
        return _cosine_dist(_resnet_features(rgb, rn50, device_rn), centroid)

    boardwalk_d = _dist_for(str(RES / "natural" / "boardwalk_nature.jpg"))
    linear_d    = _dist_for(str(RES / "synthetic" / "linear_gradient.png"))
    hard_d      = _dist_for(str(RES / "synthetic" / "hard_shapes.png"))
    gauss_d     = _dist_for(gen_cache["noise_gauss50"], is_gray=True)

    log(f"  boardwalk_nature   d = {boardwalk_d:.4f}  (natural — should be NEAR)")
    log(f"  linear_gradient    d = {linear_d:.4f}  (synthetic — should be FAR)")
    log(f"  hard_shapes        d = {hard_d:.4f}  (synthetic — should be FAR)")
    log(f"  noise_gauss50      d = {gauss_d:.4f}  (noise — should be FAR)")
    log()

    natural_far_fail = linear_d <= boardwalk_d or hard_d <= boardwalk_d
    if natural_far_fail:
        log("  SANITY CHECK FAILED.")
        log(f"  linear_d={linear_d:.4f} <= boardwalk_d={boardwalk_d:.4f} or "
            f"hard_d={hard_d:.4f} <= boardwalk_d={boardwalk_d:.4f}")
        log("  The ResNet50 metric does NOT order images as expected.")
        log("  The metric is broken — aborting analysis.")
        _write_results(OUT_FILE, lines)
        return
    log("  SANITY CHECK PASSED.")
    log(f"  linear_d ({linear_d:.4f}) > boardwalk_d ({boardwalk_d:.4f}) ✓")
    log(f"  hard_d   ({hard_d:.4f}) > boardwalk_d ({boardwalk_d:.4f}) ✓")
    log()

    # ── 6. Load ResShift engine ────────────────────────────────────────────────
    log("─" * 72)
    log("  STEP 6: Load ResShift engine for calibrations")
    log("─" * 72)
    op     = BicubicDownsample(SCALE)
    engine = ResShiftEngine(op)
    log()

    # ── 7. Build full image list and compute distances ─────────────────────────
    log("─" * 72)
    log("  STEP 7: ResNet50 distances for all images")
    log("─" * 72)

    # Will collect: (label, path_or_gray, group, prior_slope|None, prior_N|None,
    #                known_null_frac|None, is_gen_gray)
    all_images: list[dict] = []

    # Existing 13
    for label, rel, group, slope, sn, nf in EXISTING:
        all_images.append({
            "label": label, "path": str(RES / rel), "group": group,
            "prior_slope": slope, "prior_N": sn, "known_null": nf,
            "is_gen": False, "gray": None,
        })

    # Downloaded images (all need new calibration)
    for label, path, group, lic, author, wiki_url in downloaded:
        all_images.append({
            "label": label, "path": path, "group": group,
            "prior_slope": None, "prior_N": None, "known_null": None,
            "is_gen": False, "gray": None,
        })

    # Generated images (all need new calibration)
    for label, group, desc in GENERATED_IMAGES:
        gray = gen_cache[label]
        all_images.append({
            "label": label, "path": None, "group": group,
            "prior_slope": None, "prior_N": None, "known_null": None,
            "is_gen": True, "gray": gray,
        })

    log(f"  {'Image':<16} {'group':<14} {'dist':>8}")
    log("  " + "─" * 42)

    for img in all_images:
        if img["is_gen"]:
            rgb  = _gray_to_rgb_224(img["gray"].astype(np.float32))
            feat = _resnet_features(rgb, rn50, device_rn)
        else:
            rgb  = _load_rgb_224(img["path"])
            feat = _resnet_features(rgb, rn50, device_rn)
        dist = _cosine_dist(feat, centroid)
        img["dist"] = dist
        log(f"  {img['label']:<16} {img['group']:<14} {dist:>8.4f}")
    log()

    # ── 8. null_frac_gt for all images ────────────────────────────────────────
    log("─" * 72)
    log("  STEP 8: null_frac_gt")
    log("─" * 72)

    for img in all_images:
        if img["known_null"] is not None:
            img["null_frac"] = img["known_null"]
        else:
            if img["is_gen"]:
                gray = img["gray"].astype(np.float64)
            else:
                gray = _load_gray_256(img["path"])
                img["gray256"] = gray
            img["null_frac"] = _null_frac_gt(gray, op)
        nf_src = "prior" if img["known_null"] is not None else "measured"
        log(f"  {img['label']:<16}  null_frac_gt = {img['null_frac']:.4f}  ({nf_src})")
    log()

    # ── 9. N=48 calibrations for new images ───────────────────────────────────
    log("─" * 72)
    need_cal = [i for i in all_images if i["prior_slope"] is None]
    est_s    = len(need_cal) * N_NEW * 1.1
    log(f"  STEP 9: N={N_NEW} calibrations — {len(need_cal)} images")
    log(f"  Estimated: {len(need_cal)} × {N_NEW} × ~1.1s ≈ {est_s:.0f}s  ({est_s/60:.0f} min)")
    log("─" * 72)

    for img in need_cal:
        if img["is_gen"]:
            gray = img["gray"].astype(np.float64)
        else:
            gray256 = img.get("gray256")
            gray = gray256 if gray256 is not None else _load_gray_256(img["path"])
        log(f"  [{img['label']}] N={N_NEW} …", end="")
        t0  = time.perf_counter()
        cal = _run_calibration(gray, op, engine, N_NEW)
        dt  = time.perf_counter() - t0
        img["prior_slope"] = cal["slope"]
        img["prior_N"]     = N_NEW
        img["cal_r"]       = cal["r"]
        log(f"  slope={cal['slope']:.4f}  r={cal['r']:+.4f}  ({dt:.0f}s)")
    log()

    # ── 10. Data table ─────────────────────────────────────────────────────────
    log("─" * 72)
    log("  STEP 10: Data table (sorted by distance)")
    log("─" * 72)
    log()

    labels_v  = []; dists_v = []; slopes_v = []; nulls_v = []; groups_v = []
    slope_Ns  = []; is_syn_v = []

    for img in sorted(all_images, key=lambda x: x["dist"]):
        labels_v.append(img["label"])
        dists_v.append(img["dist"])
        slopes_v.append(img["prior_slope"])
        nulls_v.append(img["null_frac"])
        groups_v.append(img["group"])
        slope_Ns.append(img["prior_N"])
        is_syn   = img["group"] in ("synthetic", "synthetic_gen")
        is_syn_v.append(is_syn)

    dists_v  = np.array(dists_v)
    slopes_v = np.array(slopes_v)
    nulls_v  = np.array(nulls_v)
    is_syn_v = np.array(is_syn_v)

    log(f"  {'rk':>3} {'image':<16} {'group':<14} {'dist':>8} {'slope':>8} {'null_f':>7}  slope_N")
    log("  " + "─" * 64)
    for rank, (lab, grp, d, sl, nf, sN) in enumerate(
            zip(labels_v, groups_v, dists_v, slopes_v, nulls_v, slope_Ns), 1):
        sfx = " *" if grp in ("synthetic", "synthetic_gen") else ""
        log(f"  {rank:>3} {lab:<16} {grp:<14} {d:>8.4f} {sl:>8.4f} {nf:>7.4f}  {sN}{sfx}")
    log("  (* synthetic or generated)")
    log()

    n_total = len(labels_v)
    n_nonsyn = int((~is_syn_v).sum())

    # ── 11. Correlation analysis ───────────────────────────────────────────────
    log("─" * 72)
    log("  STEP 11: Correlation analysis")
    log("─" * 72)
    log()
    log("  Fisher-z 95% CI. Treat every finding as PRELIMINARY (n is stated).")
    log()

    corr_lines = []
    _corr_block(f"all images, n={n_total}", dists_v, slopes_v, nulls_v, corr_lines)
    corr_lines.append("")
    _corr_block(f"non-synthetic only, n={n_nonsyn}",
                dists_v[~is_syn_v], slopes_v[~is_syn_v], nulls_v[~is_syn_v], corr_lines)
    for cl in corr_lines:
        log(cl)
    log()

    # ── 12. Verdict ────────────────────────────────────────────────────────────
    log("─" * 72)
    log("  STEP 12: Verdict")
    log("─" * 72)
    log()

    r_all, _ = scipy_stats.pearsonr(dists_v, slopes_v)
    lo_all, hi_all = _pearson_ci(r_all, n_total)
    r_ns, _  = scipy_stats.pearsonr(dists_v[~is_syn_v], slopes_v[~is_syn_v])
    lo_ns, hi_ns = _pearson_ci(r_ns, n_nonsyn)
    r_pa_all = _partial_corr(dists_v, slopes_v, nulls_v)
    r_pa_ns  = _partial_corr(dists_v[~is_syn_v], slopes_v[~is_syn_v], nulls_v[~is_syn_v])

    ci_ns_excl_0 = (lo_ns > 0 or hi_ns < 0)
    ci_all_excl_0 = (lo_all > 0 or hi_all < 0)

    if ci_ns_excl_0 and abs(r_pa_ns) > 0.2:
        verdict = "(a) POSITIVE — non-synthetic correlation survives; partial r consistent"
    elif ci_ns_excl_0:
        verdict = "(a-weak) POSITIVE but partial-r control weakens it; treat as preliminary"
    elif ci_all_excl_0 and not ci_ns_excl_0:
        verdict = "(b) SYNTHETIC-DRIVEN — CI excludes 0 only with synthetics; without synthetics null"
    else:
        verdict = "(b) NULL — 95% CI includes 0 in both splits; no evidence for distance→slope"

    log(f"  {verdict}")
    log()
    log(f"  n={n_total}: r={r_all:+.3f}  CI=[{lo_all:+.3f},{hi_all:+.3f}]  partial_r={r_pa_all:+.3f}")
    log(f"  n={n_nonsyn} (non-syn): r={r_ns:+.3f}  CI=[{lo_ns:+.3f},{hi_ns:+.3f}]  partial_r={r_pa_ns:+.3f}")
    log()
    log(f"  KEY RESULT: the without-synthetics correlation (n={n_nonsyn}) is the")
    log("  primary result. The all-images number is dominated by synthetic extremes.")
    log()

    runtime = time.perf_counter() - t_total
    log(f"  Total runtime: {runtime:.0f}s ({runtime/60:.1f} min)")
    log()

    # ── 13. Update SOURCES_LICENCE.md ─────────────────────────────────────────
    _update_sources(downloaded, SOURCES)

    _write_results(OUT_FILE, lines)
    print(f"\nResults written to {OUT_FILE}", flush=True)


def _write_results(path: Path, lines: list) -> None:
    with open(str(path), "w") as f:
        f.write("\n".join(lines) + "\n")


def _update_sources(downloaded: list, sources_path: Path) -> None:
    """Append new image entries to SOURCES_LICENCE.md."""
    existing = sources_path.read_text() if sources_path.exists() else ""
    additions = []
    for label, path, group, licence, author, wiki_url in downloaded:
        if label in existing:
            continue
        additions.append(f"""
### {label}.* (new_v2/)
- **Author:** {author}
- **Licence:** {licence}
- **Source:** {wiki_url}
""")
    if additions:
        with open(str(sources_path), "a") as f:
            f.write("\n## new_v2 (distance metric v2 study)\n")
            f.writelines(additions)


if __name__ == "__main__":
    main()
