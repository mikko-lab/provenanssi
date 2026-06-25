"""
eval/smoke_torch.py — PyTorch / MPS environment smoke test.

Answers two questions before we connect the provenance layer:
  1. Is torch importable and is MPS (Apple GPU) available?
  2. Does a stochastic SR model produce genuinely different outputs per run?

Model choice: MinimalSRDropout
    A 3-layer conv net (bicubic pre-upsample + 2× residual conv + 1× output conv)
    with Dropout2d in the residual blocks.  No pretrained weights — this is
    intentional.  The goal is to verify the *mechanism* (MC-dropout diversity),
    not to evaluate reconstruction quality.  We test in train() mode so dropout
    is active; eval() mode gives deterministic output as a control.

Device: MPS if available, else CPU.

No PIL, scipy, or torchvision required — PNG saving uses Python stdlib only
(struct + zlib).
"""
from __future__ import annotations
import os
import struct
import sys
import time
import zlib

import numpy as np

# --------------------------------------------------------------------------
# Set up PYTHONPATH for brew-installed torch (Python 3.14)
# This block is a no-op when torch is already importable.
_BREW_TORCH = (
    "/opt/homebrew/Cellar/pytorch/2.12.1/libexec/lib/python3.14/site-packages"
)
_BREW_NUMPY = "/opt/homebrew/opt/numpy/lib/python3.14/site-packages"
for _p in (_BREW_NUMPY, _BREW_TORCH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Minimal PNG writer (stdlib only — no PIL/Pillow required)

def _png_chunk(tag: bytes, data: bytes) -> bytes:
    c = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", c)


def save_png_gray(arr: np.ndarray, path: str) -> None:
    """Save a 2-D float array [0,1] as 8-bit grayscale PNG."""
    h, w = arr.shape
    px = (arr.clip(0, 1) * 255).astype(np.uint8)
    # PNG scanlines: prepend filter byte 0 to each row
    raw = b"".join(b"\x00" + bytes(row) for row in px)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)  # bit depth=8, gray
    idat = zlib.compress(raw)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_png_chunk(b"IHDR", ihdr))
        f.write(_png_chunk(b"IDAT", idat))
        f.write(_png_chunk(b"IEND", b""))


def _autoscale(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-9)


# --------------------------------------------------------------------------
# Minimal SR model with MC-Dropout

class ResBlock(nn.Module):
    def __init__(self, ch: int, p_drop: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.drop  = nn.Dropout2d(p=p_drop)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.drop(F.relu(self.conv1(x))))


class MinimalSRDropout(nn.Module):
    """Small SR conv-net with MC-Dropout.

    Architecture:
      1. bicubic pre-upsample (scale×) via F.interpolate
      2. conv3×3 → 32 channels
      3. 2× ResBlock(ch=32, dropout=p_drop)
      4. conv3×3 → 1 channel (output)

    In train() mode, Dropout2d is active → stochastic output.
    In eval() mode, output is deterministic.
    """

    def __init__(self, scale: int = 2, p_drop: float = 0.3) -> None:
        super().__init__()
        self.scale = scale
        self.head  = nn.Conv2d(1, 32, 3, padding=1)
        self.body  = nn.Sequential(ResBlock(32, p_drop), ResBlock(32, p_drop))
        self.tail  = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(y, scale_factor=self.scale, mode="bicubic",
                          align_corners=False)
        return x + self.tail(self.body(F.relu(self.head(x))))


# --------------------------------------------------------------------------
# Helpers

def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    return t.squeeze().detach().cpu().float().numpy()


def make_test_input(h: int = 64, w: int = 64, seed: int = 42) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    y_np = rng.random((h, w)).astype(np.float32)
    return torch.from_numpy(y_np).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)


# --------------------------------------------------------------------------
# Smoke test

def run(out_dir: str = "output") -> None:
    os.makedirs(out_dir, exist_ok=True)
    lines: list[str] = []

    def log(s: str) -> None:
        print(s)
        lines.append(s)

    log("=" * 60)
    log("smoke_torch.py — PyTorch / MPS smoke test")
    log("=" * 60)
    log("")

    # 1. Environment ---------------------------------------------------
    log(f"torch version : {torch.__version__}")
    mps_built = torch.backends.mps.is_built()
    mps_avail = torch.backends.mps.is_available()
    log(f"MPS built     : {mps_built}")
    log(f"MPS available : {mps_avail}   ← THE key question")
    device = torch.device("mps" if mps_avail else "cpu")
    log(f"device chosen : {device}")
    log("")

    # 2. Model + device ------------------------------------------------
    torch.manual_seed(0)
    model = MinimalSRDropout(scale=2, p_drop=0.3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model         : MinimalSRDropout(scale=2, p_drop=0.3)")
    log(f"parameters    : {n_params:,}")
    log(f"stochasticity : MC-Dropout (Dropout2d, p=0.3, active in train() mode)")
    log("")

    # 3. Input ---------------------------------------------------------
    y_cpu = make_test_input(64, 64, seed=42)          # 64×64 LR
    y = y_cpu.to(device)
    log(f"input shape   : {tuple(y.shape)}  (1×1×64×64 — LR measurement)")
    log("")

    # 4. Determinism check in eval() mode ------------------------------
    model.eval()
    with torch.no_grad():
        t_eval0 = time.perf_counter()
        out_eval_1 = model(y)
        out_eval_2 = model(y)
        t_eval = time.perf_counter() - t_eval0
    diff_eval = (out_eval_1 - out_eval_2).abs().max().item()
    log(f"eval() mode (deterministic control)")
    log(f"  run1 vs run2 max|diff| : {diff_eval:.3e}   "
        f"({'DETERMINISTIC ✓' if diff_eval < 1e-9 else 'UNEXPECTED VARIATION'})")
    log(f"  two runs time          : {t_eval*1000:.1f} ms")
    log("")

    # 5. Stochasticity check in train() mode (MC-Dropout) -------------
    model.train()
    with torch.no_grad():
        t_mc0 = time.perf_counter()
        members = [model(y) for _ in range(6)]
        t_mc = time.perf_counter() - t_mc0

    np_members = [tensor_to_np(m) for m in members]
    diffs = [
        float(np.max(np.abs(np_members[i] - np_members[j])))
        for i in range(6) for j in range(i + 1, 6)
    ]
    max_diff = max(diffs)
    mean_std = float(np.std(np.stack(np_members), axis=0).mean())

    log(f"train() mode (MC-Dropout, N=6 ensemble)")
    log(f"  max pairwise |diff|    : {max_diff:.6f}")
    log(f"  mean pixel std (σ)     : {mean_std:.6f}")
    log(f"  genuinely stochastic   : "
        f"{'YES ✓' if max_diff > 1e-6 else 'NO — deterministic even in train() mode'}")
    log(f"  N=6 ensemble time      : {t_mc*1000:.1f} ms  "
        f"({t_mc/6*1000:.1f} ms/member) on {device}")
    log("")

    if max_diff <= 1e-6:
        log("STOP: model is deterministic even in train() mode.")
        log("The ensemble story requires genuine stochasticity.")
        log("Check that Dropout2d is present and p_drop > 0.")
        return

    # 6. Images --------------------------------------------------------
    y_np = _autoscale(tensor_to_np(y_cpu))
    mean_out = _autoscale(np.mean(np_members, axis=0))
    std_out  = _autoscale(np.std(np_members,  axis=0))

    paths = {
        "smoke_01_input_lr.png":   y_np,
        "smoke_02_output_mean.png": mean_out,
        "smoke_03_output_std.png":  std_out,
    }
    for fname, arr in paths.items():
        p = os.path.join(out_dir, fname)
        save_png_gray(arr.astype(np.float64), p)
        log(f"saved: {p}")

    log("")
    log("Summary")
    log(f"  MPS available   : {mps_avail}")
    log(f"  MPS in use      : {device == torch.device('mps')}")
    log(f"  stochastic (N=6): max_diff={max_diff:.4f}, mean_σ={mean_std:.4f}")
    log(f"  ensemble speed  : {t_mc/6*1000:.1f} ms/member on {device}")
    log("=" * 60)

    report_path = os.path.join(out_dir, "smoke_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    run(out_dir)
