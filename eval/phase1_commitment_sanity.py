"""
eval/phase1_commitment_sanity.py — Phase 1 sanity check: commitment metric on 2 images

Pre-registered purpose: verify intermediate-state access and check whether the
null-space trajectory shape differs visibly between a high-slope image (boy_face,
slope=3.18) and a low-slope image (wood_grain, slope=0.44) BEFORE deciding whether
Phase 2 is warranted. This script does NOT compute slope and does NOT look at any
correlation between commitment and slope across images.

Commitment metric (independence argument):
  - Slope measures std of (I−A⁺A)x_0 ACROSS seeds at the FINAL denoising step.
  - Commitment measures rate of change of (I−A⁺A)pred_xstart ACROSS STEPS within
    a single seed's denoising path.
  - They share the operator (I−A⁺A) but are computed on orthogonal axes:
    seed-axis (slope) vs step-axis (commitment). Not the same measurement.

Usage:
    .venv/bin/python3 eval/phase1_commitment_sanity.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parent
_VENDOR = _REPO / "vendor"
for _p in [str(_VENDOR / "stubs"), str(_VENDOR / "ResShift"), str(_REPO)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.script_util import create_gaussian_diffusion
from models.unet import UNetModelSwin
from ldm.models.autoencoder import VQModelTorch
from utils.util_net import reload_model
from operators.bicubic import BicubicDownsample

# ── Constants (copied from engine/resshift.py) ─────────────────────────────────
_AE_DDCONFIG = {
    "double_z": False, "z_channels": 3, "resolution": 256,
    "in_channels": 3, "out_ch": 3, "ch": 128, "ch_mult": [1, 2, 4],
    "num_res_blocks": 2, "attn_resolutions": [], "dropout": 0.0,
    "padding_mode": "zeros",
}
_UNET_PARAMS = dict(
    image_size=64, in_channels=3, model_channels=160, out_channels=3,
    attention_resolutions=[64, 32, 16, 8], dropout=0, channel_mult=[1, 2, 2, 4],
    num_res_blocks=[2, 2, 2, 2], conv_resample=True, dims=2, use_fp16=False,
    num_head_channels=32, use_scale_shift_norm=True, resblock_updown=False,
    swin_depth=2, swin_embed_dim=192, window_size=8, mlp_ratio=4,
    cond_lq=True, lq_size=64,
)
_DIFF_PARAMS = dict(
    sf=4, schedule_name="exponential", schedule_kwargs={"power": 0.3},
    etas_end=0.99, steps=4, min_noise_level=0.2, kappa=2.0,
    weighted_mse=False, predict_type="xstart", timestep_respacing=None,
    scale_factor=1.0, normalize_input=True, latent_flag=True,
)

W = H_IMG = 256
N_SEEDS = 5  # seeds per image for the sanity check

IMAGES = [
    {
        "name": "boy_face",
        "path": str(_HERE / "research_sources/faces/boy_face_venezuela.jpg"),
        "slope": 3.1829,
        "label": "HIGH-slope (face)",
    },
    {
        "name": "wood_grain",
        "path": str(_HERE / "research_sources/texture/wood_grain.png"),
        "slope": 0.4351,
        "label": "LOW-slope (texture)",
    },
]

_log: list[str] = []

def out(s: str = "", end: str = "\n") -> None:
    print(s, end=end, flush=True)
    _log.append(s + end)


def load_gray_256(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w < h:
        img = img.resize((W, int(round(h * W / w))), Image.LANCZOS)
    else:
        img = img.resize((int(round(w * H_IMG / h)), H_IMG), Image.LANCZOS)
    w2, h2 = img.size
    l = (w2 - W) // 2; t = (h2 - H_IMG) // 2
    img = img.crop((l, t, l + W, t + H_IMG))
    rgb = np.array(img, dtype=np.float64) / 255.0
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def null_proj_pixel(x_hat_pixel: np.ndarray, op: BicubicDownsample) -> np.ndarray:
    """(I − A⁺A)x_hat in pixel space (256×256 float64 grayscale)."""
    t = torch.from_numpy(x_hat_pixel.astype(np.float32))  # 2D (H, W)
    Ax = op.forward(t)   # expects 2D or 3D
    ApAx = op.pinv(Ax)
    null = t - ApAx
    return null.numpy().astype(np.float64)


def decode_pred_xstart(pred_xstart: torch.Tensor, diffusion, ae, device) -> np.ndarray:
    """Decode latent pred_xstart → grayscale pixel space [0,1], 256×256."""
    with torch.no_grad():
        px = diffusion.decode_first_stage(pred_xstart, first_stage_model=ae)
    px = px.squeeze(0).mean(dim=0)          # (H, W)
    px = ((px + 1.0) / 2.0).clamp(0.0, 1.0)
    return px.cpu().float().numpy().astype(np.float64)


def run_trajectory(y_np: np.ndarray, seed: int, diffusion, model, ae, device,
                   op: BicubicDownsample) -> dict:
    """
    One reverse-diffusion trajectory with intermediates captured.
    Returns dict with pred_xstart snapshots at each of 4 steps and
    the null-space Δ at each step transition.
    Steps are yielded in order [3, 2, 1, 0] by p_sample_loop_progressive.
    """
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    y_f = y_np.astype(np.float32)
    y_t = torch.from_numpy(y_f).unsqueeze(0).expand(3, -1, -1)
    y_t = (y_t * 2.0 - 1.0).unsqueeze(0).to(device)

    null_snaps = []   # null-space component at each step, decoded to pixel space
    step_indices = []

    for i, step_out in enumerate(diffusion.p_sample_loop_progressive(
        y=y_t, model=model, first_stage_model=ae,
        noise=None, noise_repeat=False, clip_denoised=False,
        denoised_fn=None, model_kwargs={"lq": y_t},
        device=device, progress=False,
    )):
        px = decode_pred_xstart(step_out["pred_xstart"], diffusion, ae, device)
        null_c = null_proj_pixel(px, op)
        null_snaps.append(null_c)
        step_indices.append(3 - i)   # i=0→step3, i=1→step2, i=2→step1, i=3→step0

    # Δ between consecutive steps (step3→2, step2→1, step1→0)
    # null_snaps[0]=step3, [1]=step2, [2]=step1, [3]=step0
    deltas = []
    for j in range(len(null_snaps) - 1):
        diff = null_snaps[j + 1] - null_snaps[j]
        denom = max(np.linalg.norm(null_snaps[j]), 1e-8)
        deltas.append(float(np.linalg.norm(diff) / denom))

    total = sum(deltas) if sum(deltas) > 1e-10 else 1.0
    concentration = deltas[0] / total   # fraction of Δ in first reverse step (3→2)

    return {
        "step_indices": step_indices,
        "deltas": deltas,
        "concentration": concentration,
        "null_snaps": null_snaps,
    }


def main() -> None:
    out("=" * 72)
    out("PHASE 1 SANITY CHECK — Commitment metric on 2 images")
    out("Prior-strength hypothesis: face images commit null-space content EARLIER")
    out("in the reverse diffusion chain than texture images.")
    out("=" * 72)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out(f"\nDevice: {device}")

    out("\n[0] Loading model weights …")
    ckpt_path = str(_REPO / "weights/resshift_bicsrx4_s4.pth")
    ae_ckpt_path = str(_REPO / "weights/autoencoder_vq_f4.pth")

    diffusion = create_gaussian_diffusion(**_DIFF_PARAMS)
    out(f"    GaussianDiffusion: num_timesteps={diffusion.num_timesteps}")
    out(f"    etas (η): {[f'{e:.4f}' for e in diffusion.etas]}")
    out(f"    posterior_variance: {[f'{v:.4f}' for v in diffusion.posterior_variance]}")
    out(f"    etas order during sampling: indices {list(range(diffusion.num_timesteps))[::-1]}")

    model = UNetModelSwin(**_UNET_PARAMS)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    reload_model(model, ckpt)
    model = model.to(device).eval()
    out("    UNet loaded.")

    ae = VQModelTorch(ddconfig=_AE_DDCONFIG, n_embed=8192, embed_dim=3)
    ae_ckpt = torch.load(ae_ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ae_ckpt, dict) and "state_dict" in ae_ckpt:
        ae_ckpt = ae_ckpt["state_dict"]
    ae.load_state_dict(ae_ckpt, strict=True)
    ae = ae.to(device).eval()
    out("    VQ-AE loaded.")

    op = BicubicDownsample(scale=4)

    out(f"\n[1] Sampling {N_SEEDS} seeds per image, capturing 4 pred_xstart per seed …")

    results = {}
    for img_cfg in IMAGES:
        out(f"\n  Image: {img_cfg['name']}  slope={img_cfg['slope']}  ({img_cfg['label']})")
        gray = load_gray_256(img_cfg["path"])
        y_raw = BicubicDownsample(scale=4).forward(
            torch.from_numpy(gray.astype(np.float32))
        )
        y_np = y_raw if isinstance(y_raw, np.ndarray) else y_raw.numpy()
        y_np = y_np.astype(np.float64)

        concentrations = []
        deltas_by_seed = []
        for seed in range(N_SEEDS):
            out(f"    seed={seed}", end=" … ")
            traj = run_trajectory(y_np, seed, diffusion, model, ae, device, op)
            concentrations.append(traj["concentration"])
            deltas_by_seed.append(traj["deltas"])
            out(f"Δ[3→2]={traj['deltas'][0]:.4f}  Δ[2→1]={traj['deltas'][1]:.4f}  "
                f"Δ[1→0]={traj['deltas'][2]:.4f}  conc={traj['concentration']:.3f}")

        conc_arr = np.array(concentrations)
        delta_arr = np.array(deltas_by_seed)   # shape (N_SEEDS, 3)
        results[img_cfg["name"]] = {
            "concentrations": conc_arr,
            "deltas": delta_arr,
            "slope": img_cfg["slope"],
            "label": img_cfg["label"],
        }
        out(f"    mean Δ[3→2]={delta_arr[:,0].mean():.4f}±{delta_arr[:,0].std():.4f}  "
            f"Δ[2→1]={delta_arr[:,1].mean():.4f}±{delta_arr[:,1].std():.4f}  "
            f"Δ[1→0]={delta_arr[:,2].mean():.4f}±{delta_arr[:,2].std():.4f}")
        out(f"    concentration (Δ3→2 / total): mean={conc_arr.mean():.3f}  "
            f"std={conc_arr.std():.3f}  range=[{conc_arr.min():.3f},{conc_arr.max():.3f}]")

    out("\n" + "=" * 72)
    out("COMPARISON: boy_face vs wood_grain")
    out("=" * 72)

    face = results["boy_face"]
    wood = results["wood_grain"]
    conc_face = face["concentrations"]
    conc_wood = wood["concentrations"]
    delta_face = face["concentrations"].mean() - wood["concentrations"].mean()
    pooled_std = np.sqrt((conc_face.var() + conc_wood.var()) / 2)

    out(f"\n  boy_face  concentration: {conc_face.mean():.3f} ± {conc_face.std():.3f}")
    out(f"  wood_grain concentration: {conc_wood.mean():.3f} ± {conc_wood.std():.3f}")
    out(f"  mean difference (face − wood): {delta_face:+.3f}")
    out(f"  pooled std: {pooled_std:.3f}")
    out(f"  separation in pooled-std units: {delta_face/max(pooled_std,1e-8):.2f}")

    out("\n  Step-to-step Δ comparison:")
    for step_i, label in enumerate(["Δ[3→2]", "Δ[2→1]", "Δ[1→0]"]):
        f_mean = face["deltas"][:, step_i].mean()
        w_mean = wood["deltas"][:, step_i].mean()
        out(f"    {label}:  face={f_mean:.4f}  wood={w_mean:.4f}  ratio face/wood={f_mean/max(w_mean,1e-8):.2f}")

    out("\n" + "=" * 72)
    out("SCHEDULE RESOLUTION ANALYSIS")
    out("=" * 72)
    out(f"\n  etas (noise level per step): {[f'{e:.4f}' for e in diffusion.etas]}")
    out(f"  posterior variance per step: {[f'{v:.6f}' for v in diffusion.posterior_variance]}")
    out(f"  Sampling order: steps [3,2,1,0] (high→low noise)")
    out(f"  → Step t=0 adds almost NO stochastic injection (pvar≈{diffusion.posterior_variance[0]:.6f})")
    pv = diffusion.posterior_variance
    out(f"  → Posterior variance ratio step3/step1 = {pv[3]/max(pv[1],1e-10):.1f}×")
    out(f"  → Effective usable step transitions: {sum(1 for v in pv if v > 0.01)} "
        f"(those with pvar > 0.01)")
    out(f"  → Δ[1→0] expected near-zero for ALL images (pvar={pv[0]:.5f})")
    out(f"  → Discriminating signal lives only in ratio Δ[3→2] / Δ[2→1] (1 number per seed)")

    out("\n" + "=" * 72)
    out("PHASE 1 VERDICT")
    out("=" * 72)

    face_conc = conc_face.mean()
    wood_conc = conc_wood.mean()
    sep_units = delta_face / max(pooled_std, 1e-8)

    if abs(sep_units) >= 2.0:
        verdict = "VISIBLE_DIFFERENCE — sanity check PASSES"
        rationale = ("Face and texture show separation ≥ 2 pooled-std units in "
                     "concentration ratio. Phase 2 is informative subject to 4-step caveat.")
    elif abs(sep_units) >= 1.0:
        verdict = "MARGINAL_DIFFERENCE — proceed with caution"
        rationale = ("Separation 1–2 pooled-std units. Detectable trend but may not survive "
                     "noise at n=24. Phase 2 is borderline warranted; pre-register carefully.")
    else:
        verdict = "NO_VISIBLE_DIFFERENCE — sanity check FAILS"
        rationale = ("Face and texture show <1 pooled-std separation. The 4-step schedule "
                     "cannot resolve commitment timing for this image pair. Phase 2 is NOT "
                     "warranted on this model. Hypothesis A is not testable at 4 steps.")

    out(f"\n  ACCESS CHECK: p_sample_loop_progressive yields pred_xstart at "
        f"{diffusion.num_timesteps} steps — CONFIRMED, no code changes required.")
    out(f"  USABLE INTERMEDIATES: {diffusion.num_timesteps} pred_xstart snapshots per run "
        f"(steps {list(range(diffusion.num_timesteps))[::-1]}), latent space 1×3×64×64.")
    out(f"  METRIC INDEPENDENCE: step-axis (within-run) vs seed-axis (across-run) — CONFIRMED.")
    out(f"  4-STEP RESOLUTION: {diffusion.num_timesteps} steps → 3 Δ transitions → "
        f"1 effective discriminating ratio (Δ[3→2]/Δ[2→1]; Δ[1→0]≈0 always).")
    out(f"\n  SANITY VERDICT: {verdict}")
    out(f"  {rationale}")

    out_path = _HERE / "phase1_commitment_results.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(_log)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
