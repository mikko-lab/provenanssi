"""
engine/resshift.py — ResShift conditioned SR diffusion engine.

Model: resshift_bicsrx4_s4  (zsyOAOA/ResShift, GitHub release v2.0)
  - UNetModelSwin conditioned on LQ image (cond_lq=True)
  - VQ-autoencoder latent space (f=4, 64×64 latent for 256×256 HR)
  - 4-step exponential ResShift schedule
  - forward(x, timesteps, lq=...) — the lq arg IS the conditioning on y

Key property vs church-256: the model was trained on (LR, HR) pairs.
Its spread is a function of y, not a fixed prior.

R9: BoxDownsample only (scale=4 for this model's native x4 degradation).
R5: SHA-256 computed at load time.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# vendor path injection — must happen before ResShift imports

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_HERE, "..", "vendor")
_STUBS = os.path.join(_VENDOR, "stubs")
_RESSHIFT = os.path.join(_VENDOR, "ResShift")

for _p in [_STUBS, _RESSHIFT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# ResShift source imports

from models.script_util import create_gaussian_diffusion   # noqa: E402
from models.unet import UNetModelSwin                       # noqa: E402
from ldm.models.autoencoder import VQModelTorch             # noqa: E402
from utils.util_net import reload_model                     # noqa: E402

from operators.superres import BoxDownsample                # noqa: E402
from engine.base import Engine                              # noqa: E402

# ---------------------------------------------------------------------------

DEFAULT_CKPT = os.path.join(_HERE, "..", "weights", "resshift_bicsrx4_s4.pth")
DEFAULT_AE_CKPT = os.path.join(_HERE, "..", "weights", "autoencoder_vq_f4.pth")

# VQ-autoencoder architecture params (from bicx4_swinunet_lpips.yaml)
_AE_DDCONFIG = {
    "double_z": False,
    "z_channels": 3,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1, 2, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [],
    "dropout": 0.0,
    "padding_mode": "zeros",
}

# UNetModelSwin architecture params (from bicx4_swinunet_lpips.yaml)
_UNET_PARAMS = dict(
    image_size=64,
    in_channels=3,
    model_channels=160,
    out_channels=3,
    attention_resolutions=[64, 32, 16, 8],
    dropout=0,
    channel_mult=[1, 2, 2, 4],
    num_res_blocks=[2, 2, 2, 2],
    conv_resample=True,
    dims=2,
    use_fp16=False,
    num_head_channels=32,
    use_scale_shift_norm=True,
    resblock_updown=False,
    swin_depth=2,
    swin_embed_dim=192,
    window_size=8,
    mlp_ratio=4,
    cond_lq=True,   # ← conditions on the LQ image
    lq_size=64,
)

# ResShift diffusion params (from bicx4_swinunet_lpips.yaml)
_DIFF_PARAMS = dict(
    sf=4,
    schedule_name="exponential",
    schedule_kwargs={"power": 0.3},
    etas_end=0.99,
    steps=4,
    min_noise_level=0.2,
    kappa=2.0,
    weighted_mse=False,
    predict_type="xstart",
    timestep_respacing=None,
    scale_factor=1.0,
    normalize_input=True,
    latent_flag=True,
)


class ResShiftEngine(Engine):
    """ResShift bicubic-SR x4 engine.

    Conditioning arg in model.forward: lq= (the low-quality image tensor).
    This is the property church-256 lacked — the spread is a function of y.
    """

    def __init__(
        self,
        operator: BoxDownsample,
        checkpoint: str = DEFAULT_CKPT,
        ae_checkpoint: str = DEFAULT_AE_CKPT,
        device: str | torch.device | None = None,
    ) -> None:
        if not isinstance(operator, BoxDownsample):
            raise TypeError(
                f"ResShiftEngine supports BoxDownsample only (R9). Got {type(operator).__name__}"
            )
        if operator.scale != 4:
            raise ValueError(
                f"ResShift bicsrx4 model requires BoxDownsample(4). Got scale={operator.scale}."
            )
        self._op = operator

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._device = torch.device(device)

        self._diffusion = create_gaussian_diffusion(**_DIFF_PARAMS)

        # Load UNet
        self._model = UNetModelSwin(**_UNET_PARAMS)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        reload_model(self._model, ckpt)
        self._model = self._model.to(self._device).eval()

        # Load VQ-autoencoder
        self._ae = VQModelTorch(ddconfig=_AE_DDCONFIG, n_embed=8192, embed_dim=3)
        ae_ckpt = torch.load(ae_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ae_ckpt, dict) and "state_dict" in ae_ckpt:
            ae_ckpt = ae_ckpt["state_dict"]
        self._ae.load_state_dict(ae_ckpt, strict=True)
        self._ae = self._ae.to(self._device).eval()

        self._ckpt_sha256 = _sha256(checkpoint)
        self._ae_sha256  = _sha256(ae_checkpoint)

        # The conditioning argument: quoting UNetModelSwin.forward signature:
        # def forward(self, x, timesteps, lq=None, mask=None)
        # lq= is the low-quality image — this is how y enters the model.
        self._cond_arg = "lq"

        print(
            f"[ResShiftEngine] model=resshift_bicsrx4_s4  ae=autoencoder_vq_f4  "
            f"device={self._device}\n"
            f"  model SHA-256 : {self._ckpt_sha256[:16]}…\n"
            f"  ae    SHA-256 : {self._ae_sha256[:16]}…\n"
            f"  conditioning  : forward(x, t, lq=y)  [cond_lq=True]\n"
            f"  steps         : {_DIFF_PARAMS['steps']} (ResShift exponential schedule)"
        )

    # ------------------------------------------------------------------
    # Engine ABC

    def restore(self, y: np.ndarray) -> np.ndarray:
        """Single SR pass with seed=0."""
        return self._sample(y, seed=0)

    def ensemble(self, y: np.ndarray, n: int) -> list[np.ndarray]:
        return [self._sample(y, seed=i) for i in range(n)]

    def timed_ensemble(self, y: np.ndarray, n: int) -> tuple[list[np.ndarray], float]:
        t0 = time.perf_counter()
        members = self.ensemble(y, n)
        return members, time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Sampling

    def _sample(self, y_np: np.ndarray, seed: int) -> np.ndarray:
        """One ResShift SR pass conditioned on y_np (H×W grayscale in [0,1])."""
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

        # Grayscale → 3-channel, normalize [0,1] → [-1,1], add batch dim
        y_f = y_np.astype(np.float32)
        y_t = torch.from_numpy(y_f)          # (H_lr, W_lr)
        y_t = y_t.unsqueeze(0).expand(3, -1, -1)   # (3, H_lr, W_lr)
        y_t = y_t * 2.0 - 1.0               # [-1, 1]
        y_t = y_t.unsqueeze(0).to(self._device)     # (1, 3, H_lr, W_lr)

        with torch.no_grad():
            out = self._diffusion.p_sample_loop(
                y=y_t,
                model=self._model,
                first_stage_model=self._ae,
                noise=None,
                noise_repeat=False,
                clip_denoised=False,
                denoised_fn=None,
                model_kwargs={"lq": y_t},
                progress=False,
            )  # (1, 3, H_hr, W_hr) in [-1, 1]

        # Back to grayscale [0, 1]
        hr = out.squeeze(0).mean(dim=0)                    # (H_hr, W_hr)
        hr = ((hr + 1.0) / 2.0).clamp(0.0, 1.0)
        return hr.cpu().float().numpy().astype(np.float64)


# ---------------------------------------------------------------------------

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
