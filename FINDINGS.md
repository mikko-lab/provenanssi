# FINDINGS — Domain-shift investigation of calibration slope

Internal research memo. Not a finished result. Not marketing.

**Date:** 2026-06-22  
**Scripts:** `eval/calibrate_domain_shift.py`, `eval/stability_nscan.py`  
**Data:** `eval/domain_shift_results.txt`, `eval/stability_nscan_results.txt`  
**Commit:** aceaa94

---

## 1. QUESTION

When the three demo images were replaced with CC0 alternatives (grass, dirt, wood-grain
textures) prior to publication, the pooled calibration slope changed from **+1.5301**
(16 ImageNet patches, the certified result) to **+0.5533** (3 CC0 texture images) while
Pearson r stayed high in both cases (0.9667 vs 0.9968). Both numbers come from the same
model, operator, and calibration code; only the images differ.

**Starting question:** Is calibration slope systematically dependent on the image
distribution (a real, measurable effect), or was the change sampling noise from a small
per-group N?

**Secondary question:** If the effect is real, does it follow a clean monotonic ordering
with intuitive "distance from the ResShift training distribution" (ImageNet bicubic
pairs)?

---

## 2. METHOD

**Fixed infrastructure** (identical to falsify.py throughout; nothing re-tuned):
- Degradation operator: BicubicDownsample(4)
- Reconstruction model: ResShift (resshift_bicsrx4_s4, SHA-256 b160efec…)
- Ensemble: N=6 per image for the domain-shift calibration; N up to 48 for stability
- Preprocessing: resize shorter side to 256 (LANCZOS), centre-crop 256×256, BT.601 grayscale
- Calibration thresholds: MIN_R=0.9, MAX_ECE=0.3, slope ∈ [0.5, 2.0] — fixed, not touched
- Bins: 10 quantile bins; min_predicted_std=1e-6

**Experiment 1 — domain-shift calibration** (`eval/calibrate_domain_shift.py`):

Four intuitively-ordered image groups, all CC0 or public domain (sources in
`eval/research_sources/SOURCES_LICENCE.md`):

| Group | Images | Rationale |
|---|---|---|
| natural | 3 outdoor/wildlife scenes | Closest to ImageNet content |
| faces | 3 portrait photographs | Human faces; off-distribution for ImageNet SR |
| texture | 3 close-up surface textures | Same images as certified demo; repetitive fine-grain |
| synthetic | 4 programmatic images | Gradients, blobs, shapes; far from any natural image |

Each group calibrated with N=6 ensemble, pooled across images in the group.
"Distance from training distribution" is an **intuitive label only** — no FID, no
embedding-space distance computed. This ordering is a proxy, not a measurement.

**Experiment 2 — stability N-scan** (`eval/stability_nscan.py`):

Six images (2 natural, 3 faces, 1 texture) calibrated at N = 6, 12, 24, 48.  Seeds are
**nested**: N=6 uses seeds 0–5; N=12 adds seeds 6–11 to the same set; and so on. One
image (boardwalk_nature.jpg) was run to N=60 to support the noise-floor measurement.

**Noise floor** measured by taking 5 non-overlapping N=12 seed windows
(seeds 0–11, 12–23, 24–35, 36–47, 48–59) on one image (boardwalk_nature.jpg) and
computing slope for each window independently. The resulting spread is sampling
noise at N=12 on that image.

**Synthetic group excluded from N-scan.** Low-frequency images (linear and radial
gradients) have near-zero null-space energy after rectification, making the ensemble
std near-zero and the calibration regression degenerate. radial_gradient.png returned
r=0.877 (the one individual image to fall below the 0.90 threshold) and ECE=0.0005
with slope=0.464. This is a measurement-domain limitation — the calibration method
is not designed for content with negligible null-space occupation — not evidence of
domain shift.

**Hardware and runtime:**
- 300 total ResShift forward passes (N=48 × 6 images, plus 12 extra for boardwalk)
- MPS (Apple M-series), ~1.1s per pass
- Total: 321s (5.4 min)

---

## 3. RESULTS — ESTABLISHED

### 3a. Pearson r is robust to distribution

r ≥ 0.944 for every individual image in the domain-shift experiment; ≥ 0.963 pooled for
every group. It is stable across all N values in the N-scan (never drops below 0.944,
never varies by more than 0.04 across N=6→48 for any image). The one sub-0.90
individual value is radial_gradient.png (r=0.877, excluded from N-scan for the reason
stated above).

**What this means:** the uncertainty map's *ordering* — where ensemble spread is high,
reconstruction error is also high — is preserved regardless of whether the input is a
natural scene, a face photograph, or a surface texture. The map reliably identifies
WHERE the reconstruction is less trustworthy, across all tested domains.

### 3b. Slope varies between groups and the variation is above noise

**Pooled calibration results at N=6** (`eval/calibrate_domain_shift.py`):

| Group | N images | r (pooled) | slope | ECE | IS_CAL |
|---|---|---|---|---|---|
| natural | 3 | +0.9629 | 1.0782 | 0.0150 | YES |
| faces | 3 | +0.9944 | 2.1070 | 0.0174 | NO |
| texture | 3 | +0.9968 | 0.5533 | 0.0226 | YES |
| synthetic | 4 | +0.9978 | 7.1234 | 0.0063 | NO |
| *ImageNet ref (certified, commit 83ab9cd)* | *16* | *+0.9667* | *1.5301* | *0.0282* | *YES* |

**Between-group spread** at N=48 (N-scan results): natural mean 1.19, faces mean 2.57,
texture (wood_grain) 0.44 — total spread 2.14 slope units.

**Noise floor** (5 × N=12 independent seed windows, boardwalk_nature.jpg):

| Window | Seeds | slope |
|---|---|---|
| 0 | 0–11 | 0.9602 |
| 1 | 12–23 | 0.9556 |
| 2 | 24–35 | 0.9508 |
| 3 | 36–47 | 0.8932 |
| 4 | 48–59 | 0.9776 |

Mean: 0.9475 · Std: **0.0286** · Range: **0.0844**

Between-group spread (2.14) / noise-floor range (0.084) = **25×**.

The signal is ~25 times the measured noise floor. The slope differences between natural,
faces, and texture groups are not sampling noise.

**What this means:** the model's absolute uncertainty scale — how large the ensemble
spread is relative to actual reconstruction error — depends on the image domain. The
system's ordering of uncertain pixels is robust; its magnitude calibration is not
portable across distributions.

### 3c. N-scan: most slopes converge by N=24–48

Full N-scan (nested seeds, slopes only; see `eval/stability_nscan_results.txt` for r):

| Image | Group | N=6 | N=12 | N=24 | N=48 | \|Δ(48,24)\| |
|---|---|---|---|---|---|---|
| boardwalk_nature | natural | 0.9532 | 0.9602 | 0.9672 | 0.9510 | 0.0162 |
| frog_on_log | natural | 1.3085 | 1.3175 | 1.3870 | 1.4238 | 0.0368 |
| boy_face | faces | 3.1243 | 3.2046 | 3.1997 | 3.1829 | 0.0168 |
| girl_sad_face | faces | 3.0450 | 2.7936 | 2.7768 | 2.7834 | 0.0066 |
| wayuu_woman | faces | 1.4719 | 1.5941 | 1.7015 | 1.7483 | 0.0467 |
| wood_grain | texture | 0.1564 | 0.2645 | 0.3554 | 0.4351 | 0.0797 |

For four of six images, |Δ(48,24)| ≤ 0.05. The slope is essentially settled.

---

## 4. RESULTS — SHAKY OR REFUTED

### 4a. The monotonic hypothesis is refuted

The starting intuition was: slope decreases (toward 1) as images approach the training
distribution. The actual ordering by slope (N=6 pooled) is:

**texture 0.55 → natural 1.08 → faces 2.11 → synthetic 7.12**

The intuitive ordering by "distribution distance" would put texture between natural and
synthetic, not below natural. Texture has *lower* slope than natural, in the opposite
direction from faces and synthetic. The ordering by slope does not match the ordering by
intuitive distribution distance. The simple monotonic hypothesis is refuted by the data.

What the data does show is a split: the two groups that calibrate within the [0.5, 2.0]
IS_CALIBRATED window (natural, texture) vs the two that fall outside it (faces, synthetic).
But even this split does not follow a clean distance ordering, and the texture result is
itself uncertain (see §4b).

### 4b. The texture group result is uncertain — wood_grain has not converged

The texture group's low pooled slope (0.5533 at N=6) is driven largely by wood_grain,
which had slope=0.1564 at N=6. **This value has not converged at N=48.**

wood_grain slope trajectory: **0.1564 → 0.2645 → 0.3554 → 0.4351** (monotonically
rising, still moving at N=48, |Δ(48,24)| = 0.080). The trend adds approximately +0.09
per doubling of N. Extrapolating:

- N=96: ~0.52  
- N=192: ~0.61  
- Asymptote: unclear, but likely in the 0.5–0.6 range

This matters because the IS_CALIBRATED threshold is slope ≥ 0.5. At N=6 wood_grain
reads as severely underconfident (slope=0.16). At the extrapolated large-N asymptote it
may be close to or inside the calibrated window. **The claim that textures are
systematically underconfident is uncertain until wood_grain is rerun at N≥100.**

This directly qualifies the original observation that triggered the investigation. The
"CC0 demo images gave slope=0.55" finding reflects a real but N-dependent measurement.

The other two texture images (dirt_soil slope=0.5814, grass_meadow slope=0.2783 at N=6)
were not included in the N-scan and their convergence behaviour is unknown.

### 4c. The faces group is internally heterogeneous — wayuu_woman is a genuine outlier

Within-group slope spread at N=6: 1.4719 – 3.1243 (span 1.65).  
Within-group slope spread at N=48: 1.7483 – 3.1829 (span 1.43).

The span barely shrinks with N. wayuu_woman (slope=1.7483 at N=48) is 1.43 slope units
below the other two faces images (3.18 and 2.78) and the gap persists at N=48. This is
not sampling noise — it is a real image-level difference. "Faces" is a domain label,
not a calibration constant. Two images in the faces group have slope ~3; one has slope
~1.75. The faces group may contain qualitatively different sub-cases (close-up portrait
vs. wider shot, lighting, background complexity) that the group label does not capture.

### 4d. girl_sad_face had a large N=6 artifact

girl_sad_face N=6 slope = 3.0450, N=12 = 2.7936, N=24–48 ≈ 2.777–2.783.  The N=6 value
is 0.26 above the stable value. This is within the noise floor range (0.084) × ~3,
suggesting the N=6 reading for this image was a moderate sampling artifact. Any analysis
that uses N=6 slope as a precise measurement should treat it as having ±0.2–0.3
uncertainty for high-slope images.

---

## 5. OPEN / UNKNOWN

**No rigorous distribution-distance metric.** The "natural / faces / texture / synthetic"
ordering is intuitive. ResShift was trained on ImageNet bicubic pairs; images resembling
that content are "closer" by assumption. No FID, FD∞, CLIP distance, or embedding-space
metric was computed. Any claim of the form "slope tracks distribution distance" must
carry this qualification until a rigorous metric is substituted.

**Noise floor measured on one image only.** The 5-window noise floor (std=0.029,
range=0.084) was measured on boardwalk_nature.jpg, a natural-scene image with typical
null-space energy for this operator. Images with lower null-space energy (approaching
smooth content) will have a smaller effective null-space projection per pixel and
different per-sample variance. The 25× SNR is likely an upper bound for the natural
group and may be significantly lower for texture and synthetic groups, where null-space
occupation varies more. The noise floor should be characterised per-group before
claiming that between-group differences in those groups are fully above noise.

**Detection without ground truth — not attempted.** The logical next step after
establishing the slope/distribution effect would be to test whether domain shift can be
detected from slope alone, without knowing the ground-truth image. This step was
explicitly not pursued because two preconditions are not yet met:

1. The texture group's true (high-N) slope is unknown — it may collapse the texture/natural
   separation once wood_grain converges.
2. The noise floor is not characterised for groups other than natural.

Building a distribution-shift detector on the current data would be premature. The
signal needs to be characterised more completely before a detector makes sense.

**Per-image vs pooled stability.** The N-scan was run on one image per group (except
faces). Whether the per-group stability and convergence behaviour generalises to other
images in each group is untested. The domain-shift calibration experiment used N=6 for
all images; those numbers have the uncertainty implied by the N-scan results.

---

## 6. STATUS

The project's certified result — **r = +0.9667, slope = 1.5301, ECE = 0.0282 on 16
ImageNet images, BicubicDownsample(4) + ResShift, commit 83ab9cd** — is unchanged.
`python falsify.py --full` at that commit reproduces it. This is the only number that
should be stated as a finished, verified result in any public context.

The domain-shift investigation is a **promising but incomplete research thread**. What
is established: r is robust across distributions; slope varies between groups by ~25×
the measured noise floor; the variation is not sampling noise at the between-group level.
What is not established: the texture group's slope at converged N; the noise floor for
groups other than natural; whether the effect is detectable without ground truth. These
are not minor caveats — the texture non-convergence directly qualifies the observation
that started the investigation.

The domain-shift finding **should not be stated as a conclusion in any public post,
talk, or repository description** until at minimum: (a) wood_grain and the texture group
are rerun at N≥100 to confirm or revise the group slope; (b) the noise floor is measured
on at least one image from each group; and (c) the between-group spread at converged N
is confirmed to still exceed the revised noise floor. Until then, the honest
characterisation is: *"preliminary evidence that calibration slope is
distribution-dependent; investigation ongoing."*
