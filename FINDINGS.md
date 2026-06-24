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

---

## Update: closing the open threads (2026-06-22)

**Scripts:** `eval/close_findings.py`  
**Data:** `eval/close_findings_results.txt`  
**Commit:** (this commit)

This update closes the two threads flagged in §5 above. The results are mixed: one thread
closes cleanly, the other reveals a deeper problem.

---

### Thread 1 resolved: wood_grain slope convergence

Extended N-scan (N=48→96→144→192, seeds nested 0..N−1). Consistency check: N=48 slope
= 0.4351 matches the prior session exactly.

Full trajectory (N=6 through N=192):

**0.1564 → 0.2645 → 0.3554 → 0.4351 → 0.5237 → 0.5790 → 0.5969**

| N | slope | r | \|Δ\| | IS_CAL |
|---|---|---|---|---|
| 48 | 0.4351 | +0.9800 | — | NO |
| 96 | 0.5237 | +0.9794 | 0.0886 | **YES** |
| 144 | 0.5790 | +0.9794 | 0.0553 | YES |
| 192 | 0.5969 | +0.9834 | 0.0179 | YES |

The slope crosses IS_CALIBRATED (≥0.5) between N=48 and N=96. At N=192 the delta has
shrunk to 0.018; the slope is near-plateau. Extrapolating the decelerating trend, the
asymptote is approximately 0.62–0.65, comfortably inside the [0.5, 2.0] window.

**Verdict on texture group:** The characterisation from N=6 (slope=0.1564, "severely
underconfident") was substantially a sampling artefact. At converged N, wood_grain has
slope ~0.60 — inside the calibrated window. The texture group is not distinguished from
natural by being underconfident; the two groups overlap at high N. The natural↔texture
slope difference (~1.19 − 0.60 = 0.59 at converged N) is real but small. The §4b
uncertainty flag is resolved: **the texture-group result has been revised, not confirmed.**

---

### Thread 2 partially resolved: noise floor generalisation

Measured 5 × N=12 independent seed windows on three images spanning null-space energy
(null_frac_gt = ||(I−A⁺A)x||²/||x||², measured from the GT image and the operator).

| Image | null_frac_gt | slope mean | std | range |
|---|---|---|---|---|
| wood_grain | 0.0159 | 0.2323 | 0.0186 | **0.0522** |
| boardwalk_nature | 0.0046 | 0.9475 | 0.0286 | **0.0844** |
| soft_blobs | 0.0014 | 4.3726 | 0.3391 | **0.8992** |

Boardwalk replicates the prior session exactly (range=0.0844 both times). Consistency confirmed.

**Unexpected finding — noise floor scales inversely with null-space energy.**

The prior expectation was that images with less null-space content (smoother) would have
less ensemble variation and thus more stable slope estimates. The opposite is true. The
lowest-null-energy image (soft_blobs, null_frac_gt=0.0014) has the highest noise floor
by a large margin: range=0.899, which is **11× higher than boardwalk** and **17× higher
than wood_grain**.

Physical interpretation: images with low null-space energy produce tiny ensemble spread
(the model cannot vary much in the null space). The calibration regression is therefore
fitting a near-flat signal, and tiny fluctuations in how the 12 members sample that
signal drive large changes in the estimated slope. At high slope values (~4.37 for
soft_blobs), small changes in the regression geometry translate to large absolute slope
changes. The noise floor is dominated by the *weakness* of the null-space signal, not by
the magnitude of null-space variation.

This also explains why the wood_grain noise floor is *lower* than boardwalk despite
wood_grain having higher null energy: wood_grain's slope mean (~0.23 at N=12) is much
lower in absolute terms, keeping absolute slope variance small even though
null-space occupation is high.

**Revised SNR calculations:**

Between-group spread at N=48 (from prior session, unchanged): natural mean 1.1874, faces
mean 2.5715, texture (wood_grain at N=48) 0.4351, spread = 2.1364.

| Noise floor reference | range | Overall SNR | Natural↔Faces SNR |
|---|---|---|---|
| Prior (boardwalk only) | 0.0844 | 25.3× | 16.4× |
| Worst-case (soft_blobs) | 0.8992 | **2.4×** | **1.5×** |

Using the worst-case noise floor (as instructed), **all group contrasts fall below 3×
the noise floor**. This means the prior verdict (a) — "slope signal is measurable above
noise" — is **not supported** under worst-case assumptions.

**Why the worst-case is not directly applicable to the main contrast.**

The soft_blobs noise floor (range=0.899) comes from an image with null_frac_gt=0.0014
and slope ~4.37 — a qualitatively different regime from the natural group (slope ~1.19)
and faces group (slope ~2.57). Slope variance scales with slope magnitude; the
large noise floor for soft_blobs partly reflects that its slope is ~4× higher than
natural images. Applying a slope-~4.37 noise floor to a slope-~1.5–2.5 contrast is
conservative beyond what the regime warrants.

Using boardwalk (a natural image, null_frac_gt=0.0046, slope ~0.95) as the reference
for the natural side of the natural↔faces contrast, the SNR is 16×. But the noise
floor for *face images specifically* has not been measured. The face-group noise floor
remains an open variable. If face images have noise floors comparable to boardwalk
(likely, given similar null-frac_gt), the natural↔faces contrast is well above noise.
If they are anomalously higher, the contrast could be marginal.

**What the noise floor data does establish:** the synthetic group's results are
noise-dominated. soft_blobs has noise floor range=0.899 vs a slope of ~4.37 — a
coefficient of variation of ~21%. Any slope reported for smooth/low-null synthetic
images at N=6 or N=12 carries uncertainty comparable to its own magnitude. The
synthetic group results from the original domain-shift experiment should not be
treated as stable measurements.

---

### Revised verdict

**What is now established:**

1. r remains robust across all tested domains and all N values (unchanged from §3a).
2. wood_grain slope converges to ~0.60 at N≥100, inside the IS_CALIBRATED window.
   The "texture group is severely underconfident" claim from N=6 is retracted; texture
   and natural are closer than originally appeared at converged N.
3. The noise floor scales *inversely* with null-space energy. Low-null-energy images
   (smooth/synthetic) have much higher slope variance than textured or natural images.
   This is a novel finding about the calibration method's measurement properties.

**What is revised or withdrawn:**

- The 25× SNR from FINDINGS.md §3b was based on boardwalk as a universal reference.
  That was the wrong comparison image for the synthetic group. **Withdrawn.**
- The "texture group is underconfident" characterisation from N=6 data. **Revised to:**
  texture group converges into the calibrated window at high N; group is not clearly
  distinct from natural in slope direction.
- The original verdict (a): "slope signal above noise, domain shift is measurable."
  Under worst-case noise floor: **not supported**. Under same-regime reference
  (boardwalk for natural↔faces): still supported. The verdict depends on an unmeasured
  quantity — the face-group noise floor.

**Remaining open thread:**

Noise floor for the faces group has not been measured. This is the one measurement
needed to determine whether the natural↔faces contrast (the strongest remaining
signal, difference 1.38 at N=48) is above noise or not. Until measured: the contrast
is plausible (~16× using boardwalk as proxy) but unconfirmed. No public claim should
rest on it.

**Runtime:** 334s (5.6 min), 330 forward passes on MPS.

---

## Update 2: faces noise floor — final verdict (2026-06-22)

**Script:** `eval/faces_noise_floor.py`
**Data:** `eval/faces_noise_floor_results.txt`
**Runtime:** 788s (13.1 min), 720 forward passes on MPS.

This update closes the one remaining open thread: the faces group's own noise floor,
needed to determine whether the natural↔faces slope contrast is real or noise.

---

### Measurement: noise floor for all three face images

Protocol: 5 non-overlapping seed windows at N=12 (seeds 0–59) and N=48 (seeds 0–239),
240 members collected per image. null_frac_gt measured from GT image and operator.

**N=12 noise floor (5×N=12 windows):**

| Image | null_frac_gt | slope mean | std | range |
|---|---|---|---|---|
| boy_face | 0.0049 | 3.1776 | 0.0511 | **0.1497** |
| girl_sad_face | 0.0032 | 2.7961 | 0.0374 | **0.1051** |
| wayuu_woman | 0.0162 | 1.5848 | 0.0472 | **0.1359** |

**N=48 noise floor (5×N=48 windows):**

| Image | null_frac_gt | slope mean | std | range |
|---|---|---|---|---|
| boy_face | 0.0049 | 3.1381 | 0.0411 | **0.1153** |
| girl_sad_face | 0.0032 | 2.7410 | 0.0261 | **0.0822** |
| wayuu_woman | 0.0162 | 1.7296 | 0.0189 | **0.0482** |

---

### Faces on the energy→noise curve

Prior curve (from Update 1, at N=12):

| Image | null_frac_gt | slope mean | range |
|---|---|---|---|
| wood_grain | 0.0159 | 0.2323 | 0.0522 |
| boardwalk | 0.0046 | 0.9475 | 0.0844 |
| soft_blobs | 0.0014 | 4.3726 | 0.8992 |

Face images added:

| Image | null_frac_gt | slope mean | range |
|---|---|---|---|
| boy_face | 0.0049 | 3.1776 | **0.1497** |
| girl_sad_face | 0.0032 | 2.7961 | **0.1051** |
| wayuu_woman | 0.0162 | 1.5848 | **0.1359** |

**Unexpected finding: the inverse-energy relationship breaks at equal null energy but
different slope magnitude.** The prior data showed higher null → lower noise floor
(wood_grain vs soft_blobs). The faces group violates this simple picture:

- boy_face has null=0.0049 — nearly identical to boardwalk (0.0046) — but noise floor
  range=0.1497, **1.77× higher than boardwalk's 0.0844**. The only difference:
  boy_face slope is ~3.18 vs boardwalk's ~0.95.
- wayuu_woman has null=0.0162 — nearly identical to wood_grain (0.0159) — but noise
  floor range=0.1359 vs wood_grain's 0.0522. wayuu_woman slope is ~1.58 vs
  wood_grain's ~0.23.

The noise floor is not determined by null_frac_gt alone. Slope magnitude is a second
driver: at the same null energy, higher-slope images have larger absolute regression
variance. The energy→noise curve from Update 1 was confounded by slope covarying with
null energy across those three reference images. The faces data separates these factors
and shows both matter.

---

### Honest natural↔faces SNR

**Input quantities (N=48 group means from stability_nscan_results.txt, unchanged):**
- faces mean = 2.5715  (boy_face 3.1829 / girl_sad_face 2.7834 / wayuu_woman 1.7483)
- natural mean = 1.1874  (boardwalk 0.9510 / frog_on_log 1.4238)
- difference = **1.3841**

**Noise floor denominator** = max(natural N=12 range, worst face N=12 range)
= max(0.0844, 0.1497) = **0.1497** (boy_face, faces group)

**Honest SNR at N=12: 1.3841 / 0.1497 = 9.2×**

At N=48 windows: worst face range = 0.1153 (boy_face); natural (estimated by scaling
boardwalk N=12 × 0.5) ≈ 0.042; worst = 0.1153.

**Honest SNR at N=48: 1.3841 / 0.1153 = 12.0×**

Both values clear the 3× threshold comfortably. The faces noise floor is higher than
boardwalk (by ~1.5–1.8×), but the 1.38-unit contrast still clears the worst noise
floor by 9×.

---

### wayuu_woman overlap check

Natural group range at N=48: 0.9510 – 1.4238.
wayuu_woman at N=48 = 1.7483; separation from natural max = **0.3245**.

At N=12 noise floor:
- max(wayuu N=12 range=0.1359, boardwalk N=12 range=0.0844) = 0.1359
- SNR = 0.3245 / 0.1359 = **2.4×** — marginal, below 3× threshold

At N=48 noise floor:
- max(wayuu N=48 range=0.0482, boardwalk N=48 estimated range≈0.042) = 0.0482
- SNR = 0.3245 / 0.0482 = **6.7×** — robust

**wayuu_woman is not clearly separable from the natural range at N=12.** The separation
becomes robust only at N=48. boy_face and girl_sad_face (slopes 3.18, 2.78) are
separated from the natural maximum (1.42) by 1.76 and 1.36 slope units respectively —
both clear the noise floor by >10× at N=12.

---

### Final verdict on domain-shift-in-slope

**(a) SURVIVES.**

The natural↔faces slope difference (1.3841 at N=48) is **9.2×** the worst
in-contrast noise floor (0.1497, boy_face at N=12). At N=48, the SNR is 12.0×. The
signal survives on the honest measurement — using the faces group's own noise floor,
not boardwalk as a proxy.

**Caveats that stand:**

1. The contrast is heterogeneous: two of three faces images (boy_face, girl_sad_face)
   drive the group elevation; wayuu_woman (slope=1.75) is only marginally above the
   natural max at N=12, becoming clearly separable at N=48.

2. "Faces" is a domain label. The within-group slope range (1.75–3.18 at N=48) is
   larger than the natural group's full range (0.95–1.42). A single slope threshold
   does not characterise the faces group cleanly.

3. The natural↔faces contrast is the strongest domain-shift signal in the data. Other
   contrasts (natural↔texture at converged N, faces↔texture) are weaker and have not
   been separately characterised against per-group noise floors.

4. "Distance from training distribution" remains an intuitive proxy. No FID or
   embedding-space metric was computed.

**The certified result** (r=+0.9667, slope=1.5301, ECE=0.0282, 16 ImageNet images,
commit 83ab9cd) is unchanged and remains the only number that should appear in any
public-facing context.

**The domain-shift investigation status:** the natural↔faces slope elevation is
established above noise and is a real effect. It is not a finished, characterised
result — the mechanism is unknown, the grouping is intuitive, and detection without
ground truth has not been attempted. The honest summary remains: *"preliminary evidence
that calibration slope is distribution-dependent; natural↔faces contrast is 9× above
the measured noise floor."*

---

## Update 3: Distance-metric thread (§11 item 1)

**Date:** 2026-06-23  
**Script:** `eval/distance_metric.py`  
**Data:** `eval/distance_metric_results.txt`

**Thread question:** Can a measured per-image feature-space distance from the
ImageNet/ResShift training distribution replace the intuitive domain groupings and
predict calibration slope?

---

### Method

**Metric (precisely named):** Cosine distance in ResShift VQ-autoencoder
(autoencoder_vq_f4) pre-quantization encoder latent space, to the L2-normalized
centroid of 16 ILSVRC2012 validation image encodings.

Preprocessing for all images (both reference and eval): BT.601 grayscale
(0.299R + 0.587G + 0.114B), channel-replicated to pseudo-RGB, normalized to
[-1, 1]. This is identical to the ResShift calibration preprocessing.

**Reference set:** 16 ILSVRC2012 validation images (`vendor/ResShift/testdata/Bicubicx4/gt/`), 256×256 RGB.

**Eval set:** 13 images across 4 intuitive groups (natural, faces, texture, synthetic).

**n=13 is statistically thin.** Fisher-z 95% CI width ≈ ±0.6 in r at this sample
size. All correlations below are PRELIMINARY. This section reports a measured null
result, not a confirmed finding.

**Limitations (stated alongside any claimed association):**

- n_ref=16: centroid is a noisy estimate. 16 images poorly cover the ImageNet
  manifold; the centroid is not a reliable proxy for "the training distribution
  centroid."
- Grayscale conversion drops all color. Chromatic ImageNet texture features cannot
  contribute to the distance.
- VQ-AE was trained for reconstruction, not as a metric space. Pre-quantization
  continuous codes have no guaranteed metric properties.
- Calling this "distance from the training distribution" would be inaccurate.
  The precise claim is: distance to the centroid of 16 reference images in one
  particular encoder's feature space.

---

### Results: distance ranking

All 13 eval images, sorted by cosine distance (ascending = closest to centroid):

```
rank  image           group        dist    slope   null_frac
   1  hard_shapes     synthetic   0.7565   8.018    0.0426
   2  soft_blobs      synthetic   0.7780   4.292    0.0014
   3  linear_grad     synthetic   0.8640   6.530    0.0059
   4  nature_land     natural     0.8837   1.115    0.0128
   5  radial_grad     synthetic   0.9196   0.638    0.0000
   6  boy_face        faces       0.9318   3.183    0.0049
   7  boardwalk       natural     0.9383   0.951    0.0046
   8  grass_meadow    texture     0.9390   0.578    0.0477
   9  frog_on_log     natural     0.9413   1.424    0.0111
  10  wayuu_woman     faces       0.9477   1.748    0.0162
  11  girl_sad        faces       0.9620   2.783    0.0032
  12  wood_grain      texture     0.9670   0.597    0.0159
  13  dirt_soil       texture     0.9968   0.854    0.0088
```

For reference, within the 16 ImageNet reference images themselves, cosine distance
to the centroid ranges 0.6463–0.8702 (mean 0.7275).

**Unexpected ordering:** The synthetic images (hard_shapes, soft_blobs, linear_grad)
rank at the bottom of the distance table — i.e., they are *closer* to the ImageNet
centroid in VQ-encoder space than natural or texture images. This is the opposite of
the intuitive expectation. Simple patterns (blobs, shapes, gradients) apparently
project to latent codes that resemble the "average" of diverse ImageNet images; the
VQ-encoder centroid of varied ImageNet images may itself be a blurred, low-frequency
representation that simple synthetics happen to match.

This ordering failure means the metric does not capture what we wanted to measure.

---

### Correlation results

**Slopes used:** N=48 for all images except wood_grain (N=192, converged). Seven
images ran N=48 calibration in this script (nature_land, dirt_soil, grass_meadow,
soft_blobs, hard_shapes, linear_grad, radial_grad). Synthetic N=48 slopes carry
high noise (soft_blobs noise floor range≈0.90 at N=12 windows) and should be read
with that caveat.

All 13 images:

```
Pearson r(dist, slope)      = -0.785   95% CI [-0.933, -0.412]   p=0.001
Spearman rho(dist, slope)   = -0.582   p=0.037
Pearson r(null, slope)      = +0.178   (null_frac_gt confounder)
Partial r(dist, slope|null) = -0.777   95% CI [-0.934, -0.366]   (approx, n-4 df)
```

Non-synthetic only (n=9):

```
Pearson r(dist, slope)      = -0.087   95% CI [-0.710, +0.613]   p=0.824
Spearman rho(dist, slope)   = -0.250   p=0.516
Pearson r(null, slope)      = -0.513   (null_frac_gt confounder)
Partial r(dist, slope|null) = -0.163   95% CI [-0.778, +0.612]   (approx, n-4 df)
```

---

### Interpretation

The negative correlation in the n=13 sample (r=-0.785, p=0.001) is entirely
driven by the synthetic group. The scatter table makes this visible: three of the
four lowest-distance images are synthetic, and three of the four highest-slope
images are synthetic. When the synthetic group is removed (n=9), the correlation
collapses to r=-0.087 (CI includes 0 by a wide margin).

This is not a confound via null_frac_gt — the partial correlation r(dist,
slope|null) = -0.777 is nearly identical to the raw r=-0.785, so null energy does
not explain the association. The association is structural: the VQ-encoder
happens to place synthetic images closer to the centroid AND those images have
higher slopes. The two facts are independent artifacts of the feature space and
the calibration dynamics respectively.

**null_frac_gt is also not a useful predictor of slope** in this sample
(r=+0.178 for n=13, r=-0.513 for n=9). The sign flip across splits confirms it
is noise, not signal, at these sample sizes.

---

### Verdict

**(b) NULL.** The VQ-encoder cosine distance to a 16-image ImageNet centroid is
not a valid distribution-distance metric for this purpose.

The specific failure: the metric puts synthetic images closer to the ImageNet
centroid than natural images — opposite of the intuitive ordering. Without the
synthetic group, no correlation with slope exists (r=-0.087, CI [-0.710, +0.613]).

The §11 item 1 research question (rigorous distribution-distance metric) remains
open. This attempt rules out one specific metric. A valid metric would need to
produce a distance ordering where synthetic < texture < natural (or similar), and
this metric does not.

**What this does not rule out:** a different feature space (e.g., a classifier's
penultimate layer, or FID against a larger reference set) might produce the correct
ordering and might predict slope. That remains unresolved.

---

## Update 4: Distance-metric thread, attempt 2 (2026-06-23)

### Setup

Metric: ResNet50 (IMAGENET1K_V2) penultimate-layer features (2048-dim global-average-
pool after layer4), cosine distance to a centroid of 16 ILSVRC2012 validation images.
Weights downloaded from PyTorch hub (resnet50-11ad3fa6.pth, 98 MB). Self-contained
implementation — no torchvision dependency; layer names match official checkpoint exactly.

This addresses both failures from attempt 1: (i) semantic features instead of a VQ
codebook that measured compressibility, and (ii) a larger, continuous sample instead
of n=13 with a bimodal gap.

**Sample composition (n=23):** 13 existing images (3 natural, 3 faces, 3 texture,
4 synthetic-prior) + 2 CC0 natural landscapes downloaded from Wikimedia Commons
(nat_landscape2, nat_landscape3; Wikimedia rate-limited after the first two downloads,
blocking the planned 30–40 image target) + 8 programmatic synthetics generated in
code (3 Gaussian noise levels, 1 pink/1/f noise, 2 checkerboards, 2 stripe patterns).

This is fewer than the 30–40 target. Non-synthetic n=11 (3 faces + 5 natural + 3 texture).

### Sanity check (required gate before analysis)

Before correlation analysis, the metric must confirm that obvious synthetics score
FAR and an obvious natural photo scores NEAR:

| image            | group     | dist   |
|------------------|-----------|--------|
| boardwalk        | natural   | 0.7858 |
| linear_gradient  | synthetic | 0.9243 |
| hard_shapes      | synthetic | 0.8738 |
| noise_gauss50    | noise     | 0.9154 |

linear_d (0.9243) > boardwalk_d (0.7858) ✓  
hard_d (0.8738) > boardwalk_d (0.7858) ✓

**SANITY CHECK PASSED.** Analysis proceeds.

Within-reference cosine distances: mean=0.630, range [0.470, 0.780].

### Data table (sorted by distance, ascending = near ImageNet)

| rk | image          | group         |  dist  | slope  | null_f  | N  |
|----|----------------|---------------|--------|--------|---------|----|
|  1 | wayuu_woman    | faces         | 0.7203 | 1.7483 | 0.0162  | 48 |
|  2 | girl_sad       | faces         | 0.7365 | 2.7834 | 0.0032  | 48 |
|  3 | boardwalk      | natural       | 0.7858 | 0.9510 | 0.0046  | 48 |
|  4 | checker32      | synthetic_gen | 0.7891 | 5.4914 | 0.0602  | 48 |
|  5 | boy_face       | faces         | 0.8018 | 3.1829 | 0.0049  | 48 |
|  6 | nature_land    | natural       | 0.8023 | 1.1147 | 0.0128  | 48 |
|  7 | nat_landscape2 | natural       | 0.8041 | 1.2053 | 0.0181  | 48 |
|  8 | nat_landscape3 | natural       | 0.8177 | 1.0313 | 0.0120  | 48 |
|  9 | frog_on_log    | natural       | 0.8663 | 1.4238 | 0.0111  | 48 |
| 10 | hard_shapes    | synthetic     | 0.8738 | 8.0176 | 0.0426  | 48 |
| 11 | stripes_d      | synthetic_gen | 0.8821 | 3.0791 | 0.0001  | 48 |
| 12 | soft_blobs     | synthetic     | 0.8890 | 4.2917 | 0.0014  | 48 |
| 13 | radial_grad    | synthetic     | 0.9075 | 0.6376 | 0.0000  | 48 |
| 14 | noise_gauss10  | synthetic_gen | 0.9097 | 0.0620 | 0.0359  | 48 |
| 15 | dirt_soil      | texture       | 0.9099 | 0.8536 | 0.0088  | 48 |
| 16 | grass_meadow   | texture       | 0.9140 | 0.5778 | 0.0477  | 48 |
| 17 | noise_gauss50  | synthetic_gen | 0.9154 | 0.0793 | 0.3194  | 48 |
| 18 | linear_grad    | synthetic     | 0.9243 | 6.5295 | 0.0059  | 48 |
| 19 | stripes_h      | synthetic_gen | 0.9246 | 5.0579 | 0.0001  | 48 |
| 20 | noise_gauss05  | synthetic_gen | 0.9246 | 0.0762 | 0.0093  | 48 |
| 21 | pink_noise     | synthetic_gen | 0.9273 | 0.1279 | 0.0172  | 48 |
| 22 | checker8       | synthetic_gen | 0.9289 | 0.0329 | 0.1881  | 48 |
| 23 | wood_grain     | texture       | 0.9483 | 0.5969 | 0.0159  | 192|

Slope = ResShift + BicubicDownsample(4), N=48 measurements (N=192 for wood_grain).  
null_f = null_frac_gt (fraction of augmented crops with energy > original).  
Rows without asterisk are non-synthetic (groups: natural, faces, texture).

### Correlation results

Fisher-z 95% CI throughout. All findings preliminary.

**All images (n=23):**
- Pearson r(dist, slope) = −0.127, 95% CI [−0.512, +0.301], p=0.564
- Spearman ρ(dist, slope) = −0.478, p=0.021
- Partial r(dist, slope | null_frac_gt) = −0.077, CI [−0.483, +0.356]

**Non-synthetic only (n=11) — KEY RESULT:**
- Pearson r(dist, slope) = −0.618, 95% CI [−0.889, −0.029], p=0.043
- Spearman ρ(dist, slope) = −0.736, p=0.010
- Partial r(dist, slope | null_frac_gt) = −0.512, CI [−0.863, +0.174]

Note on direction: negative r means higher distance (farther from ImageNet) → lower
calibration slope. Equivalently, images closer to the ImageNet distribution (faces,
natural photos) have higher slopes than images far from it (textures).

### Interpretation

The Pearson CI for non-synthetic images barely excludes 0 (upper bound −0.029).
The Spearman ρ=−0.736 (p=0.010) is more robust and suggests a consistent
monotonic pattern. However, several caveats apply:

1. **Group-level, not continuous.** Non-synthetic images form three tight clusters:
   faces (dist 0.72–0.80, slope 1.75–3.18), natural photos (0.80–0.87, slope 0.95–1.42),
   texture (0.91–0.95, slope 0.58–0.85). With only 3 groups and n=3–5 per group, the
   observed correlation largely reflects group identity rather than a continuous
   distance effect. A per-group one-way ANOVA would be the more appropriate test.

2. **Partial r CI includes 0.** After controlling for null_frac_gt, partial r=−0.512
   but CI=[−0.863, +0.174]. The confidence interval includes 0, meaning the partial
   correlation result is not reliably distinguished from zero at this sample size.
   The automated verdict code checked only |partial_r| > 0.2 but not whether its CI
   excludes 0 — an oversight in the pre-specified logic.

3. **n=11 is marginal.** With 11 observations, SE(z) ≈ 0.33 in Fisher-z space,
   giving CI half-width ≈ ±0.65 in r. The Pearson upper bound of −0.029 is barely
   below 0. Small leverage effects (e.g., the single boardwalk point, which is close
   to ImageNet but has an unusually low slope for a natural photo) may be influential.

4. **Synthetic images are incoherent with the metric.** The generated synthetics
   scatter widely in both slope (0.03 to 5.49) and distance (0.79 to 0.93), with no
   interpretable pattern. Their inclusion destroys the all-images correlation (r=−0.127)
   relative to the non-synthetic result (r=−0.618).

### Verdict

**(a-weak): BORDERLINE POSITIVE — real signal, but group-level and n=11.**

The automated code issued "(a) POSITIVE" by the pre-specified logic (non-synthetic
CI excludes 0 AND |partial_r| > 0.2). On the numbers alone that verdict is defensible:
the Spearman ρ=−0.736, p=0.010 is not a fluke of one data point, and the direction
is semantically coherent (ResShift trained on ImageNet-like data applies stronger
correction to in-distribution images).

However, the claim that a *continuous* distance metric predicts slope is not supported.
The signal is group-level: faces cluster near ImageNet with high slopes; textures
cluster far with low slopes; naturals sit in between. Within any single group the
relationship is unclear. The partial r CI includes 0 after null-energy control.
n=11 with three natural clusters provides limited power for a continuous claim.

**Practical implication:** image domain (faces / natural photos / textures /
synthetic patterns) appears to predict calibration slope more reliably than a
continuous distance number. This is actionable without requiring a distance metric.

**Status of §11 item 1:** The attempt-2 metric (ResNet50 semantic features)
passes the sanity check and produces a borderline significant correlation in the
non-synthetic split. The §11 constraint ("do not extend to distribution-dependent
calibration without a rigorous distance metric") can be relaxed for group-level
observations (e.g., "faces have higher slopes than textures"), but a continuous
distance-based calibration adjustment remains unsupported at this evidence level.

---

## Update 5: Distance-metric thread — SHELVED (2026-06-23)

**Status:** Shelved at a known state. No data or code deleted. Resume recipe below.

**Three attempts, in order:**

1. **Intuitive grouping** (natural / faces / texture / synthetic): Established that
   slope varies between groups above noise (natural↔faces 9.2× noise floor). The grouping
   is an intuitive proxy, not a metric. Result: (a) SURVIVES — group-level only.

2. **VQ-encoder cosine distance** (ResShift autoencoder_vq_f4 pre-quantisation latents):
   Metric places synthetic images CLOSER to the ImageNet centroid than natural images —
   opposite of the expected ordering. Correlation with slope is driven entirely by the
   synthetic group; non-synthetic r=−0.087. Result: (b) NULL — metric measures
   compressibility, not distribution membership.

3. **ResNet50 penultimate-layer features** (2048-dim, IMAGENET1K_V2): Sanity check passes
   (synthetics score far, naturals near). Non-synthetic n=11: Spearman ρ=−0.736 p=0.010,
   Pearson r=−0.618 CI [−0.889, −0.029] (barely excludes 0). Partial r controlling
   null_frac_gt = −0.512, CI [−0.863, +0.174] — includes 0. All-images (n=23): r=−0.127,
   no signal. Result: (a-weak) BORDERLINE — group-level signal, not continuous distance
   effect.

**Why this thread is shelved, not closed:**
The bottleneck is statistical power, not the metric. Non-synthetic images form three
tight clusters (faces / natural / texture), n=3–5 per cluster. With n=11 total, any
observed correlation largely reflects group identity. A continuous distance effect
cannot be separated from a group-level effect at this sample size (Fisher-z CI
half-width ≈ ±0.65 in r at n=11).

**Resume recipe:**
- Same metric: ResNet50 penultimate-layer cosine distance to 16-image ImageNet centroid
  (script: `eval/distance_metric_v2.py`, data: `eval/distance_metric_v2_results.txt`)
- Sample: ~60–100 CC0/PD images, balanced across the distance axis, with 10–15 images
  per distance band (not per intuitive group), so within-band and between-band effects
  can be separated
- Same calibration protocol: N=48 ResShift passes, 5×N=12 noise-floor windows per image
- Same analysis: Spearman ρ + partial r + power law fit for the distance→slope relationship
- Estimated runtime: 48 passes × 100 images × 1.1s ≈ 1.5h on MPS
- This is research-paper scope, not a session task. Do not start without a dedicated block.

---

## Update 6: Slope → noise-floor mechanism (2026-06-23)

**Script:** `eval/slope_noise_mechanism.py`
**Data:** `eval/slope_noise_mechanism_results.txt`
**Input:** all existing per-window slope data — no new GPU passes.

**Question:** Is the per-image slope noise floor driven by slope magnitude as an OLS
estimator artifact, or is it a model/data property?

---

### Analytic derivation

The calibration slope β̂ is the OLS slope of binned (predicted_std, actual_error).
Working from the regression formula with heteroscedastic bin variances:

- `x_out_i = A⁺y + (I−A⁺A)x̂_i`: the range part A⁺y is fixed; only the null-space
  component varies across seeds.
- For null-space pixels: `actual_err[p] ≈ |bias[p] + δ[p]|` where `bias[p]` is the
  irreducible model error and `δ[p] ~ N(0, pred_std²[p]/N)` is sampling noise.
- In the calibrated regime (bias >> δ): `Var_windows[y_k] ≈ x_k²/(n_eff·N)`.

Under the **iid pixel assumption** (independent actual_err values within a bin):

```
Var(β̂) = Σ_k c_k² · x_k²/(n_eff·N)    where c_k = (x_k−x̄)/Σ(x_j−x̄)²
```

**Key algebraic property — scale invariance:** if all x_k → s·x_k (same distribution
shape, rescaled), then c_k → c_k/s and Var(y_k) → s²·Var(y_k), leaving Var(β̂) unchanged.
The OLS slope variance is β-INDEPENDENT under iid pixels. Prediction: **α = 0**.

If α > 0 empirically, the iid assumption is violated — pixel reconstruction errors are
spatially correlated within bins. The free parameter is n_eff < n_k (effective independent
pixels per bin). If n_eff varies with image type, a slope→noise correlation emerges as a
**model/data effect**, not an estimator artifact.

---

### Empirical test (existing data: 6 images, 5×N=12 windows each)

Power law fit `std(β̂) ≈ a · slope^α` in log-log space (OLS):

| Fit | n | α̂ | SE | t-CI (95%) | α=0 excluded? |
|-----|---|----|----|------------|---------------|
| All images | 6 | 0.69 | 0.316 | [−0.19, 1.56] | NO |
| Excl. soft_blobs | 5 | 0.36 | 0.089 | [0.07, 0.64] | YES (p<0.05) |

The full fit includes 0 because soft_blobs is a high-leverage outlier (separate
mechanism, see below). The restricted fit (n=5) excludes 0 at p<0.05 (df=3, t_{0.025}=3.182).

**Matched-pair evidence (boardwalk vs. boy_face — nearly identical null_frac ≈ 0.0046):**

| | boardwalk | boy_face | ratio |
|--|-----------|----------|-------|
| slope | 0.95 | 3.18 | 3.35× |
| std(β̂) | 0.0286 | 0.0511 | 1.78× |

Under α=0: expected std ratio = 1.00×. Under α=1: expected = 3.35×. Under α=0.5:
expected = √3.35 = 1.83× — **matches actual 1.78×**.

**N-scaling (SD at N=48 vs. N=12 for three face images):**
Under iid, SD ∝ 1/√N → SD(N=48)/SD(N=12) = 0.50. Observed ratios: 0.40–0.80.
With 5-window std estimates (SE ≈ std/2.8), these are noisy but not contradictory.

**soft_blobs as a separate regime:** null_frac_gt=0.0014 — near-zero null-space energy.
The calibration bins have tiny x_k; fitting a near-flat signal makes slope estimates
ill-conditioned. std=0.339 (11× boardwalk) is a low-signal estimator artifact: when
x_k·√n_k is small, even small fluctuations in y_k drive large slope changes. This is
consistent with the iid model (Var(y_k) ∝ x_k²/(n_k·N) explodes when x_k→0) and
is NOT driven by slope magnitude.

---

### Verdict

**(b) MODEL/DATA EFFECT — sub-proportional (α ≈ 0.35–0.5).**

- The OLS estimator predicts α=0 under iid pixels. This is not approximate — it is a
  scale-invariance property of the regression formula. Any α>0 must come from the data.
- Empirical evidence: α=0 excluded at p<0.05 (restricted fit); boardwalk/boy_face pair
  consistent with α≈0.5.
- Mechanism: pixel reconstruction errors are spatially correlated within ensemble windows.
  The effective sample size n_eff < n_k. Images where ResShift generates spatially coherent
  hallucinations (faces) have lower n_eff → higher slope noise floor at fixed N.
- soft_blobs is a SEPARATE estimator artifact (ill-conditioning, iid-consistent).

**α ≈ 1 (proportional) is ruled out** by the boardwalk/boy_face matched pair (1.78× actual
vs. 3.35× predicted). The effect is sub-proportional.

---

### Implication for §10 domain-shift SNR — FLAG (not self-edited)

**The §10 9.2× SNR verdict is UNCHANGED and, if anything, is conservative.**

The SNR calculation already used the FACES group's own noise floor (boy_face, range=0.1497)
as the denominator. This is the higher-slope side of the contrast. If the faces noise floor
is partly inflated by slope magnitude (model effect), the denominator is inflated, making
the 9.2× a lower bound rather than a biased overestimate.

Under a slope-normalised noise measure, the natural↔faces SNR would be HIGHER than 9.2×.
The §10 verdict does not need revision. This finding does not require editing §10.

---

## Update 7 — Coherence mechanism: direct test

**Date:** 2026-06-23  
**Script:** `eval/spatial_coherence.py`  
**Data:** `eval/spatial_coherence_results.txt`  
**GPU passes:** 84 (7 images × N=12, ~96s on MPS)

**Question:** Update 6 INFERRED that high-slope images produce more spatially correlated
null-space hallucinations (lower n_eff → higher noise floor). This update tests that
inference DIRECTLY by measuring the spatial correlation of seed-to-seed null-space
deviations without using any noise-floor or slope machinery.

---

### Metric definition and independence argument

**Metric — rho_nn (primary):**

For each ensemble of N=12 members:
1. null_dev_i[p] = (I−A⁺A)x̂_i[p] − mean_seeds[(I−A⁺A)x̂_j[p]]
2. null_dev_norm_i[p] = null_dev_i[p] / std_seeds[(I−A⁺A)x̂_j[p]]
3. Compute 2D spatial ACF of null_dev_norm_i via zero-padded FFT
4. Average over members → mean radial ACF profile ρ(r)
5. **rho_nn = ρ(1)** (normalized ACF at nearest-neighbor lag r=1)

By the interchangeability of expectations:

```
mean_members[ ACF_spatial(null_dev_norm_i, r=1) ]
= E_pixels[ Cov_seeds(null_i[p], null_i[p+1]) / (std[p] · std[p+1]) ]
```

This IS the average across-seed correlation between neighboring pixels — the
direct quantity that enters the n_eff formula: `n_eff ≈ n / (1 + N_full(1)·ρ_nn)`.

**Why independent of the noise-floor calculation:**

| | Noise floor (Update 6) | Coherence (Update 7) |
|--|--|--|
| Protocol | 5 independent N=12 windows | 1 N=12 ensemble |
| Statistic | std of 5 scalar OLS slopes | mean 2D spatial ACF |
| Uses x_gt? | YES (actual_error = \|x_out − x_gt\|) | NO |
| Uses binning? | YES (10 pred_std quantile bins) | NO |
| Computable from N=1? | NO | YES |

Shared: the same operator, image, and N=12 ensemble members. Non-shared: every
step of the computation is structurally incomparable.

---

### Sanity check

noise_gauss50 (Gaussian white noise input): **rho_nn = −0.0022 ± 0.014** — indistinguishable
from zero. Null-space deviations for a noise input are spatially uncorrelated, as expected.
Sanity check: PASS.

---

### Measurements (N=12 per image, all existing images)

| Image | slope | null_frac | nf_std | rho_nn | n_eff/n (r=1 only) |
|-------|-------|-----------|--------|--------|---------------------|
| wood_grain | 0.232 | 0.016 | 0.019 | **0.098** | 0.56 |
| boardwalk | 0.948 | 0.005 | 0.029 | **0.161** | 0.44 |
| wayuu_woman | 1.585 | 0.016 | 0.047 | **0.245** | 0.34 |
| girl_sad_face | 2.796 | 0.003 | 0.037 | **0.154** | 0.44 |
| boy_face | 3.178 | 0.005 | 0.051 | **0.219** | 0.36 |
| soft_blobs | 4.373 | 0.001 | 0.339 | 0.014 | 0.99 |

n_eff/n (nearest-neighbor only) = 1/(1 + 8·rho_nn). This is the reduction from
nearest-neighbor correlation alone; higher-lag anti-correlation (ρ(2..5) < 0 for all
images) partially cancels it, so the true n_eff/n is closer to 1 than shown.

**Ordering:** rho_nn mostly increases with slope — wood_grain < boardwalk < {wayuu_woman, boy_face} — with one anomaly: girl_sad_face has lower rho_nn (0.154) than wayuu_woman (0.245) despite higher slope (2.80 vs 1.58). This weakens but does not eliminate the coherence-slope link.

**soft_blobs** (degenerate regime): rho_nn ≈ 0 despite highest slope. Consistent
with Update 6 — soft_blobs is an ill-conditioning artifact, not a coherence effect.

---

### Correlation analysis (excl. soft_blobs, n=5)

| Correlation | r | 95% CI | CI excl. 0? |
|-------------|---|--------|-------------|
| r(rho_nn, slope) | +0.549 | [−0.927, +0.994] | NO |
| r(rho_nn, nf_std) | +0.925 | [−0.555, +0.999] | NO |
| r(slope, nf_std) | +0.814 | [−0.805, +0.998] | NO |
| partial r(slope, nf_std \| rho_nn) | +0.964 | [−0.244, +1.000] | NO |

All CIs span most of [−1, +1] at n=5. No individual correlation is clearly
distinguished from zero. The **trends** are consistent with the mechanism:
positive rho_nn, positive rho_nn-slope association, but no mediation detected.

**gamma_nn (secondary metric, first-5-ring ACF integral):** shows NEGATIVE correlation
with slope (r=−0.804), driven by strong anti-correlation at lags r=2–5 for face images.
This is a property of the high-pass null-space domain (ACF oscillates), not a reversal
of the mechanism. The anti-correlations at r=2–5 partially cancel the positive rho_nn
contribution to n_eff, making the true n_eff effect smaller than rho_nn alone suggests.

---

### Verdict

**(a-weak) PARTIALLY SUPPORTED — consistent with mechanism, not directly confirmed.**

- rho_nn is higher for face images (0.22–0.24) than texture images (0.10–0.16) and near-zero
  for Gaussian noise (sanity baseline −0.002). The direction is right.
- The correlation r(rho_nn, slope) = +0.549 at n=5 is positive but CI is too wide to exclude
  zero. A sample of ≥20 images is needed for a definitive test.
- Mediation is not detected (partial r not attenuated), possibly because rho_nn itself is
  a noisy measure at N=12 and n=5 images.
- girl_sad_face anomaly (lower rho_nn than wayuu_woman despite higher slope) shows the
  relationship is not clean.

**Implication for Update 6:** The causal language in Update 6 ("images where ResShift
generates spatially coherent hallucinations have lower n_eff") is consistent with but NOT
directly confirmed by Update 7. That language should be read as "hypothesised mechanism,
consistent with direct measurement" — not as "confirmed". Update 6's empirical α≈0.35 fact
(slope predicts noise floor above noise floor) is unaffected by this finding.

**Update 6 status:** RETAIN AS-IS. No softening needed beyond noting Update 7 exists.
The mechanism remains "inferred and consistent", not "confirmed".

---

## Update 8 — Large-sample resolution (n=24, pre-registered, 2026-06-24)

**Date:** 2026-06-24  
**Scripts:** `eval/phase1_assemble.py`, `eval/phase2_analysis.py`  
**Pre-registration commit:** 8b5e68b (locked before any Phase 2 data collected)  
**Results:** `eval/phase2_results.txt`  
**GPU passes:** ~2370 (~44 min on MPS)

---

### Sample

26 non-synthetic images assembled (Phase 1, commit f7bc949). After applying pre-stated
exclusion criteria (calibration r < 0.90):

- **Excluded:** `texture_brick` (cal_r=0.59) and `texture_stone` (cal_r=0.83). Both
  uniform close-up texture crops; the predicted variance does not track actual error,
  suggesting near-zero null-space variation for these images.
- **Active n=24:** 5 faces, 5 paintings, 9 naturals, 5 textures.

ResNet50 distance range (non-synthetic, n=24): [0.72, 0.94]. No bimodality.

---

### Thread 1 — Distance → slope (continuous relationship)

| Test | Metric | Value | CI [95%] | n | Verdict |
|------|--------|-------|----------|---|---------|
| A (Pearson, all) | r(dist, slope) | −0.689 | [−0.855, −0.396] | 24 | CI excl. 0 |
| B (no paintings) | r(dist, slope) | −0.643 | [−0.849, −0.266] | 19 | CI excl. 0 |
| C (partial \| null_frac) | r(dist, slope \| null) | −0.663 | [−0.844, −0.345] | 24 | CI excl. 0 |
| D (Spearman) | ρ(dist, slope) | −0.693 | — | 24 | p<0.001 |

**Verdict: (a) POSITIVE** — pre-stated decision rule satisfied (r<0 and CI excludes 0 in
both Tests A AND C). Effect is robust: removing paintings (Test B) does not weaken the
correlation, and controlling for null_frac_gt (Test C) does not eliminate it.

**Within-group r (Test E):**

| Category | n | r(dist, slope) | CI [95%] | Significant? |
|----------|---|----------------|----------|--------------|
| faces | 5 | +0.438 | [−0.724, +0.952] | No |
| natural | 9 | +0.136 | [−0.581, +0.734] | No |
| **painting** | **5** | **−0.930** | **[−0.995, −0.263]** | **Yes (p=0.022)** |
| texture | 5 | −0.570 | [−0.966, +0.628] | No |

The only within-group correlation is in paintings: higher distance → lower slope
(r=−0.930). This reflects content: portraits/still-life (Vermeer, Rembrandt; dist~0.77,
slope~2.5–2.7) vs landscape paintings (Monet; dist~0.83, slope~1.3–1.4). The naturals and
faces groups show near-zero within-group correlations — the between-group r(dist,slope)
is driven almost entirely by category contrasts, not a continuous within-category effect.

---

### Thread 1 — Dissociation test (pre-registered 2026-06-24)

**Window:** 0.72 ≤ dist ≤ 0.87 (faces, paintings, naturals interleaved; textures excluded).
n=18 images.

| Category | n | Mean slope | Std |
|----------|---|-----------|-----|
| faces | 5 | 2.204 | 0.727 |
| painting | 5 | 2.073 | 0.685 |
| natural | 8 | 1.325 | 0.367 |

**Kruskal-Wallis:** H=7.938, p=0.019, η²=0.396 (p < 0.05 ✓)  
**Mann-Whitney faces vs natural:** U=37, p=0.011, mean_diff=+0.879 (pre-stated pair ✓)  
**Mann-Whitney faces vs painting:** U=16, p=0.548, mean_diff=+0.131 (pre-stated pair, NS)

**Verdict: DISSOCIATION CONFIRMED** — slope varies significantly where distance does not.
Within the 0.72–0.87 window, faces and paintings have slope ~2.1, while naturals have
slope ~1.3, despite all three categories overlapping in ResNet50 distance. Max mean
difference = 0.879 (faces vs naturals), far exceeding the pre-stated threshold of 0.5.

**Interpretation:** Distance is a correlate of slope across the full range, but within
the mid-distance band, category/content type determines slope elevation, not distance
per se. "Face-like" content — including portrait paintings — drives high slope regardless
of distance. Distance captures category identity at the coarse level (textures far, faces
near ImageNet) but is not the proximal cause. The relationship must be described as
content-driven slope elevation, with distance as a proxy for content type.

---

### Thread 2 — Slope → noise-floor α (power-law re-estimation)

**Fit set:** n=24 (all active images with slope ≥ 0.2). Wood_grain included per
pre-registration; NF bias flagged.

**OLS log-log fit (n=24, df=22):**
```
  α̂  = +0.7729   SE = 0.1282   R² = 0.623
  95% CI: [+0.507, +1.039]
```

**Sensitivity (wood_grain excluded, n=23):** α̂=0.803, CI [0.513, 1.094] — nearly
identical; wood_grain's NF bias does not materially affect the fit.

**Verdict: PROVISIONAL** — CI excludes 0 (H0 rejected: α≠0) but includes 1.0
(sub-proportional claim α<1 is NOT confirmed at this n). The pre-stated definitive
threshold requires CI entirely below 1.0; that threshold is not met.

**⚠ MATERIAL CHANGE FROM UPDATE 6 — FLAG FOR §10:**  
Update 6 reported α̂=0.36, CI [0.07, 0.64] at n=5. The Phase 2 estimate at n=24 is
α̂=0.77, CI [0.51, 1.04]. These are incompatible: the n=5 estimate was severely biased
by the small sample and by the influence of one extreme point (soft_blobs was
excluded, but the 5 remaining images did not represent the full slope range).

**The sub-proportional claim from Update 6 (α≈0.35–0.5, "clearly below 1.0") is no
longer supported at n=24.** The mechanism claim (noise floor scales with slope, α>0)
is now on firmer footing — but the exponent is higher than previously estimated, and
the CI includes proportional scaling (α=1). §10 currently reads "α̂=0.36, t-CI [0.07,
0.64]" — this must be updated. Do not use the n=5 estimate in any forward claims.

---

### Thread 3 — Coherence mechanism (rho_nn → slope, NF, mediation)

| Test | Metric | Value | CI [95%] | n | Verdict |
|------|--------|-------|----------|---|---------|
| F (primary) | r(rho_nn, slope) | +0.441 | [+0.046, +0.717] | 24 | **(a) POSITIVE** |
| G | r(rho_nn, nf_std) | +0.609 | [+0.272, +0.812] | 24 | CI excl. 0 |
| H (mediation) | partial r(slope, nf_std \| rho_nn) | +0.515 | [+0.131, +0.765] | 24 | No mediation |

Test F verdict: **(a) POSITIVE** — r>0 and CI excludes 0 (pre-stated rule satisfied).  
At n=24, the coherence–slope link is confirmed at the 95% level.

Test G: rho_nn is even more strongly correlated with nf_std (r=+0.609) than with slope.
This makes sense: coherent null-space deviations directly reduce n_eff and inflate
window-to-window variance.

**Mediation (Test H):** Direct r(slope, nf_std)=+0.635. Partial r(slope, nf_std |
rho_nn)=+0.515. Threshold = 0.7×|direct| = 0.445. |partial|=0.515 > threshold →
**mediation NOT confirmed**. Rho_nn does not screen off the slope→nf_std relationship.

**Interpretation:** Coherence (rho_nn) and slope are both independently associated with
nf_std; rho_nn is not a complete mediator. Two plausible mechanisms:
1. rho_nn captures one pathway (spatial correlation reduces n_eff → inflates nf_std).
2. Slope has an additional direct path (higher slope images have more extreme null-space
   energy, harder to calibrate per OLS window even without correlation).
Both may operate simultaneously.

---

### Girl_sad_face anomaly

5 face images with slopes spanning 1.60–3.18:

| Image | slope | rho_nn |
|-------|-------|--------|
| face_algerian | 1.604 | 0.166 |
| face_red_hair | 1.703 | 0.166 |
| wayuu_woman | 1.748 | 0.245 |
| girl_sad_face | 2.783 | 0.154 |
| boy_face | 3.183 | 0.219 |

**Within-faces r(slope, rho_nn):** r=+0.075, CI [−0.864, +0.898], n=5, p=0.904.

**Verdict: CI includes 0 → anomaly is noise at n=5.** The girl_sad_face anomaly
(lower rho_nn than wayuu_woman despite higher slope) neither dissolves nor confirms at
n=5 faces. The within-faces CI spans most of [−1, +1]; no conclusion is possible.

The anomaly has not been resolved. Resolving it requires ≥15–20 face images.

---

### Overall: ESTABLISHED vs OPEN

**ESTABLISHED (pre-registered, n=24):**

1. **Slope is content-driven, not distance-driven (within distance bands).**
   Dissociation confirmed: at the same ResNet50 distance, face photographs and portrait
   paintings have mean slope ~2.1 while landscape naturals have mean slope ~1.3 (KW
   p=0.019, MW faces vs naturals p=0.011). Category/content type is the proximate driver.

2. **Distance correlates with slope across the full range (continuous relationship
   confirmed, but this is a group-contrast effect, not a within-group continuous effect).**
   r(dist, slope)=−0.689, CI [−0.855, −0.396], n=24; survives partial r on null_frac.
   Within-group correlations are near zero for faces, naturals, and textures — the
   between-group contrast (textures far/low slope; face-like content near/high slope)
   drives the aggregate r.

3. **Coherence (rho_nn) is positively correlated with slope: confirmed.**
   r(rho_nn, slope)=+0.441, CI [+0.046, +0.717], n=24. Sanity check passes.

4. **Noise floor scales with slope (α>0): confirmed at PROVISIONAL level.**
   α̂=0.77, CI [0.51, 1.04], n=24. CI excludes 0. Whether scaling is sub-proportional
   (α<1) is NOT confirmed (CI includes 1.0). Update 6's α≈0.35 was severely underpowered.

**OPEN:**

- **Whether α is sub-proportional (α<1):** requires larger n and/or wider slope range.
- **Within-faces coherence→slope:** n=5 faces, CI spans [−0.86, +0.90]. Not resolvable
  at this n. Requires ≥15 diverse face images.
- **Whether content type (faces/portrait paintings) or representational complexity is the
  real driver:** the dissociation is established, but the latent feature driving slope
  elevation within face-like content is not identified.

---

### Notes on new exclusion pattern (texture_brick, texture_stone)

Uniform close-up texture crops with very fine grain fail the r≥0.90 calibration threshold.
This appears to be a structural issue: for images where null-space energy is effectively
zero across all seeds (uniform textures with no semantic content), the predicted_std bins
collapse and the calibration curve is underdetermined. This is NOT a failure of the
calibration method — it is the correct behaviour: these images have near-zero null-space
signal and should not be reported as reliably calibrated.

---

## Update 9 — Phase 2 audit (peer-review pass, 2026-06-24)

**Date:** 2026-06-24  
**Purpose:** Honest-strength recording of Phase 2 verdicts. No new experiments.  
**Scope caveat (standing limitation on all Phase 2 claims):** All findings below are
specific to ResShift (4-step conditional diffusion SR) on BicubicDownsample(4). "Content-
bound slope elevation," the α estimate, and the coherence correlation are properties
observed in this model–operator pair. Whether they hold for other SR models, other
degradation operators, or other model families is untested.

---

### 1. Pre-registration integrity check

**Question:** Was the cal_r < 0.90 exclusion criterion pre-registered before the run?

**Answer: YES.** The original pre-registration (commit f7bc949, written 2026-06-23 before
any Phase 2 data was collected) states under **Sample → Exclusions**:

> "Any image where N=48 calibration fails to converge (r < 0.90)"

This exact criterion was applied: texture_brick (cal_r=0.586) and texture_stone
(cal_r=0.832) were excluded. Both values fall below 0.90. The exclusion is legitimate
and pre-registered. Thread 1 retains confirmatory status.

Note: the dissociation test was added in a separate amendment commit (8b5e68b,
2026-06-24) also before the run. Both pre-registration commits precede any data
collection.

**No post-hoc exclusions. n=26 → n=24 is legitimate.**

---

### 2. Threads recorded at honest strength

---

#### DISSOCIATION (lead result) — CONFIRMED, pre-registered

**Result:** KW p=0.019, η²=0.40. At the same ResNet50 distance (0.72–0.87):
- Faces mean slope = 2.20 ± 0.73, n=5
- Paintings mean slope = 2.07 ± 0.69, n=5
- Naturals mean slope = 1.33 ± 0.37, n=8
- Mann-Whitney faces vs naturals: U=37, p=0.011, mean difference +0.88

**Limitation:** n=5 per group (faces, paintings); the KW test has modest power at these
group sizes. The finding meets the pre-stated threshold (KW p<0.05 AND max mean
difference >0.5) but rests on small per-group n. A replication at n≥10 per group
would materially strengthen it.

**Interpretation (pre-registered):** Slope varies significantly where distance does not.
Distance is NOT the proximal driver of slope elevation within the 0.72–0.87 band.

**Post-hoc observation (NOT pre-registered, flagged exploratory):** The within-painting
correlation (Test E, pre-stated) is r=−0.930 CI [−0.995, −0.263], p=0.022, n=5.
Inspecting the data: Vermeer and Rembrandt portrait paintings have slopes 2.3–2.7 while
Monet landscape paintings have slopes 1.3–1.4 — replicating the faces/naturals pattern
within the painting category. This suggests the relevant variable is "portrait/figure
content" rather than "face photograph" specifically. This interpretation is POST-HOC
(not pre-stated) and is flagged as exploratory support; it does not contribute to the
confirmatory verdict.

**Status: RESOLVED — slope is content-bound, not distance-bound, within the 0.72–0.87
ResNet50 distance window; KW p=0.019, pre-registered, n=24. SCOPE: ResShift+
BicubicDownsample(4) only; per-group n=5 faces/paintings (KW power is modest; replication
at n≥10 per group would materially strengthen the result).**

---

#### THREAD 1 — Distance → slope: POSITIVE but BETWEEN-GROUP ONLY

**Pre-stated verdict (A+C rule): (a) POSITIVE.**
- Test A: Pearson r=−0.689, CI [−0.855, −0.396], n=24, p<0.001
- Test C: Partial r(dist, slope | null_frac) = −0.663, CI [−0.844, −0.345], n=24
- Both CI exclude 0, r<0 ✓ — pre-stated rule met.

**What the within-group analysis (Test E, pre-stated) shows — decisive context:**

| Group | n | r(dist, slope) | CI [95%] | Significant? | Direction |
|-------|---|----------------|----------|--------------|-----------|
| faces | 5 | +0.438 | [−0.724, +0.952] | No | WRONG (predicted: −) |
| naturals | 9 | +0.136 | [−0.581, +0.734] | No | WRONG (predicted: −) |
| paintings | 5 | −0.930 | [−0.995, −0.263] | Yes | correct |
| textures | 5 | −0.570 | [−0.966, +0.628] | No | correct |

The pre-stated Test E decision rule states: "if within-group r is near zero for ALL
groups but between-group r is large → it's a group-level effect, not continuous. Report
explicitly."

This applies exactly. Within faces and naturals (the two groups with the most images
and the largest between-group contrast), the within-group r is near zero and in the
WRONG direction. The aggregate r=−0.689 is **entirely a between-group contrast**:
textures (dist~0.90–0.94, slope~0.3–0.9) vs face-like content (dist~0.72–0.83,
slope~1.6–3.2). There is no evidence of a continuous within-category distance→slope
function.

**Honest verdict:** (a) POSITIVE per the A+C rule (between-group correlation is real
and robust), but the continuous distance→slope hypothesis is NOT supported by the
within-group analysis. The correlation captures group identity (content type), not a
smooth distance-to-slope mapping. This is the correct characterisation per Test E.

**Status: RESOLVED.** Distance is a proxy for category identity (and, within paintings,
for representational content), not a continuous causal driver of slope.

---

#### THREAD 2 — α: CONFIRMED POSITIVE, MAGNITUDE UNRESOLVED, MECHANISM REOPENED

**Result:** α̂ = 0.773, 95% CI [0.507, 1.039], n=24, R²=0.623.
Sensitivity (wood_grain excluded, n=23): α̂ = 0.803, CI [0.513, 1.094] — near-identical.

**CI verdict (per pre-stated rule):**
- CI excludes 0 → PROVISIONAL (not Null)
- CI includes 1.0 → NOT Definitive (sub-proportional claim not confirmed)

**What α near 1 means for Update 6's mechanism claim:**

Update 6 argued that α<1 (sub-proportional) supports the spatial-correlation mechanism
and rules out a simple OLS scale-invariance artifact (which predicts α=0). With CI
including 1.0, two possibilities are no longer distinguishable:

1. **Spatial-correlation mechanism (Update 6):** Pixel reconstruction errors are spatially
   correlated → n_eff < N → noise floor scales sub-proportionally (α<1). This predicts
   α in (0, 1) and was the stated evidence for the mechanism.

2. **Proportional amplitude effect (alternative):** Higher-slope images have proportionally
   larger null-space amplitudes. If both predicted_std and actual_error scale with slope,
   the OLS calibration slope varies proportionally across windows, giving α≈1. This does
   NOT require spatial correlation; it is a different artifact of the OLS estimator when
   slope is large.

Since the CI [0.51, 1.04] includes 1.0, possibility (2) cannot be ruled out. The simple
OLS scale-invariance artifact (α=0) remains ruled out. But whether α<1 (mechanism 1) or
α≈1 (alternative) is the correct description is unresolved.

**Update 6's "model/data effect, not estimator artifact" claim is WEAKENED.** α>0
is confirmed, which rules out the specific α=0 estimator artifact. But α=1 constitutes
a different kind of amplitude-scaling effect that does not require n_eff reduction. The
mechanism is not settled; it is open.

**9.2× SNR (§10) — unchanged as a lower bound:**

The 9.2× ratio (faces vs naturals slope contrast divided by boy_face N=12 noise floor)
is directly measured and does not depend on α. At any α>0, using the face group's noise
floor as denominator is conservative (face NF is elevated by slope), so 9.2× is a lower
bound regardless of α's exact value. At α̂=0.77, the face NF is approximately
slope^0.77 ≈ (2.8/0.95)^0.77 ≈ 2.3× larger than boardwalk NF, making the denominator
more conservative than at α̂=0.36. The 9.2× lower bound stands.

**Status: OPEN.** α>0 is established. Whether α<1 (spatial correlation mechanism) or
α≈1 (amplitude effect) is the correct characterisation is unresolved at n=24.

---

#### THREAD 3 — Coherence: CORRELATED SIDE-PROPERTY, NOT CONFIRMED MEDIATOR

**Pre-stated verdict: (a) POSITIVE for Test F.**

- Test F: r(rho_nn, slope) = +0.441, CI [+0.046, +0.717], n=24, p=0.031
- Test G: r(rho_nn, nf_std) = +0.609, CI [+0.272, +0.812], n=24, p=0.002
- Test H (mediation): direct r(slope, nf_std) = +0.635; partial r(slope, nf_std | rho_nn)
  = +0.515; threshold = 0.7×0.635 = 0.445; |partial| = 0.515 > threshold → no mediation

**(a) POSITIVE** is the correct pre-stated verdict for Test F. But two limitations
reduce its strength:

**Fragility of the CI:** the lower bound is +0.046, barely above zero. One influential
image could shift this to null. r=+0.441 at n=24 has power ≈0.55 at α=0.05 (below 80%),
meaning the result is not stably detectable at this n.

**Mediation failure — key nuance understated in Update 8:** Update 6 proposed the causal
chain: high slope → spatially coherent hallucinations → lower n_eff → higher noise floor
(slope → rho_nn → nf_std). The mediation test directly tests whether rho_nn screens off
the slope→nf_std association. It does not: partial r = +0.515 exceeds the pre-stated
threshold of 0.445. rho_nn is NOT a confirmed mediator.

**This weakens Update 6's causal narrative.** The chain slope → rho_nn → nf_std is not
confirmed. rho_nn correlates with slope and with nf_std, but it is a correlated
side-property of high-slope images, not a confirmed causal intermediate. An alternative
is possible: all three variables (slope, rho_nn, nf_std) may co-vary with a common
upstream factor (e.g. local spatial regularity of the image content), without rho_nn
specifically mediating slope's effect on nf_std.

**Honest verdict:** r(rho_nn, slope) is positive and significant (CI barely excludes 0).
rho_nn does NOT mediate the slope→nf_std pathway. Coherence is a correlate, not a
confirmed causal channel.

**Status: OPEN.** The mechanism chain is not established.

---

#### GIRL_SAD_FACE ANOMALY — UNRESOLVED

Within-faces r(slope, rho_nn) = +0.075, CI [−0.864, +0.898], n=5, p=0.904.
CI spans almost all of [−1, +1]. No information. Requires ≥15 face images.

**Status: OPEN.** Uninformative at n=5.

---

### 3. What is now established, what is not

**ESTABLISHED (pre-registered, robust):**
1. Slope elevation is content-bound, not distance-bound. "Face/portrait" content elevates
   slope regardless of ResNet50 distance (dissociation confirmed, KW p=0.019, n=24).
   Within the 0.72–0.87 distance window, faces and portrait paintings have mean slope ~2.1
   while landscape naturals have mean slope ~1.33.

2. The aggregate r(dist, slope) = −0.689 is real but is a between-group contrast, not a
   continuous distance effect. Within faces and naturals, no within-group distance→slope
   correlation is detectable (CIs include 0 for all non-painting groups).

3. α > 0 is established (CI [0.51, 1.04] excludes 0). The simple OLS scale-invariance
   artifact (α=0) is ruled out.

4. rho_nn > 0 correlates with slope (r=+0.441) and more strongly with nf_std (r=+0.609).
   These correlations are real but fragile at n=24.

**NOT ESTABLISHED (requires further work):**
1. Whether α < 1 (sub-proportional, spatial-correlation mechanism) vs α ≈ 1 (proportional
   amplitude effect). Unresolvable at n=24 with CI spanning [0.51, 1.04].

2. Whether rho_nn mediates the slope→nf_std relationship. Mediation test failed.

3. The specific causal chain from Update 6 (coherent hallucinations → n_eff reduction →
   sub-proportional noise floor). Consistent with data but not confirmed.

4. Generality: all findings are ResShift + BicubicDownsample(4) only.

---

## Update 10 — Content-driver investigation (2026-06-24)

**Script:** `eval/content_driver_analysis.py`
**Pre-registration:** `eval/pre_registration_content_driver.md` (commits 4f19d64, 75b2d46)
**Results:** `eval/content_driver_results.txt`

**Question:** What content property drives slope elevation in faces and portrait paintings vs landscapes?

**Sample:** n=24 (same Phase 2 dataset). Slopes from Phase 2 (commit dd987b1), unchanged.

### Three pre-registered hypotheses

| Feature | Operationalisation | r | 98.3%CI (Bonferroni) | Direction check | Verdict |
|---|---|---|---|---|---|
| H1 (−β_spec) | −(log-log slope of radially averaged GT power spectrum, f∈[2/256,64/256]) | +0.717 | [+0.324, +0.899] | PASS | CONFIRMED |
| H2 (rho_gt) | Mean ACF of GT image at Δ=1 px (same method as rho_nn, applied to GT) | +0.633 | [+0.179, +0.865] | PASS | CONFIRMED |
| H3 (A_dom) | Binary: prominent human face/figure present? (pre-annotated, n_1=8) | +0.774 | [+0.433, +0.921] | PASS (trivial) | CONFIRMED |

All three hypotheses are CONFIRMED: each feature correlates with slope at Bonferroni-corrected
significance AND the portrait-vs-landscape painting directional check passes for each.

### Portrait vs landscape paintings directional check (Addition A)

Portrait paintings (n=3): Vermeer Pearl, Vermeer Milk, Rembrandt — mean slope=2.559, mean −β_spec=2.780, mean rho_gt=0.981
Landscape paintings (n=2): Monet Magpie, Monet Lilies — mean slope=1.345, mean −β_spec=2.034, mean rho_gt=0.879

All three features go in the correct direction (portrait > landscape). However: **this contrast
CANNOT confirm content vs painting style as the driver**. Content (face/figure) and style
(Vermeer/Rembrandt chiaroscuro vs Monet impressionism) are fully confounded at n=3 vs n=2.
Minimum achievable p=0.10 at these group sizes — directional check is informative, not confirmatory.

### Collinearity check (Addition B, pre-registered)

Pairwise r among predictors:
  r(−β_spec, rho_gt) = +0.816  ← above 0.70 threshold
  r(−β_spec, A_dom) = +0.643
  r(rho_gt, A_dom) = +0.465

VIF: VIF(−β_spec)=4.1, VIF(rho_gt)=3.1, VIF(A_dom)=1.7. Max VIF=4.1.
1/3 pairwise r > 0.70 → LOW/MODERATE COLLINEARITY by pre-registered rule
(ENTANGLED required ≥2 pairs > 0.70 AND all partials including 0).

### Partial correlations — ONE_INDEPENDENT (Addition B, pre-registered)

Partial r (df=20, CIs are wide at this n):

| Partial | r | 95%CI | Verdict |
|---|---|---|---|
| P1: A_dom ∣ −β_spec, rho_gt | +0.612 | [+0.229, +0.831] | CI excl 0 → INDEPENDENT |
| P2: −β_spec ∣ rho_gt, A_dom | +0.143 | [−0.323, +0.553] | CI incl 0 |
| P3: rho_gt ∣ −β_spec, A_dom | +0.246 | [−0.224, +0.623] | CI incl 0 |

**Pre-registered verdict: ONE_INDEPENDENT.** A_dom (semantic face/figure presence)
survives controlling both spectral and coherence features. H1 (−β_spec) and H2 (rho_gt)
do NOT survive after controlling for A_dom — their marginal correlations appear to reflect
co-variation with the face/portrait split rather than an independent spectral mechanism.

**Caveats (pre-stated):**
- P1 CI lower bound = +0.229 — fragile; partial df=20 gives wide CIs; one influential image
  could shift this result.
- n=24 with 3 controls leaves the partials underpowered. This is not a strong claim of
  A_dom as the sole driver.
- NO content-vs-style claim: the painting contrast is directional only (Addition A).

### H3 internal falsification: natural_caribou

natural_caribou: A_dom=0 (not human), slope=2.1056 = **100th percentile** among other A_dom=0 naturals
(n=8, mean=1.204, range 0.951–1.506). The caribou has the highest slope of any A_dom=0 non-painting image.

Caribou spectral features: −β_spec=2.107, rho_gt=0.886.
Other A_dom=0 naturals have −β_spec ranging 1.790–2.487 and rho_gt ranging 0.756–0.965.
The caribou's −β_spec is NOT particularly elevated among A_dom=0 naturals — several naturals
with lower slope have higher −β_spec (e.g., nat_landscape3: −β_spec=2.713, slope=1.031).

**H3 challenge (pre-stated):** Caribou is a prominent foreground animal (not human → A_dom=0),
yet has elevated slope. H1/H2 features do not clearly account for the elevation.

### A_dom_broad secondary (pre-stated)

A_dom_broad=1: includes prominent non-human animals (adds natural_caribou, frog_on_log; n_broad=10).

r(A_dom_broad, slope) = +0.800, 95%CI [+0.569, +0.914] — n_1=10, mean_slope_1=2.223
r(A_dom, slope) = +0.774 — n_1=8, mean_slope_1=2.337

r_broad (+0.800) > r_A_dom (+0.774). Broadening to prominent animals improves the correlation.
Note: frog_on_log (A_dom_broad=1, slope=1.424) is only modestly elevated. The improvement is
driven mainly by natural_caribou (slope=2.106) entering the A_dom_broad=1 group.

**Interpretation:** The data are slightly more consistent with "prominent foreground subject
(including animals)" than "human face/figure specifically." However, the difference (0.800 vs
0.774) is small and well within noise at n=24.

### Honest synthesis

**What is established (pre-registered):**
- All three content features correlate with slope at Bonferroni-corrected significance (n=24).
- Slope tracks the face/portrait vs landscape/texture split equally well whether measured as
  spectral smoothness (H1), local image coherence (H2), or semantic face/figure presence (H3).
- After controlling the other two features, A_dom (semantic face/figure presence) is the sole
  feature whose partial correlation clearly excludes 0 (P1=+0.612, CI [+0.229, +0.831]).
  −β_spec and rho_gt do not survive as independent predictors.

**What is NOT established:**
- H1 and H2 are not confirmed as independent causal drivers. Their marginal correlations likely
  reflect co-variation with the face/portrait identity (r(−β_spec, A_dom)=0.643).
- Whether A_dom reflects training data over-representation, semantic prior, or another mechanism
  is not addressed. A_dom is a behavioral label, not a mechanistic variable.
- Content vs style: the portrait-vs-landscape painting contrast is directional only. Cannot
  separate Vermeer/Rembrandt face content from their stylistic conventions.
- Prominent foreground subject (A_dom_broad) may be slightly better than human-specific (A_dom),
  but the difference is small and not statistically separable at n=24.
- Caribou (A_dom=0, slope highest among A_dom=0 naturals) challenges A_dom's completeness.
  Its H1/H2 features are not elevated relative to other naturals. Whether it represents an
  exception, a measurement outlier, or evidence for A_dom_broad requires targeted study.

**Generalisation:** ResShift+BicubicDownsample(4) only.

**Status: PROVISIONAL.** A_dom is the strongest independent predictor, but the result is
fragile at n=24, entails no mechanism claim, and has no content-vs-style warrant.

---

### Update 10 — Robustness addendum (2026-06-24)

**Script:** `eval/robustness_analysis.py`
**Pre-registration:** `eval/pre_registration_robustness.md` (commit 5850951, before run)

#### Check 1: Leave-one-out on P1 (A_dom | H1, H2)

n=23 at each drop, df=19, t-critical at df=19.

| | A_dom | A_dom_broad |
|---|---|---|
| Full-sample P1 | +0.612  CI [+0.229, +0.831] | +0.661  CI [+0.305, +0.854] |
| LOO P1 range | [+0.554, +0.688] | [+0.607, +0.719] |
| Exclude-0 count | **24/24** | **24/24** |
| Pivotal images | none | none |
| LOO verdict | **ROBUST** | **ROBUST** |

**Pre-registered verdict: A_dom_ROBUST.** (Both are 24/24; by the pre-stated rule,
equal LOO counts default to A_dom_ROBUST rather than A_dom_BROAD_PREFERRED.)

The A_dom independence is not point-driven. All 24 single-image drops keep the
partial CI above zero. This substantially strengthens Update 10's provisional claim.

#### Check 2: A_dom_broad — caribou consistency

natural_caribou under A_dom_broad=1: slope=2.106 vs group (n=9 other A_dom_broad=1
images): mean=2.236 ± SD=0.631 → z=−0.21. **Caribou is fully consistent with the
A_dom_broad=1 group.** It is not an outlier when classified as "prominent foreground
subject (animal or human)."

A_dom_broad also has higher full-sample P1' (+0.661 vs +0.612), higher marginal r
(+0.800 vs +0.774), and is equally robust (24/24 LOO). A_dom_broad is a marginally
more complete operationalisation of the content feature. However, since both are
equally robust by the pre-stated decision rule, the result is reported under both labels.

#### Annotation-reliability caveat (standing limitation, pre-registered)

A_dom and A_dom_broad are **single-annotator binary labels** (Claude Code, 2026-06-24).
No inter-rater agreement was measured. The caribou case demonstrates the boundary is
judgment-dependent: a "prominent foreground caribou" could be A_dom=0 (not human)
or A_dom_broad=1 (prominent animal), depending on how narrowly the rule is interpreted.
All content-driver results are conditional on this annotation scheme.

**This caveat must appear in any citation of the content-driver result.**

#### Revised status

**SUPPORTED — dominant foreground subject (A_dom_broad: prominent animal or human)
independently predicts slope elevation above low-level image statistics (spectral slope
H1, GT autocorrelation H2), robust to single-observation removal (LOO 24/24, no pivotal
images). SCOPE: ResShift+BicubicDownsample(4) only; behavioural label, no mechanism
claim; content-vs-style not testable at this n (painting contrast n=3 vs n=2 confounded);
single-annotator annotation, not inter-rater-validated.**

Upgraded from PROVISIONAL: all 24 single-image LOO drops keep both P1 (A_dom) and P1'
(A_dom_broad) CIs above zero. Independence from spectral/coherence features is not a
statistical artifact of any single observation.
