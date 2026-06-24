# Pre-registration: Content-Driver Investigation
# What content property drives calibration slope elevation in faces and portrait paintings?
# Written: 2026-06-24 — BEFORE any feature-vs-slope correlation is computed.
# THIS DOCUMENT IS LOCKED ON COMMIT. No post-hoc changes allowed.

## Motivation

Phase 2 (n=24, pre-registered, commit dd987b1) confirmed a dissociation: at the same
ResNet50 distance (0.72–0.87), faces and portrait paintings have higher calibration slope
than landscape naturals (KW p=0.019, η²=0.40). Distance is a proxy for content type,
not a continuous driver.

The open question: WHAT content property of "face/portrait" images causes slope elevation?

Primary discriminating contrast — within paintings, same medium and era:
  Portrait paintings: Vermeer Pearl Earring, Vermeer Milkmaid, Rembrandt Self-Portrait
    n=3, slopes 2.30–2.71
  Landscape paintings: Monet Magpie, Monet Water Lilies
    n=2, slopes 1.27–1.42
  Mean difference ≈ 1.2. Both subgroups span ResNet50 distance 0.76–0.83. Distance
  cannot explain this contrast.

## HARD ANTI-FISHING RULE

Exactly 3 content features are tested. Features were chosen before any feature-vs-slope
correlation was computed. Only these 3 are tested. Any feature noticed during computation
is labeled POST-HOC EXPLORATORY and excluded from the confirmatory table.

---

## Hypotheses

### H1 — Global spectral slope (−β_spec)

**What it measures:**
The log-log slope of the radially averaged power spectrum of the 256×256 grayscale GT
image. Specifically: compute 2D FFT of the grayscale image, compute power (|F|²), radially
average across pixels at each normalized frequency f, fit OLS to log(P) vs log(f) for
f ∈ [2/256, 64/256] (structures 4–128 pixels in size, excluding DC and high-frequency
noise). The slope β_spec is negative for all natural images (1/f^|β| structure). More
negative β = steeper spectral decline = more power at low frequencies = smoother image.

Predictor entered in correlation: −β_spec (positive for smooth/structured images).

**Pre-stated prediction:**
- r(−β_spec, slope) > 0, CI excludes 0, full sample n=24
- Portrait paintings have more negative β_spec (steeper) than landscape paintings:
    mean β_spec(portraits) < mean β_spec(landscapes)  [directional check only, n=3 vs n=2]

**Mechanistic rationale:** Smooth skin/background regions dominate portrait images →
more power concentrated at low spatial frequencies → ResShift null-space patterns follow
this large-scale structure → more organized ensemble variation → higher calibration slope.

---

### H2 — GT spatial autocorrelation at nearest-neighbor lag (rho_gt)

**What it measures:**
Mean ACF of the grayscale GT image at Δ=1 pixel lag. Computation: normalize image to
zero mean, unit std; compute ACF in horizontal and vertical directions at Δ=1:
  rho_gt = mean(z[:, :-1] * z[:, 1:]) averaged with mean(z[:-1, :] * z[1:, :])
This is the identical computation used for rho_nn (null-space deviation ACF in Phase 2)
but applied to the GT image pixels themselves.

**Note on H1 vs H2:** These measure related but distinct properties. rho_gt captures
LOCAL coherence (adjacent pixel similarity, Δ=1). β_spec captures the GLOBAL spectral
distribution. By the Wiener-Khinchin theorem these are related, but can dissociate: an
image with high rho_gt (locally smooth) can have a shallow β (little long-range structure).
If one predicts slope and the other does not, this indicates the scale at which the
mechanism operates.

**Pre-stated prediction:**
- r(rho_gt, slope) > 0, CI excludes 0, full sample n=24
- Portrait paintings have higher rho_gt than landscape paintings (directional check)

**Mechanistic rationale:** A GT image with highly correlated adjacent pixels may constrain
the null-space prior toward similarly coherent deviation patterns → more organized ensemble
variation → higher calibration slope.

---

### H3 — Dominant human face/figure annotation (A_dom)

**What it measures:**
Binary annotation (0/1) of whether the image contains a prominent human face or human
figure as the central/dominant subject.

**Annotation rule (strict):**
A_dom = 1 if and only if:
  (a) the central/dominant subject is human (not animal, not landscape), AND
  (b) the human face or figure occupies a major portion of the frame or is the clear
      focal point.
Annotation based on image content knowledge (image names and known visual descriptions)
only — NOT based on slope values. This annotation was fixed before any feature computation.

**Conflict-of-interest note:** Phase 2 slope values were observed before this annotation
was finalized. However, the annotation is content-based: "is this image of a human face
or figure?" For example, natural_caribou is annotated A_dom=0 despite its above-average
slope of 2.11 (the rule is applied consistently even where it weakens the correlation).

**Pre-stated annotations for all n=24 images:**

| Image                 | A_dom | Basis                                              |
|-----------------------|-------|----------------------------------------------------|
| face_red_hair         | 1     | Portrait photograph of a woman with red hair       |
| wayuu_woman           | 1     | Portrait photograph of Wayuu woman                 |
| girl_sad_face         | 1     | Child portrait                                     |
| paint_vermeer_pearl   | 1     | Girl with a Pearl Earring — close-up face portrait |
| paint_vermeer_milk    | 1     | The Milkmaid — human figure is dominant subject    |
| boardwalk             | 0     | Landscape/nature scene, no person present          |
| paint_rembrandt_self  | 1     | Self-portrait — face is dominant subject           |
| nat_landscape2        | 0     | Nature landscape, no person                        |
| nature_land           | 0     | Nature scene, no person                            |
| face_algerian         | 1     | Portrait photograph                                |
| boy_face              | 1     | Child portrait                                     |
| nat_landscape3        | 0     | Nature landscape, no person                        |
| natural_caribou       | 0     | Animal (caribou) — not human per annotation rule   |
| paint_monet_magpie    | 0     | Landscape painting; bird is not a human            |
| paint_monet_lilies    | 0     | Landscape painting, no person                      |
| natural_coral_reef    | 0     | Underwater scene, no human                         |
| natural_fir_snow      | 0     | Forest/snow scene, no person                       |
| frog_on_log           | 0     | Animal (frog) — not human per annotation rule      |
| grass_meadow          | 0     | Texture, no person                                 |
| dirt_soil             | 0     | Texture, no person                                 |
| natural_snow_mountain | 0     | Mountain landscape, no person                      |
| texture_sand          | 0     | Texture, no person                                 |
| texture_cement        | 0     | Texture, no person                                 |
| wood_grain            | 0     | Texture, no person                                 |

Summary: A_dom=1: n=8 (all 5 face photos + 3 portrait paintings)
          A_dom=0: n=16

**Pre-stated prediction:**
- Point-biserial r(A_dom, slope) > 0, CI excludes 0, n=24
- All portrait paintings annotated A_dom=1, all landscape paintings A_dom=0 (trivially
  satisfied by the annotation above)

**Mechanistic rationale:** Human face/figure content may be over-represented in SR training
data (ResShift and similar models). For face/figure inputs, the null-space prior is better
constrained (the model "knows what faces look like") → more confident and structured
ensemble variation → higher calibration slope.

**Critical internal test case:** natural_caribou (A_dom=0, slope=2.11) is above average
for its category. If H3 (human face/figure) is the driver, this image's slope should NOT
be predicted by A_dom. Its slope elevation should either be unexplained by all three
hypotheses (pointing to a fourth feature, such as prominent animal figure) or be captured
by H1/H2 via spectral properties. This case is noted as a pre-stated interpretive anchor.

---

### Pre-stated secondary analysis: A_dom_broad

**Definition:** A_dom_broad=1 if the image contains a prominent non-human animal OR human
figure as the central/dominant subject.

Differs from A_dom only in: natural_caribou → 1, frog_on_log → 1.

**Purpose:** If A_dom fails but A_dom_broad passes, the effect is about "prominent
foreground subject" (including animals), not specifically human faces.

**Status:** Secondary, labeled, reported separately. Does not enter Bonferroni correction.

---

## Sample

n=24. Same as Phase 2 dataset. Excluded per Phase 2 pre-registered criterion:
  texture_brick (cal_r=0.586 < 0.90), texture_stone (cal_r=0.832 < 0.90).

Phase 2 slopes (for reference, computed before this registration):
  face_red_hair=1.70, wayuu_woman=1.75, girl_sad_face=2.78, paint_vermeer_pearl=2.71,
  paint_vermeer_milk=2.30, boardwalk=0.95, paint_rembrandt_self=2.66, nat_landscape2=1.21,
  nature_land=1.11, face_algerian=1.60, boy_face=3.18, nat_landscape3=1.03,
  natural_caribou=2.11, paint_monet_magpie=1.27, paint_monet_lilies=1.42,
  natural_coral_reef=1.26, natural_fir_snow=1.51, frog_on_log=1.42, grass_meadow=0.58,
  dirt_soil=0.85, natural_snow_mountain=1.14, texture_sand=0.34, texture_cement=0.34,
  wood_grain=0.44.

---

## Discriminating contrast: within paintings (n=3 vs n=2)

Portrait paintings: paint_vermeer_pearl, paint_vermeer_milk, paint_rembrandt_self
Landscape paintings: paint_monet_magpie, paint_monet_lilies

POWER WARNING: n=3 vs n=2 → C(5,2)=10 possible arrangements. Minimum achievable
one-tailed p for any rank test = 1/10 = 0.10. Statistical significance (p<0.05) is
structurally impossible at these group sizes. NO statistical test is reported for this
contrast. It is a DIRECTIONAL CHECK only.

For each feature, the check passes if:
  H1: mean(−β_spec, portraits) > mean(−β_spec, landscapes)
  H2: mean(rho_gt, portraits) > mean(rho_gt, landscapes)
  H3: trivially passes (portrait paintings all A_dom=1, landscape paintings all A_dom=0)

A feature that fails the directional check within paintings is NOT confirmable even if
it passes Bonferroni on the full sample (the correlation may be driven by the
faces/texture extremes rather than the portrait content specifically).

---

## Decision rules (per hypothesis)

Each hypothesis is assessed against BOTH conditions:

| Verdict                      | Full-sample CI (n=24)          | Within-paintings direction  |
|------------------------------|--------------------------------|-----------------------------|
| CONFIRMED                    | Bonferroni CI excludes 0       | Correct direction           |
| EXPLORATORY-SUPPORTED        | Uncorrected CI excl. 0, Bonferroni fails | Correct direction |
| EXPLORATORY-INDETERMINATE    | CI includes 0 OR one condition fails | Correct direction      |
| REFUTED                      | Point estimate in wrong direction | —                        |

Bonferroni threshold: α = 0.05/3 = 0.0167.
CI method: Fisher-z transform with t-critical at df=n-2=22 (same as Phase 2).
For H3 (binary A_dom): point-biserial correlation treated identically to Pearson r.

---

## Multiple-comparison plan

- 3 primary tests (H1, H2, H3): Bonferroni α = 0.0167
- All 3 reported regardless of outcome; no selective omission
- A_dom_broad: secondary, separate, excluded from Bonferroni
- Any additional test not listed here: POST-HOC EXPLORATORY, excluded from confirmatory table

---

## What distinguishes the hypotheses

H1 vs H2: If rho_gt (H2) is the stronger predictor but β_spec (H1) is weaker, the
mechanism operates at the local scale (adjacent pixel coherence), not global spectral
structure. The converse points to the full spectral distribution.

H1/H2 vs H3: If H3 (A_dom) is confirmed but H1 and H2 are not, the effect is specifically
about semantic face/figure content — not about spectral smoothness shared by non-face
smooth images. If H1/H2 pass but H3 fails, any smooth/structured image should elevate
slope (testable with smooth non-face images, currently outside our sample).

All three fail: the content property driving slope is not (a) spectral smoothness,
(b) local image coherence, nor (c) human-figure presence. This would require new
hypotheses (e.g., model training data representation, semantic complexity, object
segmentation structure) and a new pre-registration.

---

## Scope constraint

All results are ResShift+BicubicDownsample(4) specific. GT images are 256×256 grayscale
crops from the Phase 2 natural_images/ collection (same images used in Phase 2). No
generalization to other engines or operators is claimed.

---

## STOP

This document defines Phase 1 (design). Phase 2 (feature computation and correlation
tests) does not begin until this document is committed and user approval is given.
