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

## ADDITION A — Painting-contrast power limitation and content-vs-style ceiling
## (Added 2026-06-24, BEFORE any feature computation)

Portrait paintings: paint_vermeer_pearl, paint_vermeer_milk, paint_rembrandt_self (n=3)
Landscape paintings: paint_monet_magpie, paint_monet_lilies (n=2)

POWER WARNING — structural: n=3 vs n=2 → C(5,2)=10 possible arrangements. Minimum
achievable one-tailed p for any rank test = 1/10 = 0.10. Statistical significance (p<0.05)
is structurally impossible at these group sizes. NO statistical test is reported.
This contrast is a DIRECTIONAL CHECK only.

CEILING ON INTERPRETATION — content vs style: The portrait-vs-landscape painting split
confounds two things simultaneously: (a) content (face/figure vs scene), and (b) painting
style (Vermeer/Rembrandt chiaroscuro vs Monet impressionism). These are not separable in
the current dataset. THEREFORE:

  THIS SESSION CANNOT CONFIRM "CONTENT, NOT PAINTING STYLE" AS THE DRIVER.

Even a perfect directional check (all three portrait features above all landscape features)
would not establish that content drives slope rather than stylistic painting conventions.
This limitation must appear in every written verdict. Specifically:
  - "portrait paintings have X > landscape paintings" is acceptable.
  - "this shows content, not style, drives slope" is NOT — content and style are
    confounded at n=3 vs n=2.

What the painting contrast CAN do: serve as an out-of-sample directional consistency
check for correlations established across the full n=24 sample. If a feature separates
portrait from landscape paintings in the same direction as its n=24 correlation, that is
evidence of consistency. It is not evidence of the mechanism.

For each feature, the directional check passes if:
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
- Partial correlations (Addition B below): separate section, labeled discriminating analysis
- Any additional test not listed here: POST-HOC EXPLORATORY, excluded from confirmatory table

---

## ADDITION B — Discriminating partial-correlation structure
## (Added 2026-06-24, BEFORE any feature computation)

### Motivation

H1 (−β_spec), H2 (rho_gt), and H3 (A_dom) are expected to co-vary: all three separate
faces and portrait paintings from landscapes/textures, so all three may correlate with
slope for the same underlying reason — the face/non-face split. Three separate
"CONFIRMED" marginal correlations would NOT establish three independent drivers; they
could be three measurements of one underlying face/portrait identity.

To discriminate the hypotheses, we pre-register partial correlations that ask: does each
feature predict slope BEYOND the other two?

### Pre-stated partial correlations

Three partials computed by regressing out the other two predictors (OLS residuals):

  P1: partial r(A_dom, slope | −β_spec, rho_gt)
      Does semantic face/figure content predict slope after removing spectral and coherence
      effects? If yes → A_dom is an independent driver beyond low-level statistics.
      If collapses to ~0 → semantic label is a proxy for spectral/coherence structure.

  P2: partial r(−β_spec, slope | rho_gt, A_dom)
      Does global spectral slope predict slope after removing local coherence and semantic
      content? If yes → spectral structure is an independent driver.

  P3: partial r(rho_gt, slope | −β_spec, A_dom)
      Does local nearest-neighbor coherence predict slope after removing spectral and
      semantic effects? If yes → local coherence is an independent driver.

### Decision rule for independence

A hypothesis is an INDEPENDENT DRIVER if: its partial correlation CI (computed from OLS
residual Pearson r, df=n-2-2=20 for n=24 with 2 controls) excludes 0.

IMPORTANT CAVEAT ON POWER: df=20 for a single-predictor correlation gives t-critical
≈ 2.09 (α=0.05). At n=24, controlling 2 covariates, the CI for partial r will be
substantially wider than for the marginal r. Even a true partial r of 0.45 has only
~50% power at df=20. Partials should be treated as the discriminating evidence when they
clearly include or exclude 0, but intermediate results are genuinely inconclusive.

### Collinearity check (pre-stated, not optional)

Before interpreting partials, report:
  - Pairwise Pearson r among the three predictors: r(H1,H2), r(H1,H3), r(H2,H3)
  - Variance Inflation Factor (VIF) for each predictor in the multiple regression of
    slope on H1, H2, H3: VIF_i = 1/(1 − R²_i) where R²_i is from regressing predictor
    i on the other two.
  - VIF > 5 is treated as "high collinearity." VIF > 10 is "severe."

### Entanglement verdict (pre-stated — the honest outcome if partials are inconclusive)

If ALL THREE partial CIs include 0 AND at least two pairwise r's among predictors are
|r| > 0.70: report verdict as "ENTANGLED — cannot separate drivers at n=24." This is a
valid and informative result, not a failure. In that case, state:
  - What the marginals establish: the full predictor set tracks the face/portrait split
  - What they do not establish: which specific property (spectral, coherence, semantic)
    is the active mechanism
  - What would resolve it: approximate sample size or design change needed to separate
    the drivers (e.g., add smooth non-face images with high −β_spec but A_dom=0; add
    face images with disrupted spectral structure)

If ONE partial CI clearly excludes 0 while the others clearly include 0: report that
predictor as the independent driver with its caveats (VIF, n=24 limitation, no
content-vs-style claim).

If TWO OR MORE partial CIs exclude 0: report that collinearity has not screened them
apart and both remain candidate drivers — this is the same as entanglement at two
predictors, not confirmation of two independent mechanisms.

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
