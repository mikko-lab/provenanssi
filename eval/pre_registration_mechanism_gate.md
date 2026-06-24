# Pre-registration: Mechanism gate — low-level robustness check
# Does A_dom_broad survive an expanded low-level control set?
# Written: 2026-06-24 — BEFORE any feature computation or slope correlation.
# THIS DOCUMENT IS LOCKED ON COMMIT. No post-hoc changes.

## Context

Update 10 + robustness addendum (commits b6c5fd7, e41682b): A_dom_broad (dominant
foreground subject: prominent animal or human) independently predicts slope above H1
(−β_spec, global spectral slope) and H2 (rho_gt, GT nearest-neighbor autocorrelation),
partial r=+0.661, CI [+0.305, +0.854], LOO 24/24 robust.

The skeptic's objection: H1 and H2 are only two facets of "low-level image texture."
A broader control set could reveal that A_dom_broad is a proxy for an unmeasured low-
level statistic. This session closes that gate.

## ANTI-FISHING RULE

Exactly 3 new low-level control features are defined below, chosen before any slope
correlation is computed. No feature selection. The analysis uses all 5 controls (H1, H2,
C1, C2, C3) in a single pre-stated partial. Sub-models (one control dropped) are
pre-stated for proxy detection only and run only if the full-5 partial fails.

---

## New low-level features (C1, C2, C3)

### C1 — Mean local patch variance (V_patch)

**Operationalisation:** Divide the 256×256 grayscale GT image into non-overlapping 8×8
pixel patches. Compute the variance of pixel values within each patch. V_patch is the
mean of these per-patch variances.

**Why 8×8?** 8 pixels ≈ half the width of a facial feature (eye ~15 px); this scale
captures within-feature texture rather than between-feature luminance gradients.

**Justification as confounder:** Face and portrait images have large smooth regions
(skin, plain backgrounds) that contribute low per-patch variance. V_patch may absorb
the "smooth skin / plain background" signal that also separates face-type images from
textured natural scenes. If A_dom_broad is merely a proxy for "low local smoothness at
the patch scale," V_patch will reveal it.

**Pre-stated prediction direction:** faces/portraits → low V_patch; textures → high
V_patch. Expected to correlate negatively with slope if it absorbs H1/H2's signal.

### C2 — Mean gradient magnitude (G_mean)

**Operationalisation:** Compute the image gradient via Sobel operators (horizontal Gx,
vertical Gy). G_mean = mean(sqrt(Gx² + Gy²)) over all pixels.

**Justification as confounder:** Edge density directly measures "how much fine detail
is present." High G_mean = many strong edges = high-texture image. Face images have
structured edges (at feature boundaries) but large edgeless smooth regions → lower
G_mean than fine-texture images (grass, sand, wood grain). If slope elevation is driven
by edge density (fewer edges → longer null-space correlation → higher slope) rather than
by face-presence semantics, G_mean will absorb A_dom_broad.

**Pre-stated prediction direction:** G_mean negatively correlated with slope (lower edge
density → higher slope).

### C3 — Local variance heterogeneity (V_het)

**Operationalisation:** Using the same 8×8 non-overlapping patches as C1: V_het is the
standard deviation of the per-patch variances.

**Justification as confounder:** V_het measures whether the image has mixed texture
regions (some smooth, some detailed) vs. uniformly textured. Face/portrait images have
high V_het: smooth skin patches coexist with detailed hair, eyes, and clothing. Uniform
texture images (sand, cement, grass) have low V_het. V_het would absorb A_dom_broad if
"mixed smooth/detailed structure" — not the semantic face label — drives slope.

**Note on C1 vs C3:** V_patch captures mean texture level; V_het captures within-image
texture variability. These can dissociate: a uniformly smooth image has low V_patch AND
low V_het; a face has low V_patch (smooth regions dominate) but high V_het (mixed). They
are related but measure distinct properties. Both are included because both are plausible
confounders of A_dom_broad.

---

## The test

**Primary test:** Partial r(A_dom_broad, slope | H1, H2, C1, C2, C3)

Computation: OLS residual partial correlation with 5 control variables.
  df = n − 5 − 2 = 24 − 7 = 17
  se = 1 / sqrt(n − 5 − 3) = 1 / sqrt(16)
  t-critical at df=17 for 95%CI ≈ 2.110

**LOO:** Re-run partial 24 times (n=23 at each drop), df=16, se=1/sqrt(15).

---

## Pre-stated decision rules

### SEMANTICS-ROBUST
Condition: full-5 partial CI excludes 0 AND LOO exclude-0 count = 24/24.
Interpretation: A_dom_broad's independence from low-level statistics survives a
broader skeptical control. The content-driver reading is materially strengthened.
Still no mechanism claim; still a behavioural label.

### PROXY-REVEALED
Condition: full-5 partial CI INCLUDES 0.
Sub-procedure (pre-stated): re-run with each new control dropped one at a time
(4 sub-models with H1, H2, + 2 of C1/C2/C3). If exactly one sub-model's partial
CI excludes 0 while the full-5 does not, the dropped control is the proxy feature.
Verdict: "A_dom_broad is a proxy for [feature]."
Status consequence: the SUPPORTED status in §11 and FINDINGS.md must be DOWNGRADED.
Do not self-edit the status — flag for user review.

If multiple sub-model removals restore CI exclusion → "partially proxy-revealed" or
"entangled at the pairwise level" (see below).

### ENTANGLED
Condition: full-5 partial CI includes 0 AND VIF is high (max VIF > 10) AND no single
control removal restores CI exclusion.
Interpretation: multicollinearity at n=24 with 5 controls makes it impossible to
separate A_dom_broad from the low-level predictors. Honest verdict: "cannot separate
semantic from low-level driver at n=24 with 5 controls."
Report the approximate n required to achieve df≥30 with 5 controls (answer: n≥38).

---

## Collinearity guard

Report for every control predictor:
  - Pairwise Pearson r among all 5 predictors
  - VIF_i = 1/(1−R²_i) for each predictor i in the 5-predictor set
  - VIF > 5 = high; VIF > 10 = severe

If all VIFs are high (>10), the CI width is dominated by multicollinearity inflation,
not genuine independence. In that case: report the point estimate of the partial r
alongside the wide CI, and note that CI inclusion/exclusion is not informative — the
honest verdict is ENTANGLED regardless of the point estimate's sign.

This guard exists because n=24 with 5 controls is near the limit of useful partial
regression. Surviving this test with a wide CI is not the same as SEMANTICS-ROBUST.

---

## Proxy detection sub-models (pre-stated)

If full-5 partial fails, run these four sub-models in order:
  M4a: partial r(A_dom_broad, slope | H1, H2, C2, C3)  [drop C1]
  M4b: partial r(A_dom_broad, slope | H1, H2, C1, C3)  [drop C2]
  M4c: partial r(A_dom_broad, slope | H1, H2, C1, C2)  [drop C3]
  M4d: partial r(A_dom_broad, slope | H1, H2)           [original 2-control model]

Report each sub-model's partial r and CI. Proxy identification rule:
  - If M4x CI excludes 0 but full-5 does not → Cx is the proxy feature.
  - If none of M4a–M4c restore CI exclusion → the collapse is not attributable to
    any single new control; the entanglement is joint.

---

## Scope and standing limitations

- ResShift + BicubicDownsample(4) only
- A_dom_broad is a single-annotator label; no inter-rater validation
- This session tests low-level robustness only; it does not identify a mechanism
- A SEMANTICS-ROBUST verdict does not claim "content causes slope elevation" —
  it claims "content adds predictive information beyond a broader low-level control set"
  at n=24 with 5 controls (which may still be narrow)
