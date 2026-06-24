# Pre-registration: Content-driver robustness checks
# Stress-testing the ONE_INDEPENDENT result from Update 10
# Written: 2026-06-24 — BEFORE any LOO computation.
# THIS DOCUMENT IS LOCKED ON COMMIT. No post-hoc changes.

## Context

Update 10 (commit b6c5fd7) found ONE_INDEPENDENT: partial r(A_dom | −β_spec, rho_gt)
= +0.612, CI [+0.229, +0.831], df=20. Two known vulnerabilities motivated this check:
  1. CI lower bound +0.229 is close to 0; one influential observation could flip it.
  2. A_dom_broad (dominant foreground subject, animal OR human) had higher marginal r
     (+0.800 vs +0.774) and accommodates the natural_caribou counter-case (A_dom=0
     but slope at 100th percentile of A_dom=0 naturals).

## ANTI-FISHING RULE

Only the two pre-stated checks below are run. No feature selection, no additional
predictors, no post-hoc model comparison beyond what is listed here.

---

## CHECK 1 — Leave-one-out robustness of P1 (A_dom | H1, H2)

### Procedure

For i = 1 to 24: drop image i from the n=24 dataset; re-compute P1 on the remaining
n=23 images using the same method (OLS residual partial correlation, Fisher-z CI with
t-critical at df = 23 - 2 - 2 = 19, se = 1/sqrt(23 - 2 - 3) = 1/sqrt(18)).

Report for each drop:
  - P1_i (partial r on n-1=23 images)
  - 95%CI [lo_i, hi_i]
  - whether CI excludes 0 (True/False)
  - the dropped image's name, slope, A_dom, and leverage h_ii (hat-matrix diagonal)

Aggregate:
  - range of P1_i (min, max)
  - count of LOO runs where CI excludes 0 ("exclude_0_count" out of 24)
  - list of images where drop causes CI to INCLUDE 0 ("pivotal images")

### Decision rule (pre-stated)

  ROBUST: exclude_0_count = 24 (CI excludes 0 in ALL 24 drops)
    → A_dom independence is not driven by any single point.

  FRAGILE: exclude_0_count < 24 (one or more drops flip the CI)
    → Report the pivotal image(s) by name. Report their slope, A_dom, leverage h_ii.
    → Downgrade the independence claim to:
       "A_dom independence is conditional on inclusion of [image(s)]; the result
        is not robust at n=24."
    → Do NOT claim A_dom is an independent driver if the result is FRAGILE.

### Leverage computation

h_ii = diagonal of the hat matrix H = Z(Z'Z)^{-1}Z' where Z is the design matrix for
the partial correlation regression (Z = [−β_spec, rho_gt, constant], n×3 matrix,
before computing residuals). High leverage: h_ii > 2k/n = 2×3/24 = 0.25.

---

## CHECK 2 — Re-run partial structure with A_dom_broad as semantic predictor

### Procedure

A_dom_broad = 1 if image contains a prominent non-human animal OR human figure as the
dominant subject (pre-stated annotation, same as Update 10 secondary analysis):
  Differs from A_dom in: natural_caribou → 1, frog_on_log → 1 (total n_1=10 vs 8).

  (a) Re-compute marginal r(A_dom_broad, slope) with 95%CI (Bonferroni α=0.0167).
      Report whether it still CONFIRMS by the pre-registered criterion (Bonferroni
      CI excl. 0 AND directional check). Note: directional check trivially passes
      (portrait paintings A_dom_broad=1, landscape A_dom_broad=0).

  (b) Re-compute P1' = partial r(A_dom_broad | −β_spec, rho_gt) on n=24, with 95%CI.
      Same method as P1 but with A_dom_broad substituted for A_dom.

  (c) Run leave-one-out on P1' (same procedure as Check 1, substituting A_dom_broad).
      Report: exclude_0_count_broad, pivotal images for A_dom_broad.

  (d) Caribou consistency check: under A_dom_broad, natural_caribou is annotated 1.
      Report: does the caribou now fall within the A_dom_broad=1 group's slope
      distribution (mean±std of the other 9 A_dom_broad=1 images)?
      Is caribou's slope consistent with being in the elevated group?

### Decision rule (pre-stated)

  A_dom_broad PREFERRED over A_dom if:
    - P1' CI excludes 0 AND exclude_0_count_broad > exclude_0_count_A_dom
    - AND caribou is consistent with A_dom_broad=1 group
    → Re-describe the driver as "dominant foreground subject (animal or human)."

  NEITHER ROBUST (most likely given n=24):
    - exclude_0_count < 24 for BOTH A_dom and A_dom_broad
    → Honest verdict: "slope tracks a foreground-subject content split, but no single
      semantic feature is established as the independent driver at n=24."

  A_dom MORE ROBUST (unexpected):
    - exclude_0_count_A_dom > exclude_0_count_broad AND A_dom CI excl. 0 in more drops
    → A_dom_broad does not improve robustness; A_dom remains the best candidate but
      still requires larger n for confirmation.

---

## Annotation-reliability caveat (pre-stated, standing limitation)

A_dom and A_dom_broad are single-annotator binary labels (Claude Code, 2026-06-24,
based on image content knowledge). The natural_caribou case shows the annotation
boundary is judgment-dependent: a "prominent foreground caribou" could plausibly be
classified as A_dom=1 if the rule were "prominent foreground vertebrate" rather than
"dominant human." No inter-rater agreement was measured. This is a standing limitation:

  ALL content-driver results (Update 10 and this robustness check) are conditional on
  this specific annotation scheme. The labels should be treated as one operationalisation
  of "face/portrait-type content," not as an objective ground truth.

This caveat must appear in every written verdict that cites A_dom or A_dom_broad as
a predictor.

---

## What would resolve the open questions

The following are PRE-STATED as the resolution paths (not commitments for this session):

1. LOO fragility due to a single face image → need ≥30 face/portrait images to establish
   robustness of the A_dom partial.
2. LOO fragility due to a texture image at the low-slope extreme → need balanced sampling
   across the slope range, not just at the extremes.
3. A_dom vs A_dom_broad question → need images with prominent non-human animals (dogs,
   birds, wildlife close-ups) in the 0.75–0.85 distance range to directly test whether
   animal prominence drives slope similarly to human faces.
4. Mechanism → A_dom is a behavioral label. Confirming whether it reflects training data
   distribution, semantic prior strength, or something else requires probing ResShift's
   internal representations, which is out of scope for this session.

---

## Scope

ResShift + BicubicDownsample(4) only. No new images. No new features. Existing n=24 only.
