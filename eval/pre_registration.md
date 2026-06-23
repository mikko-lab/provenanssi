# Pre-registered Analysis Plan
# Phase 2 — Large-sample resolution of three power-limited threads
# Written: 2026-06-23  BEFORE any Phase 2 data is collected.
# This plan is locked before Phase 2 runs. No post-hoc changes.

## Context

Three threads hit the same statistical-power wall at n=5–11:
1. Distance→slope (Update 4): r(ResNet50_dist, slope), n=11 non-synthetic in 3 clusters
2. Slope→noise-floor α (Update 6): power law fit, n=5 images
3. Coherence test (Update 7): r(rho_nn, slope), n=5 images; girl_sad_face anomaly

One larger balanced sample resolves all three.

## Sample

Target: all non-synthetic images available after Phase 1 assembly.
Exclusions:
- `soft_blobs`, `noise_gauss50`, `checker32`, `hard_shapes` (synthetic/generated)
- Any image where N=48 calibration fails to converge (r < 0.90)
- Any image where null_frac_gt < 0.001 (degenerate null space)

Primary set: all remaining non-synthetic images (expected n=25–35).

## Thread 1: distance→slope continuous relationship

### Measurement
- dist_i = ResNet50 cosine distance to 16-image ILSVRC2012 centroid (same metric as Update 4)
- slope_i = OLS calibration slope at N=48 (same protocol as prior measurements)

### Tests
A. Pearson r(dist, slope) + Fisher-z 95% CI + n — ALL non-synthetic images
B. Pearson r(dist, slope) + CI — EXCLUDING paintings (to test sensitivity to new category)
C. Partial r(dist, slope | null_frac_gt) + CI — controls for null energy confound
D. Spearman rho(dist, slope) + p — rank-based (non-parametric, same as Update 4)

### Decision rule
- (a) POSITIVE if: r < 0 (negative expected — higher dist → lower slope is the hypothesis
  based on Update 4), AND CI excludes 0 in tests A AND C.
  NOTE: the hypothesis is that slope is HIGHER for images CLOSER to ImageNet (faces),
  so we expect r(dist, slope) < 0. Recode if needed.
- (a-weak) if: r in predicted direction, CI includes 0, but p < 0.10 (one test)
- (b) NOT SUPPORTED if: r near zero or in wrong direction, CI includes 0

Pre-stated expected direction: r(dist, slope) < 0 (nearer ImageNet → higher slope).
EXCEPTION if paintings turn out near ImageNet but low slope — recheck mechanism.

### Within-group test (new: n≥6 per group)
E. For each group with n≥6: report within-group r(dist, slope)
   Decision: if within-group r is near zero for ALL groups but between-group r is large
   → it's a group-level effect, not continuous. Report explicitly.

### Dissociation test (added 2026-06-24, BEFORE Phase 2 run, pre-registered)

Motivation: the Phase 1 distance distribution shows faces (0.72–0.81), paintings (0.76–0.83),
and naturals (0.78–0.93) overlap substantially, while texture separates clearly (0.90–0.94).
A raw r(dist, slope) across all groups mostly re-tests "texture vs rest."
The sharper question this sample CAN answer: do groups at the SAME distance differ in slope?

Test F (dissociation):
- Restrict to images in the overlapping distance window: 0.72 ≤ dist ≤ 0.87
  (this excludes textures and most high-dist naturals, keeping only the mixed-category region)
- Within that window, run Kruskal-Wallis test on slope across categories
  (faces / paintings / naturals — groups with ≥2 images in the window)
- Report: KW statistic + p; per-group slope mean ± std; effect size η²
- Also report: pairwise Mann-Whitney U for faces vs paintings and faces vs naturals
  (two pre-stated pairs only; painting vs natural is post-hoc if chosen after seeing data)

Decision rule:
- DISSOCIATION CONFIRMED: KW p < 0.05 AND largest pairwise mean slope difference > 0.5
  → slope varies where distance does not → distance does NOT drive slope at this scale
  → Thread 1's continuous hypothesis is falsified within the overlapping range
- BORDERLINE DISSOCIATION: KW p < 0.10 AND mean difference between highest/lowest-slope
  group > 0.5 → consistent with dissociation, inconclusive
- NULL (no dissociation detected): KW p ≥ 0.10 OR mean differences < 0.5
  → cannot distinguish groups at same distance → consistent with (but not confirming)
  a continuous distance effect
- ANTI-RESULT: If slope ordering across groups MATCHES distance ordering even within the
  window → interpret cautiously, but note the window may be too narrow to differentiate

## Thread 2: slope→noise-floor α re-estimation

### Measurement
- slope_i at N=48 (as above)
- noise_floor_std_i = std of β̂ across 5 independent N=12 windows (same protocol as Update 6)

### Test
Power law fit: log(noise_floor_std) = log(a) + α·log(slope), OLS
Fit on: non-synthetic images EXCLUDING soft_blobs and any image with slope < 0.2

### Decision rule (α confidence interval)
- Definitive: t-CI for α excludes 0 AND is entirely below 1.0 (α<1 = sub-proportional)
- Provisional: CI excludes 0 but includes 1.0
- Null: CI includes 0

Pre-stated H0: α=0 (OLS scale-invariance under iid pixels)
Pre-stated H1: α>0 (spatial correlation of reconstruction errors)
Pre-stated expected value from Update 6: α≈0.35–0.50 (sub-proportional)

## Thread 3: coherence→slope, coherence→noise-floor mediation

### Measurement
- rho_nn_i = mean nearest-neighbor ACF of normalized null-space deviations (same as Update 7)
  N=12 ensemble per image, 1 run (no repeats for coherence — the ACF is stable per run)

### Tests
F. r(rho_nn, slope) + CI + n
G. r(rho_nn, noise_floor_std) + CI + n
H. Partial r(slope, noise_floor_std | rho_nn) — mediation test

### Decision rule
For F: (a) if r > 0 and CI excludes 0; (a-weak) if r > 0.2 and CI includes 0; (b) if r ≤ 0.
For mediation H: confirmed if |partial_r| < 0.7 × |direct_r(slope, nf_std)|

### Girl_sad_face anomaly
Update 7: girl_sad_face had rho_nn=0.154 despite slope=2.80 (lower than wayuu_woman 0.245,
slope=1.58). With n≥6 faces, test:
- Within-faces: r(rho_nn, slope) + CI
- Decision: if CI still includes 0 within faces → anomaly is noise (n=5 too small)
             if within-faces r < 0 → coherence and slope are decoupled within faces

## Reporting format

All results reported as:
  r = X.XXX  CI [±XX.XX%, df=n-2, t-corrected]  n=XX  verdict

No selective reporting: ALL pre-stated tests reported, even nulls.
A definitive NULL (CI clearly includes 0 with n≥20) is a full result.

## Anti-fishing clause

This plan was written before Phase 2 data collection. If the data suggest
additional interesting tests, those are flagged as POST-HOC EXPLORATORY and
not included in the confirmatory results table. The four pre-stated tests
above are the only ones that count for the confirmatory verdict.

## Statistical thresholds

α = 0.05 (two-tailed) for all confirmatory tests.
CI: Fisher-z with t-critical value at df=n-2 (t-corrected, not z=2).
Partial r: computed from OLS residuals (standard partial correlation).
No multiple-comparison correction (four pre-stated tests on the same dataset
are correlated, and the primary question is pattern, not individual p-values).
