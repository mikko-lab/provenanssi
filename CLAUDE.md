# CLAUDE.md — Provenanssi

**Working title:** Provenance Layer for Generative Image Restoration
**Repo (suggested):** `provenanssi` (rename freely)
**Type:** Research project. Distinct from commercial products.
**Author:** WP Saavutettavuus (Y-tunnus 3404806-1)
**Goal:** Research project: measure, label, and calibrate provenance in AI image
restoration — separating what the input forces (measured/recovered) from what the
model invented, with calibrated uncertainty.
**Domain:** Computational photography / image restoration / trustworthy AI.

---

## 0. One-line pitch

When a generative model "enhances" an image, most of the result is invented, not
recovered. This project ships a **deterministic layer that labels every output pixel
as `measured`, `recovered`, or `invented`** — separating what the input forces from
what the prior guessed. "Liputa, älä piilota," applied to imaging.

This is the same signature as qubit-harness (a deterministic safety/observability
layer wrapped around a probabilistic engine), moved into the computational-imaging
domain — the same domain that mobile-camera AI research (Huawei et al.) lives in.

### Competitive thesis ("better than Huawei")

We do **not** try to beat a production camera on image quality — that needs a research
team, years, and proprietary NPU data. We beat them on the axis where they *structurally
cannot compete*: **honesty about what the model invented.** A consumer camera's
incentive is to make fabricated detail look as real as possible ("industry-leading image
experiences"). This layer does the opposite — it exposes the fabrication and quantifies
it. That is a stronger research framing than "my upscaler is also sharp," and it is a
claim we can actually defend.

**Honesty boundary (do not cross):** the goal is not "this beats Huawei." It is "this
wins the one dimension Huawei leaves empty — provenance." Keeping the claim that narrow
is what keeps it true (see R2). Do not let the README inflate it.

---

## 1. The falsifiable claim (this is the whole project)

> **Claim.** For a known linear degradation operator `A` and a model estimate `x̂`,
> the range-space component `A⁺A·x̂` is *determined by the input* and the null-space
> component `(I − A⁺A)·x̂` is *fabricated by the prior*. Pixels dominated by null-space
> energy carry no input-derived information about ground truth beyond what the prior
> supplies.

**How to kill it.** Run on synthetic data where we own the ground truth `x` (we
downsample it ourselves to make `y = A·x`):

- `measured`-flagged regions MUST reconstruct `x` to within tolerance `ε` and MUST be
  invariant to the random seed. If they are not → operator or rectification is wrong →
  claim falsified for that configuration. **Fix the code, do not loosen the claim.**
- `invented`-flagged regions MUST vary with the seed and MUST NOT systematically
  converge to `x` as we add more samples. If they reliably converge to GT, the region
  was actually constrained and we mislabeled it.

If the demo cannot survive this test on at least four operator types (bicubic SR, box
downsample, deblur, inpaint), the project is not publishable. Say so.

---

## 2. Scientific guardrails (R1–R10)

These are hard constraints. Claude Code does not relax them to make a result look
better. If a guardrail and a "nicer demo" conflict, the guardrail wins.

- **R1 — Known operator only for hard claims.** The `{measured/recovered/invented}`
  *labeling* is only asserted when `A` is known (synthetic degradation). For real
  photos with unknown `A`, we may show the **ensemble-variance uncertainty map only**,
  explicitly labeled "approximate provenance, operator unknown." Never present the
  hard three-way label on a real photo as if `A` were known.

- **R2 — `invented` ≠ `wrong`.** A null-space pixel can be plausible and even
  accidentally correct. The flag means *"not determined by the input,"* not *"false."*
  All UI copy and README must state this. This is the honest core of the project.

- **R3 — Data consistency is enforced, not hoped for.** Use range-null rectification
  (DDNM-style): at the end, `x_out = A⁺·y + (I − A⁺A)·x̂`. Verify numerically that
  `‖A·x_out − y‖ ≤ ε`. Log the residual. If it exceeds `ε`, fail loudly.

- **R4 — Ground truth is held out.** When GT exists (synthetic), it is used ONLY for
  evaluation, never fed to the model or the layer. No leakage.

- **R5 — Reproducibility.** Every figure is regenerable from a seed + config file.
  Seeds, operator definition, and model checkpoint hash are logged with each run.

- **R6 — No cherry-picking.** Report a fixed evaluation set. Include at least one
  failure case in the README (e.g. a region the layer mislabels, or an operator where
  separation is weak). A portfolio piece that hides its failure modes is a liability.

- **R7 — Operator math is unit-tested.** `A`, `A⁺`, and the projector `A⁺A` are
  covered by tests asserting `A·A⁺·A = A` and idempotence `(A⁺A)² = A⁺A` to numerical
  tolerance, per operator. The science rests on these being correct.

- **R8 — Quantify the map, don't just colour it.** The provenance map must come with
  numbers: per-image fraction of null-space energy, calibration of the uncertainty map
  against actual error on synthetic data (reliability curve). Pretty overlay + no
  metric = not done.

- **R9 — Scope honesty.** Linear `A` only in v1 (SR, Gaussian/motion deblur with known
  kernel, inpainting mask). Nonlinear/unknown ISP degradation is explicitly future
  work, named as such. Do not imply the method handles real camera pipelines yet.

- **R10 — Accessibility is an engineering criterion.** The provenance overlay must not
  rely on colour alone (R2 of WCAG-think): use colour + texture/hatching + a togglable
  legend, AA contrast. The demo is also a WCAG-aware artifact, consistent with the
  rest of the portfolio.

---

## 3. Architecture

Two-layer split, mirroring turve-InSAR (engine vs. review checkpoint):

```
┌─────────────────────────────────────────────────────────┐
│  PROBABILISTIC ENGINE  (the restorer)                     │
│  pretrained diffusion / SR model — treated as a black box │
│  produces estimate x̂ (and an ensemble {x̂_1..x̂_N})         │
└───────────────────────────┬───────────────────────────────┘
                            │  x̂, ensemble, y, A
                            ▼
┌─────────────────────────────────────────────────────────┐
│  DETERMINISTIC PROVENANCE LAYER  (the contribution)       │
│  • range-null decomposition         → measured vs invented│
│  • DDNM-style rectification         → enforce A·x_out = y │
│  • ensemble variance                → epistemic uncertainty│
│  • per-pixel label {meas/recov/inv} → provenance map      │
│  • calibration + metrics            → numbers, not vibes  │
└───────────────────────────┬───────────────────────────────┘
                            ▼
              provenance map + uncertainty map + JSON report
```

The engine is interchangeable; the **layer is the product.** Keep that boundary clean
in the code so the README can say "works with any linear-inverse restorer."

### Module breakdown

```
provenanssi/
├── CLAUDE.md                      # this file
├── README.md                      # public-facing: claim, method, failure case
├── FINDINGS.md                    # research memo: domain-shift investigation
├── PUBLISH.md                     # publication checklist (dissemination deferred)
├── falsify.py                     # project gate: exit 0 = FALSIFY: GREEN
├── operators/
│   ├── base.py                    # Operator ABC: forward A, pinv A⁺, projector A⁺A
│   ├── bicubic.py                 # BicubicDownsample — primary calibration operator
│   ├── superres.py                # BoxDownsample
│   ├── deblur.py                  # CircularBlur (known-kernel convolution + pinv)
│   └── inpaint.py                 # MaskOperator (masking, trivial pinv)
├── engine/
│   ├── base.py                    # Engine ABC
│   ├── oracle.py                  # OracleEngine (test / calibration baseline)
│   ├── resshift.py                # ResShiftEngine — primary: conditional diffusion SR
│   ├── real.py                    # RealEngine
│   ├── esrgan.py                  # ESRGANEngine
│   └── diffusion_ddpm.py          # DDPMDDNMEngine
├── layer/
│   ├── decompose.py               # range/null split, rectification (R3)
│   ├── ensemble.py                # N-sample variance → uncertainty map
│   ├── classify.py                # per-pixel {measured/recovered/invented}
│   └── calibrate.py               # reliability curve vs GT (R8)
├── eval/
│   ├── calibrate_resshift.py      # primary calibration (16 ImageNet images, N=6)
│   ├── calibrate_domain_shift.py  # domain-shift experiment across 4 CC0 groups
│   ├── stability_nscan.py         # N-scan + noise-floor measurement
│   ├── close_findings.py          # wood_grain convergence + generalised noise floor
│   ├── faces_noise_floor.py       # final natural↔faces noise floor measurement
│   ├── build_demo_assets.py       # generates demo/assets/ from eval outputs
│   └── research_sources/          # CC0/PD source images used in experiments
├── tests/
│   ├── test_operators.py          # R7: A·A⁺·A=A, projector idempotence, per operator
│   ├── test_layer.py              # layer integration tests
│   ├── test_calibrate.py          # calibration unit tests
│   └── test_real_engine.py        # RealEngine tests
└── demo/
    ├── index.html                 # interactive demo (static, self-contained HTML)
    ├── assets/                    # pre-computed provenance maps + calibration data
    └── ASSETS_LICENCE.md          # CC0/PD provenance for demo source images
```

---

## 4. The math, concretely (so Claude Code implements the right thing)

For a linear degradation `y = A·x`:

- **Range-null decomposition** of any estimate `x̂`:
  `x̂ = A⁺A·x̂ + (I − A⁺A)·x̂`
  - `A⁺A·x̂` — **range/row space**: the part the measurement constrains.
  - `(I − A⁺A)·x̂` — **null space**: what `A` is blind to. **All fabrication lives here.**

- **Rectification (DDNM-style, R3):** force the data-consistent solution
  `x_out = A⁺·y + (I − A⁺A)·x̂`
  → guarantees `A·x_out = A·A⁺·y = y` (noiseless), so the range part is *measured*,
  the null part is *invented*, by construction.

- **Per-pixel provenance score:** map the local null-space energy (e.g. magnitude of
  `(I − A⁺A)·x̂` in a neighbourhood, normalised) to a continuous "invented-ness," then
  threshold into the three labels. `recovered` = range-space content that is non-trivial
  (e.g. genuine multi-frame/structural recovery), distinguished from `measured` =
  directly present in `y`.

- **Ensemble uncertainty:** draw `N` seeds → `{x̂_i}` → rectify each (they share the
  same range space by R3) → per-pixel variance is **purely null-space disagreement** =
  clean epistemic uncertainty. Calibrate it against actual error on synthetic data (R8).

Reference family (for the README, paraphrased — do not copy text): DDRM, DPS, and
DDNM established null-space data-consistency for diffusion restoration. **Our novelty
is not the decomposition — it's exposing the null-space component to the user as a
provenance/trust artifact, with calibration and an accessible overlay.** State this
honestly; it's a stronger story than pretending the math is new.

---

## 5. The deliverable / demo (the "money shot")

A single interactive page. The two centrepieces are the **side-by-side comparison** and
the **calibration proof** — these are what make it "better than Huawei," not the
restoration quality.

### 5a. The headline view — "Consumer mode vs Provenance mode"

Two panels, same restored image, side by side:

- **Left — "Consumer mode":** the polished result exactly as a phone would present it.
  No flags, no caveats. Looks great. This is what Huawei ships.
- **Right — "Provenance mode":** the identical result, with every invented pixel flagged
  (colour + hatching, R10) and the null-space energy fraction shown as a number.

Same pixels, two levels of honesty, in one screen. This single comparison *is* the
thesis — it shows what the consumer pipeline hides, without claiming to out-restore it.
A slider/wipe between the two is the strongest single interaction on the page.

### 5b. The proof view — calibration (this is what separates it from a toy)

A reliability curve: on synthetic data (known `A`, held-out GT), does the uncertainty
map actually predict real error? Bin pixels by predicted uncertainty, plot mean actual
error per bin. A well-calibrated diagonal means the flags are *measured*, not decorative.

This is the view no consumer-camera marketing material ever shows, and it is the
difference between "nice overlay" and "defensible result." Make it prominent, not a
footnote (R8). If calibration is poor, show it honestly and explain why (R6) — a shown
weakness is more credible than a hidden one.

### 5c. Full panel list

1. **The setup** — sharp GT, our synthetic downsample `y`, the operator `A` named.
2. **Consumer mode vs Provenance mode** — the §5a headline wipe. The reveal.
3. **Honesty panel** — flip seed; measured stays put, invented changes (R1/R2 live).
4. **Calibration** — the §5b reliability curve. The proof.
5. **Failure case** — one example where separation is weak; explain why (R6).
6. **Real photo, honest mode** — uncertainty-only map, labeled "operator unknown" (R1/R9).

Build the page per `frontend-design` skill conventions; WCAG AA, colour-independent
overlay (R10). Keep CSS/JS inline if it's a single HTML artifact.

---

## 6. Tech stack

- Python + PyTorch for engine + layer (PyTorch is the lingua franca of this domain).
- A small pretrained diffusion or SR checkpoint — pick the smallest that demos cleanly;
  the layer is the point, not SOTA restoration quality.
- NumPy for operators; operators kept as explicit linear maps where feasible so `A⁺`
  and the projector are testable (R7).
- Web demo: static HTML/JS (precompute results offline, ship the maps as assets) so the
  page works without a GPU backend — same deployment pattern as your other demos.

---

## 7. Dissemination — deferred

Publishing is explicitly not a current goal. `linkedin-draft-provenanssi.md` is
retained on disk as an artifact only, not a plan.

---

## 8. Explicitly out of scope for v1

- Nonlinear / unknown real-camera ISP degradation (named future work, R9).
- Training or fine-tuning a restorer (we wrap a pretrained one).
- Claiming the decomposition itself is novel (it isn't — see §4).
- Beating SOTA restoration metrics (irrelevant to the contribution).

---

## 9. Working notes — COMPLETE

All original working notes are done. See §10 for current research status.

---

## 10. Research status

**Full record: FINDINGS.md.** This section summarises. FINDINGS.md is the authority.

### Core result — certified, unchanged

Four operators (`BicubicDownsample`, `BoxDownsample`, `MaskOperator`, `CircularBlur`)
pass `A·A⁺·A = A` at 1e-12. ResShift (conditional diffusion SR) is the primary engine.
Calibrated on 16 held-out ImageNet patches, BicubicDownsample(4):

| Metric | Value | Threshold | Status |
|---|---|---|---|
| Pearson r | **+0.9667** | ≥ 0.90 | ✓ PASS |
| Slope | **1.5301** | 0.5 – 2.0 | ✓ PASS |
| ECE | **0.0282** | ≤ 0.30 | ✓ PASS |

`python falsify.py --full` at anchor commit **83ab9cd** reproduces these numbers. This
is the only finished, citable result.

### Domain-shift investigation — CLOSED

The following is what was found and what was retracted. Both carry equal weight.

**What holds (narrow):** Pearson r is robust across all tested domains and all N values
— r ≥ 0.94 everywhere. Calibration slope varies per image above sampling noise: the
natural↔faces group contrast (N=48 means: natural 1.1874, faces 2.5715, difference
1.3841) is **9.2× the worst in-contrast noise floor** (boy_face N=12 range = 0.1497,
measured directly from the faces group's own noise floor). Slope is image-dependent
beyond noise. The 9.2× figure is a **lower bound**: the mechanism analysis (§10 below)
established that the faces noise floor is partly inflated by slope magnitude, making
the denominator conservative.

**Why the grouping is weak:** "faces vs natural" is a weak domain label. The
within-faces slope span at N=48 is 1.75–3.18, which exceeds the full natural range
0.95–1.42. The effect is more accurately described as "slope is image-dependent beyond
noise, face images tend higher" — not a validated distribution-shift detector. Do not
extend this to "distribution-dependent calibration" without a rigorous distance metric.

**Withdrawn (equal weight to the above):**
- **Texture overconfidence = N artifact.** wood_grain slope at N=6 was 0.16; at N=192
  it converges to ~0.60, inside IS_CALIBRATED. The claim that texture is systematically
  underconfident was retracted.
- **25× and 1.5× SNRs = wrong reference choices.** 25× used boardwalk as a universal
  reference (too optimistic). 1.5× used soft_blobs from the synthetic group (wrong
  regime, too pessimistic). Both retracted. The honest figure is 9.2×.
- **Inverse energy→noise curve = confounded by slope magnitude.** The pattern "higher
  null_frac_gt → lower noise floor" was confounded: face images with similar null
  energy to boardwalk but ~3× higher slope had 1.5–1.8× higher noise floors. Null
  energy alone does not predict the noise floor. The mechanism is now characterised
  (see slope-noise mechanism below): spatial correlation of reconstruction errors
  reduces effective sample size n_eff for high-slope images.

**If you read this section and conclude we have a domain-shift detector or a clean
group-level result, re-read the withdrawals above.** The honest summary: slope is
image-dependent above noise, grouping is weak, one contrast measured, three prior
claims retracted.

### Distance-metric thread — SHELVED (not closed)

Three attempts to replace intuitive grouping with a rigorous per-image distance metric:

1. **VQ-encoder cosine distance** (ResShift autoencoder_vq_f4): null result — metric
   measured compressibility, placed synthetics closer to ImageNet centroid than naturals.
2. **ResNet50 penultimate-layer cosine distance** (IMAGENET1K_V2, 2048-dim): borderline
   signal in non-synthetic split (Spearman ρ=−0.736 p=0.010, n=11), but the correlation
   reflects group identity (3 clusters × 3–5 images) not a continuous distance effect.
   Partial r controlling null_frac has CI including 0.

The bottleneck is **statistical power**, not the metric. n=11 non-synthetic images in 3
tight clusters cannot separate a continuous distance effect from a group-level one.
Scripts and data are preserved; resume recipe in FINDINGS.md Update 5.
Do not restart this thread without a dedicated block (~60–100 images, ~1.5h GPU time).

### Slope→noise-floor mechanism — RESOLVED (α estimate revised, see FLAG)

The calibration slope noise floor scales as slope^α, not as an OLS estimator artifact.
Analytic derivation shows α=0 under iid pixels. Empirical fit at n=24 (Phase 2,
pre-registered, 2026-06-24): **α̂=0.77, CI [0.51, 1.04]** — CI excludes 0 (H0 rejected),
but CI includes 1.0 (sub-proportional claim NOT confirmed at n=24). Mechanism: spatial
correlation of null-space deviations (rho_nn > 0, now confirmed at n=24) reduces n_eff.
Full derivation: FINDINGS.md Update 6; script: `eval/slope_noise_mechanism.py`.

**⚠ FLAG — §10 SUPERSEDED ESTIMATE:** Update 6 reported α̂=0.36, CI [0.07, 0.64] at
n=5. Phase 2 (n=24) gives α̂=0.77, CI [0.51, 1.04]. The n=5 estimate was severely
underpowered and biased. Do NOT use α≈0.35–0.5 or "sub-proportional" in any claims.
See FINDINGS.md Update 8 (Thread 2) for full account.

**⚠ FLAG — §10 "9.2× SNR" UNAFFECTED:** The 9.2× noise-floor signal-to-noise ratio
(natural↔faces contrast) was computed from within-group noise floors, not from α. It
remains valid as reported.

---

## 11. Open / not yet attempted

- **Rigorous distribution-distance metric — SHELVED, not closed.** Three attempts made
  (VQ-encoder: null; ResNet50 group-level: borderline). The bottleneck is statistical
  power (n=11 non-synthetic in 3 clusters). Resume recipe in FINDINGS.md Update 5.
  Any future claim that slope tracks distribution distance on a *continuous* scale still
  requires a properly powered study (~60–100 images, balanced across the distance axis).

- **GT-free shift detection.** Whether domain shift can be detected from slope alone,
  without ground-truth images, was never attempted. Prerequisites: (a) texture noise
  floor — now characterised (wood_grain converges to ~0.60, inside calibrated window);
  (b) rigorous distance metric — still open (shelved). GT-free detection is still
  premature; the group-level signal is established but a detector needs a metric.

- **Content-type vs distance as driver of slope — RESOLVED (Update 8).** Phase 2
  (n=24, pre-registered) confirmed DISSOCIATION: at the same ResNet50 distance (0.72–0.87),
  face photographs and portrait paintings have mean slope ~2.1 while landscape naturals
  have mean slope ~1.3 (KW p=0.019). Distance is a correlate (r=−0.689, n=24), but
  content type is the proximate driver. "Face-like representational content" — including
  portrait paintings — elevates slope regardless of distance.

- **Faces-group generality — PARTIALLY RESOLVED.** Phase 2 includes 5 diverse face images
  (3 Wilfredor CC0 + 2 new: face_red_hair, face_algerian). All 5 have slopes 1.60–3.18,
  consistently above naturals. Slope elevation is not specific to one photographer.
  Within-faces coherence→slope analysis remains underpowered (n=5, CI spans [−0.86, +0.90]).

- **Slope-magnitude → noise-floor mechanism — RESOLVED (α provisional).** α>0 confirmed
  at n=24. α̂=0.77, CI [0.51, 1.04]. Sub-proportional (α<1) not confirmed. See §10 FLAG.

- **Coherence mechanism — CONFIRMED (a) POSITIVE at n=24.** r(rho_nn, slope)=+0.441,
  CI [+0.046, +0.717], n=24 (Update 8). Upgraded from (a-weak) at n=5 (Update 7).
  Mediation (rho_nn screens off slope→nf_std) NOT confirmed: partial r=+0.515 >
  threshold 0.445. See FINDINGS.md Update 8 and `eval/phase2_analysis.py`.
