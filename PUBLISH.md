# Publication checklist — run these yourself, in order

**Before running anything:** re-read the ImageNet licence flag in the audit report.
If you have not yet resolved the demo image provenance (see Check 2), do that first.

---

## Step 0 — Final local verification (no network required)

```bash
# Confirm the gate is still GREEN
python falsify.py --fast

# Confirm no weight/venv/secret files would be pushed
git ls-files | grep -E '(weights/|\.venv/|\.pth|\.bin|\.DS_Store)'
# Expected: output/calibration_resshift_bicubic_report.txt only (text file, benign)

# Confirm current log looks right
git log --oneline -8
```

---

## Step 1 — Create the GitHub repository

### Option A: gh CLI (recommended)

```bash
gh repo create mikko-lab/provenanssi \
  --public \
  --description "Provenance-aware image reconstruction: labels which pixels are measured vs invented, with calibrated uncertainty." \
  --homepage "" \
  --disable-wiki
```

> **Note:** `--public` makes it immediately visible. If you want to inspect the
> rendered README on GitHub before it's searchable, use `--private` here and
> change visibility manually in Settings → Danger Zone → Change visibility
> after reviewing.

### Option B: Web UI

1. Go to https://github.com/new
2. Owner: your account (or organisation)
3. Repository name: `provenanssi`
4. Description: *Provenance-aware image reconstruction: labels which pixels are measured vs invented, with calibrated uncertainty.*
5. Visibility: Public (or Private to review first — see note above)
6. Do **not** initialise with README, .gitignore, or licence (you have all of these)
7. Click **Create repository**

---

## Step 2 — Add remote and push

```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/provenanssi.git

# Verify the remote before pushing
git remote -v

# Push all commits + set upstream
git push -u origin main
```

After the push, open `https://github.com/YOUR_USERNAME/provenanssi` and verify:
- README renders correctly (Moon Mode hook visible, metrics table, reproduce command)
- No weight files or .venv in the file tree
- Commit count matches local `git log --oneline | wc -l`

---

## Step 3 — Enable GitHub Pages for the demo (optional)

The demo (`demo/index.html`) is standalone and works as a local file. To host it
on GitHub Pages so the LinkedIn post can link directly to it:

### Option A: gh CLI

```bash
# GitHub Pages from a /docs folder or a gh-pages branch is the standard path.
# The cleanest approach for a /demo subfolder is a gh-pages branch:

git worktree add ../provenanssi-pages gh-pages 2>/dev/null || \
  git checkout --orphan gh-pages && git rm -rf . && git checkout main -- demo/

# Simpler: just set Pages source to main branch /demo in the web UI (Option B).
```

> The gh CLI Pages configuration is limited; the web UI is more reliable here.

### Option B: Web UI (recommended for /demo subfolder)

1. Go to the repo → **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: `main` / Folder: `/demo`
4. Click **Save**
5. Wait ~60 seconds; Pages URL will appear as `https://YOUR_USERNAME.github.io/provenanssi/`

The demo `index.html` uses only relative `assets/` paths and works without a build step.

> **Note:** GitHub Pages serves from the folder root, so the URL will be
> `https://YOUR_USERNAME.github.io/provenanssi/` (which loads `demo/index.html`
> if you configure it as the source folder, or you may need to navigate to
> `/demo/index.html` depending on how Pages is configured).

---

## Step 4 — Post-publication check

```bash
# Confirm remote is clean (no accidental extra files)
gh repo view YOUR_USERNAME/provenanssi --json diskUsage,defaultBranchRef

# Spot-check a few key files are visible
gh api repos/YOUR_USERNAME/provenanssi/contents/falsify.py | python3 -c "import json,sys; d=json.load(sys.stdin); print('size:', d['size'], 'bytes')"
gh api repos/YOUR_USERNAME/provenanssi/contents/weights 2>&1 | grep -q "Not Found" && echo "weights/: correctly absent"
```

---

## What NOT to do

- Do not run `git push --force` — the commit history is clean and linear
- Do not add weights (`*.pth`, `*.bin`) to the repo
- Do not commit `.venv/` contents
- Do not enable Actions or workflows that would re-run `falsify.py --full` on every
  push without confirming the runner has the weights available

---

## ImageNet image flag (resolve before publishing)

**See Check 2 in the audit report.**

The three demo source images (`good`, `typical`, `failure`) are ILSVRC2012 validation
patches. Their redistribution licence is ambiguous. Options:

1. **Replace with CC0 images** — safest. Find three 256×256 images from
   Wikimedia Commons (CC0 or CC-BY) or similar, run `eval/build_demo_assets.py`
   with the new filenames, recommit `demo/assets/` and update the manifest.
   The `build_demo_assets.py` DEMO_IMAGES list takes arbitrary filenames; update
   the `filename` and `id` fields there.

2. **Keep ImageNet crops, add a note** — add a `LICENCE.md` (or a note in README)
   stating the demo images are ILSVRC2012 validation patches used under the
   ImageNet non-commercial research terms. Accepted for many academic repos but
   not watertight for all contexts.

3. **Remove `_gt.png` files only** — the reconstructions and overlays are model
   outputs (more defensible); the ground-truth crops are the closest to the
   original images. Removing `{id}_gt.png` removes the clearest reproduction.
   Panel A and C still work; the error map in Panel C would need updating.

**Recommended: Option 1 (replace).** This is a provenance project; having an
unambiguous provenance on your own demo images is the right call.
