# Demo asset image licence

All source images used to generate the demo assets (`demo/assets/`) are either
CC0 1.0 Universal (public domain dedication) or released into the public domain
by the copyright holder. No ImageNet or dataset-derived images are included.

---

## Source images

### good — grass_meadow.png

- **Source file:** `Grass_texture.jpg`
- **Wikimedia Commons page:** https://commons.wikimedia.org/wiki/File:Grass_texture.jpg
- **Author:** Titus Tscharntke
- **Licence:** Released into the public domain by the copyright holder
- **Dimensions (original):** 3888 × 2592 px; centre-cropped and resized to 256 × 256
- **Description:** Close-up grass texture photograph

### typical — dirt_soil.png

- **Source file:** `Dirt_Texture.jpg`
- **Wikimedia Commons page:** https://commons.wikimedia.org/wiki/File:Dirt_Texture.jpg
- **Author:** Nathan Anderson (Wikipedia username: Nathanan)
- **Licence:** [CC0 1.0 Universal Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/)
- **Dimensions (original):** 4362 × 2737 px; centre-cropped and resized to 256 × 256
- **Description:** Photograph of dirt/soil surface, Washington DC, March 2020

### failure — wood_grain.png

- **Source file:** `Gfp-wood-texture.jpg`
- **Wikimedia Commons page:** https://commons.wikimedia.org/wiki/File:Gfp-wood-texture.jpg
- **Author:** Yinan Chen (user Goodfreephotos_com)
- **Licence:** Released into the public domain (Creative Commons Public Domain Dedication)
- **Dimensions (original):** 3024 × 2160 px; centre-cropped and resized to 256 × 256
- **Description:** Wood grain texture photograph

---

## Derived assets

All files in `demo/assets/` are derived from the three source images above through
the provenance pipeline (BicubicDownsample ×4, ResShift ensemble, rectify, classify).
The derivation is reproducible via `eval/build_demo_assets.py` at the commit anchor
documented in `README.md`. No ground-truth ImageNet images are included.

## Source images location

The three 256 × 256 PNG source files are in `eval/demo_sources/` and are tracked
by git. The full-resolution originals are on Wikimedia Commons at the URLs above.
