# nicolascayuela.github.io

Personal academic website of **Nicolas Cayuela**, PhD student in phononics & metamaterials at Institut Jean Lamour (UMR CNRS 7198, Nancy) - live at **<https://nicolascayuela.github.io/>**.

Built with Jekyll on GitHub Pages, bilingual EN/FR, with an interactive ML/physics playground running entirely in the browser.

## Playground

| Demo | What it shows | Tech |
|---|---|---|
| Phonon dispersion | 1D diatomic chain (Bragg) vs mass-in-mass chain (local resonance): band gaps, complex band structure kr/ki, animated Bloch modes | pure JS, after Laude, *Phononic Crystals* |
| CIFAR-100 latent space | 3D point cloud of 1000 real CIFAR-100 images, nearest-image lookup | canvas 3D |
| Word embeddings | GloVe 6B 50d, 20k words, neighbour search and vector arithmetic (paris - france + italy = rome) | pure JS |
| Game of Life / Forest fire | classic cellular automata | pure JS |
| Symmetry groups | drawing pad replicating strokes under the 17 wallpaper / 7 frieze groups | pure JS |
| Gradient descent | SGD / Momentum / Adam rolling down 3D loss surfaces | canvas 3D |
| Dog Creator | convolutional autoencoder + PCA on AFHQ dog faces, 64 latent sliders | PyTorch → ONNX, onnxruntime-web (WebGPU) |
| Dog Diffusion | DDPM (eps-prediction UNet, 32x32) trained on AFHQ dogs, animated DDIM sampling | PyTorch → ONNX, onnxruntime-web (WebGPU) |
| Style transfer | fast neural style (Starry Night) + AdaIN arbitrary style with 23 public-domain paintings, up to 384px | PyTorch → ONNX, onnxruntime-web (WebGPU) |

All models were trained on CPU with PyTorch (see `scripts/`) and exported to ONNX; inference runs client-side, no backend.

## Repository layout

```
_data/            profile, navigation (EN/FR)
_includes/        layout partials + playground widgets
_layouts/         default layout (SEO meta, JSON-LD)
_publications/    publication entries
assets/js/        one self-contained module per demo
assets/models/    ONNX models + JSON data used by the demos
scripts/          PyTorch training / export pipelines
```

## Training pipelines (`scripts/`)

- `build_dog_creator.py` - autoencoder (perceptual VGG loss) + PCA export
- `build_dog_diffusion.py` - DDPM training (EMA, cosine LR) + ONNX export
- `export_adain.py`, `export_starry_night.py` - style-transfer ONNX exports
- `build_word_embeddings.py` - GloVe subset + 3D PCA
- `build_cifar.py` - CIFAR-100 sprite + coordinates

Data sources: [AFHQ](https://arxiv.org/abs/1912.01865) (dogs), [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), [GloVe](https://nlp.stanford.edu/projects/glove/). Style-transfer weights: [igreat/fast-style-transfer](https://github.com/igreat/fast-style-transfer) and [naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) (MIT). Paintings via Wikimedia Commons (public domain).

## Local development

```bash
bundle install
bundle exec jekyll serve     # http://localhost:4000
```

Deployment: GitHub Actions builds and publishes on every push to `main`.
