# Can a CNN tell galaxy-formation models apart from their spatial maps?

My final project for **Ay 119** (Caltech, 2026). It asks whether a small convolutional
neural network can distinguish three explanations for JWST's excess of bright galaxies
at z ≈ 10, using only where the galaxies sit on the sky, in the
Lumina simulations.

## Motivation

JWST has found more UV-bright galaxies at z ≳ 10 than pre-launch ΛCDM models predicted.
Very different explanations reproduce the same UV luminosity function, so galaxy counts
alone can't separate them. Their **clustering**, and so their cosmic variance, might.
This project tests that idea with a classifier on mock survey maps.

The three models compared:

| Label | Model | What it stands for |
|---|---|---|
| A1 | Lumina ΛCDM with every galaxy brightened by 1 mag (`delta_MUV = 1`) | enhanced star-formation efficiency |
| C1 | Lumina ΛCDM with 1 mag of scatter in M_UV at fixed halo (`sigma_UV = 1`) | bursty / stochastic star formation |
| EDE | Lumina EDE with its own UV magnitudes | early dark energy cosmology |

## Method

1. **Fields.** A JWST NIRCam-like field of 4.4′ × 2.2′ over 9.2 < z < 10.9
   (12.4 × 6.2 × 369 cMpc) tiles the simulation box.
2. **Maps.** One original field holds only 2 to 4 galaxies with M_UV < −19.5, so each map
   joins 4 × 4 neighbouring fields (≈ 49.5 × 24.7 cMpc) and bins galaxy counts onto a
   32 × 16 pixel image. That gives 200 maps per model.
3. **Matching.** All three samples are cut to the same number of galaxies, so the network
   can't win by counting.
4. **Split.** Whole x-columns of the map grid go to train (7), validation (1) and test (2),
   so neighbouring regions never leak between sets.
5. **Model.** Two conv layers (12 and 24 channels) with ReLU and max-pooling, global average
   pooling and a linear head. Input is `log1p(counts)`, standardised with the training mean
   and standard deviation. Adam, learning rate 1e-3, 30 epochs, best validation epoch kept.
6. **Controls.**
   - *Count only:* logistic regression on the total galaxy count per map.
   - *Shuffled:* the same pixels in random positions, which keeps the counts but destroys
     the spatial pattern. The full maps have to beat this to show the CNN learned clustering.
7. Three-class run, a 5-seed sweep, and two binary comparisons (C1 vs EDE, C1 vs A1).

## Status

In progress. The pipeline, baseline CNN and controls are built and running. Results are not
shown here because they use the Lumina EDE simulation, which is not public yet.

## Known issues and next steps

- [ ] **Debug the baseline training.** Re-check the preprocessing (log1p + standardisation),
      since a normalisation change may never have been saved, and confirm the network can
      overfit a small training set.
- [ ] **Small sample.** 200 maps per model, and only 40 test maps per class. Use more
      independent fields, other line-of-sight projections, or more simulation volumes.
- [ ] Try other magnitude limits (−20, −20.5 are already in the cache), map sizes and resolutions.
- [ ] Compare with summary statistics such as counts-in-cells or the two-point function.
- [ ] Try other architectures, for example a transformer on galaxy point sets.
- [ ] Move the notebook logic into scripts so runs are reproducible from the command line.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook spatial_maps.ipynb
```

The simulation data is **not** included. The notebook expects the Lumina z = 10 catalogues
(`lumina_z10.hdf5`, `luminaEDE_z10.hdf5`, about 11 GB) in `data/`. Set `build_again=True`
once to build the small cache at `cache/uv_samples_all_limits.npz`; later runs read the cache.

## Files

- `spatial_maps.ipynb`: the full analysis (outputs cleared)
- `lumina_geometry.py`: cosmology, survey geometry and M_UV conversion, copied from my
  SURF cosmic variance code
- `requirements.txt`

## Credits

Built on my SURF 2026 cosmic variance work with the Lumina simulations. The Lumina
simulations belong to their authors and are not redistributed here.
