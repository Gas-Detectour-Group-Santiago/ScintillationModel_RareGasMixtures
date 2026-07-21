# Secondary predictions

This folder evaluates secondary scintillation predictions using the fit products
exported by `primary_fits/` and the Garfield population summaries under
`data/Secondary_GarfieldData/`.

Run first, if the fitted parameters are not already present:

```bash
python primary_fits/run_primary_fits.py
```

Then generate all secondary outputs:

```bash
python secondary_predictions/run_secondary_predictions.py
```

For a much faster regeneration of only the new UV/VUV figures:

```bash
python secondary_predictions/run_secondary_uv_vuv_predictions.py
```

Useful variants are:

```bash
python secondary_predictions/run_secondary_uv_vuv_predictions.py --uv-only
python secondary_predictions/run_secondary_uv_vuv_predictions.py --vuv-only
```

The runner automatically upgrades the saved Ar--CF4 population summaries from
the already exported per-ROOT level CSVs.  It creates the three non-overlapping
argon bins required by the second-continuum model:

```text
11.50--11.70 eV  -> Ar_1s4_1s5
11.70--12.00 eV  -> Ar_1s2_1s3
12.00--100 eV    -> Ar_dbleStar
```

No ROOT rereading or `uproot` installation is required for this upgrade.

## Paper-style outputs

The default `config_paper` generates three parallel spectral groups:

- `400--800 nm`: visible + fitted IR lines.
- `200--400 nm`: total fitted UV channel.
- `100--200 nm`: Ar second continuum and the
  `CF4+*(D) -> CF4+(X)` branch shown separately.

Each group contains:

1. GEM yield versus CF4 concentration at three pressures.
2. TH-GEM yield versus CF4 concentration at three pressures.
3. Yield versus electric field at 1% CF4 and `Ne = 100 +- 20`.

The new plot folders are:

```text
secondary_predictions/plots/secondary_uv/
secondary_predictions/plots/secondary_vuv/
```

Every VUV figure contains only the argon second-continuum prediction at
128 nm, integrated over `100--200 nm`. The VUV output consists of exactly three
PDFs: GEM versus concentration, TH-GEM versus concentration, and the electric-
field scan.

## VUV treatment

The Ar branch is evaluated with `models/Ar2nd_continium.py`, including the fast
and slow excimer contributions according to `triplet_weight` and their dynamic
quenching by CF4. The 128 nm Gaussian is then integrated over `100--200 nm`.
This branch is an absolute kinetic prediction and is not divided by the fitted
Ar--CF4 normalization.

The concentration plots extend from `0.001% CF4` (`99.999% Ar`) to pure CF4.
The lowest available Garfield simulation is `0.1% CF4`. Below that point the
normalised Garfield excitation populations are held fixed at their `0.1%`
values, while the actual CF4 fraction continues to enter every kinetic
competition term. This isolates the low-CF4 extrapolation to the quenching
model and avoids extrapolating noisy avalanche populations. No numerical
pure-argon switch is used, so the prediction is continuous as CF4 tends to
zero.

The concentration figures use a logarithmic y axis because the Ar2nd yield
spans several orders of magnitude. Monteiro and the CF4 ionic VUV branch are
not included in these figures, and no additional VUV tables are exported.

## Normalization

For paper concentration scans, selected Garfield populations are divided by
`ne` before interpolation and the final prediction is divided by `NPE`.

For fitted UV/VIS/IR branches:

```text
raw_model * X_RAY_ENERGY / (Nnorm * NPE * ne)
```

For the absolute Ar second-continuum branch:

```text
raw_Ar2nd * X_RAY_ENERGY / (NPE * ne)
```

The electric-field scans use the equivalent post-evaluation division by `ne`.
