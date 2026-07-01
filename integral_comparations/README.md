# integral_comparations

Configurable comparisons of integrated experimental spectra.

## Main script

```bash
python3 integral_comparations/run_integral_comparisons.py
```

## Structure

```text
integral_comparations/
  run_integral_comparisons.py
  aux/
    paths.py
    units.py
    spectra_io.py
    integrators.py
    ratios.py
    plotting.py
  csv/
  plots/
  gaussian_fit/
```

## What it supports

- independent numerator and denominator definitions;
- independent spectrum-column choice for numerator and denominator;
- independent integration methods for numerator and denominator;
- hard-cut integrals with:
  - `trapz`
  - `simpson`
  - `quad`
- Gaussian-fit integrals with:
  - fixed centres;
  - bounded centres;
  - free centres;
  - shared global wavelength shift;
  - shared or individual sigmas;
  - baseline constante por defecto (`GAUSSIAN_BASELINE_MODE = "constant"`); la baseline no se integra en el área de los picos.

## Gaussian diagnostics

If either numerator or denominator uses `gaussian_fit`, the script writes
extra diagnostic figures under:

```text
integral_comparations/gaussian_fit/
```

By default the diagnostics are mosaics: one PDF per gaussian-fit case and
selected pressure. The top-level options in `run_integral_comparisons.py` let
you choose:

- the diagnostic mode: `"mosaic"`, `"individual"` or `"both"`;
- the pressures to draw, default `(1.0,)`;
- the concentrations to place in the mosaic;
- the number of panels through `GAUSSIAN_MOSAIC_NROWS` and
  `GAUSSIAN_MOSAIC_NCOLS`.

Each panel contains the spectrum, the fitted window, the total fitted curve
and each Gaussian component overlaid.  The fitted offset/background is kept
inside the numerical model when requested, but it is hidden in the plots by
default.

## Peak library

The main script includes selectable Ar--N2 UV profiles. The default is
`GAUSSIAN_PEAK_PROFILE = "ArN2_UV_5"`, i.e. five fixed approximate centres:

```text
336, 356, 380, 410, 430 nm
```

Other built-in profiles are:

- `ArN2_UV_4`: `336, 356, 380, 410 nm`;
- `ArN2_UV_6`: `336, 356, 380, 391.4, 410, 430 nm`;
- `ArN2_UV_7`: `336, 356, 380, 391.4, 410, 424.2, 430 nm`;
- `ArN2_UV`: backward-compatible alias of `ArN2_UV_5`.

The 410 and 430 nm components are deliberately placed after the main
336--380 nm structure and are fixed by default through:

```python
GAUSSIAN_CENTER_MODE = "fixed"
```

Change it to `"bounded"` or `"shared_shift"` if you want to allow small
wavelength-calibration shifts while still starting from the exact nominal
centres.

You can edit or extend the profiles directly in `run_integral_comparisons.py`.


## Six-case ArN2 comparison

`run_integral_comparisons.py` now builds six class-based configurations:

```text
mean + hardcut
mean + gaussian_fit
C1   + hardcut
C1   + gaussian_fit
C2   + hardcut
C2   + gaussian_fit
```

The denominator is fixed for all cases:

```text
ArCF4 95/5, 1 bar, 500-750 nm, exported mean_spectrum raw CSV, hardcut/trapz
```

It writes:

```text
integral_comparations/csv/ArN2_over_ArCF4_95_5_1bar_VIS_six_cases_ratio_scan.csv
integral_comparations/csv/ArN2_over_ArCF4_95_5_1bar_VIS_six_cases_1bar_100pct.csv
integral_comparations/plots/ArN2_over_ArCF4_95_5_1bar_VIS_six_cases_ratio_grid.pdf
```

Gaussian diagnostics are written under:

```text
integral_comparations/gaussian_fit/
```

## Spectrum input

`integral_comparations` reads the long CSV spectra exported by
`data/Analysis_spectra.py`; it does not need to reopen the experimental
pickles for the default comparison.  The preferred directory is:

```text
data/spectra/
```

For compatibility with existing exports, the loader also accepts the previous
capitalized directory:

```text
data/Spectra/
```

Expected files are `ArCF4_raw_spectra.csv` and `ArN2_raw_spectra.csv`; if those
are absent, `raw_spectra.csv` is used as a fallback.


## Absolute table from primary predictions

The module can also convert the six Ar--N2 integral cases into absolute Ar--N2
predictions.  By default it no longer uses Ar--CF4 95/5 as the physical
normalisation of the table.  Ar--CF4 remains only as a common internal
denominator for the scan plot; the absolute table is anchored to the pure-N2
primary prediction:

```python
PRIMARY_REFERENCE_ID = "N2_UV_N2"
PRIMARY_REFERENCE_SCALE_MODE = "relative_to_anchor"
PRIMARY_REFERENCE_ANCHOR_RATIO_NAME = "ArN2_mean_hardcut_over_ArCF4_95_5_VIS"
```

Therefore the table uses:

```text
absolute_i = primary(N2_UV_N2) * ratio_i / ratio(mean_spectrum, hardcut)
```

The anchor row `mean + hardcut` is exactly the value of `N2_UV_N2` read from:

```text
data/Predictions/primary_selected_yields_arcf4_vs_arn2_norm.csv
```

With the default configuration, the six rows correspond to Ar--N2 at 1 bar and
100% N2:

```python
SUMMARY_PRESSURE_BAR = 1.0
SUMMARY_CONCENTRATION_PERCENT = 100.0
```

The output is written to:

```text
integral_comparations/csv/ArN2_six_cases_scaled_to_N2_mean_hardcut_primary_norms.csv
integral_comparations/tables/ArN2_six_cases_scaled_to_N2_mean_hardcut_primary_norms.tex
```

The LaTeX table has six rows:

```text
mean + hardcut
mean + gaussian_fit
C1   + hardcut
C1   + gaussian_fit
C2   + hardcut
C2   + gaussian_fit
```

and two numerical columns:

```text
Norm. Ar--CF4
Norm. Ar--N2
```

The quoted uncertainty is the total uncertainty of the selected primary anchor
scaled by the relative integral factor.  The integral ratios themselves are
central scale factors because the spectrum-integration step does not yet assign
an independent statistical uncertainty.

For future IR or alternative comparisons, keep the same mechanism but change
`PRIMARY_REFERENCE_ID` and `PRIMARY_REFERENCE_ANCHOR_RATIO_NAME` to the row and
integral case that should define the absolute scale.
