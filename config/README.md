# Configuration

The production plot registry is intentionally limited to four CSV files:

```text
config/plots/
├── fits.csv
├── primary.csv
├── secondary.csv
└── spectra.csv
```

They are the canonical source for every production PDF. Python contains the physics, uncertainty propagation and reusable renderers; the CSV rows decide which datasets/components are drawn, the masks, normalisation, bands, axes, layout and output name.

## Conventions

- `enabled=false` disables a row without deleting it.
- Rows with the same `plot_id` belong to the same PDF.
- In primary and secondary files, one row is one curve, component, experimental dataset or scan layer.
- `components=A|B|C` requests a registered sum/combination. Normal physical totals should preferably be exposed by the model as `total`.
- `normalization` references `normalizations.csv`.
- `bands` uses `none`, `stat`, `syst`, `stat_syst`, `ocw`, `total` or `all`, where supported.
- Experimental data are referenced by `dataset_id` from `experimental_datasets.csv`; labels, markers and stable colours are defined once there.
- Secondary simulation masks are referenced by `selection_id` from `secondary_selections.csv`.

## Fits

`plots/fits.csv` controls fit diagnostics and toy-correlation PDFs. Experimental fit points always show statistical error bars. A row selects the fit, dataset/component, pressures, grid, labels, scales and output.

## Primary

`plots/primary.csv` controls standard prediction bands, low-pressure bands, multibar overlays and the electron/X-ray products. The `datasets` column is a `|`-separated list of entries from `experimental_datasets.csv`. `scale_xray_with_normalization` decides whether X-ray data follow the selected prediction normalisation.

## Secondary

`plots/secondary.csv` contains integrated-yield figures, combined components, experimental comparisons, metadata curves and generic simulation scans such as gain versus E, gain versus E/p or alpha_eff/p versus E/p. `kind` distinguishes `model`, `combined`, `experimental` and `simulation` layers.

## Spectra

`plots/spectra.csv` controls raw, generated, extended, comparison and annotated spectral products. Concentrations are mosaic panels and pressures are curves. The same row can select:

- `mosaic_rows` and `mosaic_cols`;
- shared or independent y limits;
- main wavelength range;
- optional VUV inset and its range/size/location;
- optional broken-x view;
- linear or logarithmic y scale.

## Other registries

- `experimental_datasets.csv`: one stable name, file, label, marker and colour per dataset.
- `secondary_selections.csv`: reusable pressure/gap/field/gain/population masks.
- `channels.csv`: named wavelength channels and model families.
- `normalizations.csv`: reusable absolute/reference normalisations.
- `uncertainty_policies.csv`: shared uncertainty contracts.
- `ocw.csv`: optical collection window transformations.
- `population_groups.csv`: Garfield hLevels to canonical populations.
- `primary_population_groups.csv`: Degrad channels to primary populations.
- `secondary_parameter_sets.csv`: exact secondary model + fit/literature + normalisation + OCW recipes.
- `spectral_components.csv`: registered wavelength components.
- `styles/`: global reusable plot styles.
