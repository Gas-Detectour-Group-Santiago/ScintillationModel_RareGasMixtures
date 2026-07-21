# Production plot recipes

Every official production PDF is selected from one of four CSV files:

```text
config/plots/fits.csv
config/plots/primary.csv
config/plots/secondary.csv
config/plots/spectra.csv
```

The Python modules contain physical equations and reusable renderers. The CSV files contain no equations; they choose registered data/model layers and presentation. The distributed files currently define 30 fit, 31 primary, 22 secondary and 22 spectral figure IDs.

## Shared rules

- `enabled=true|false` activates or disables a row.
- `plot_id` identifies the final PDF. Rows with the same ID belong to the same figure.
- `series_id` identifies one curve/layer inside the figure.
- `model_id` is a registered fit/model, never an arbitrary Python expression.
- `component` selects a stable output exposed by that model, for example `fast`, `slow`, `total`, `696` or `vis`.
- `normalization` references `config/normalizations.csv`.
- `bands` accepts `none`, `stat`, `syst`, `stat_syst`, `ocw`, `total` or `all` when supported.
- `output` is relative to the corresponding family product root and is collected into the canonical `outputs/figures/<family>/` tree. The `.runtime` path is internal only.

## Fits

A fit row selects one diagnostic. Experimental points always use statistical error bars; the fit/toy machinery remains unchanged.

Useful columns include:

```text
fit_id, dataset_id, component, pressures_bar
grid_scale, grid_min, grid_max, grid_points
xlabel, ylabel, xscale, yscale, xmin, xmax, ymin, ymax
show_components, show_only_fit_points, output
```

Duplicating a row and changing `dataset_id`, limits or output creates another fit diagnostic without a new plotting function.

## Primary predictions

A primary row can be a standard band, low-pressure band, multibar curve or a registered special product.

```text
plot_id, series_id, group, plot_type
model_id, component, pressure_bar
normalization, bands
datasets, scale_xray_with_normalization
x/y labels, grids, limits and output
```

`datasets` is a `|`-separated list of IDs from `config/experimental_datasets.csv`. Example:

```text
arcf4_uv_xray|amedo_cf4_uv|morozov_cf4_uv|santorelli_ar_uv
```

The dataset registry owns the source file, label, marker, fill, stable colour position and reusable transform. Current transforms are:

- `static`: an ordinary `x,y,yerr` CSV;
- `primary_xray`: processed X-ray points rescaled with the selected model normalisation;
- `primary_ir_sum`: quadrature sum of the five measured IR lines.

### Components and totals

A model should expose ordinary physical totals itself:

```text
fast
slow
total = fast + slow
```

To draw singlet and triplet separately, add two rows sharing `plot_id`, one with `component=fast` and one with `component=slow`. Add a third `component=total` row to show the sum as well. No plotting code changes are needed.

## Secondary predictions

The secondary file supports four main layer types:

- `kind=model`: one model component;
- `kind=combined`: a generic sum/combination of components;
- `kind=experimental`: a dataset from `experimental_datasets.csv`;
- `kind=simulation`, `plot_type=scan`: direct catalogue quantities such as gain or alpha_eff/p.

Simulation/model rows reference `selection_id` from `config/secondary_selections.csv`. A selection defines reusable gas, pressure, gap, field, concentration, NPE, gain and normalisation masks.

A combined layer uses entries separated by `|`:

```text
ArCF4_primary:vis:sys_stat:visible_ocw|ArCF4_IR_primary:total:sum:ir_ocw
```

The generic engine evaluates each child with the same selected simulation population and combines central values and uncertainty bands according to `operation` and `uncertainty_mode`.

Generic transport scans use columns:

```text
x, y, series, facet, filters, xscale, yscale
```

The current registry reproduces gain versus field, gain versus reduced field, effective Townsend quantities, gain resolution, charge-balance plots, official UV/VIS/IR/VUV multibands, metadata curves and GEM/THGEM/experimental comparisons.

## Spectra

One active spectra row is one spectral PDF. `annotation_profile` selects a reusable scientific annotation definition for the individually configured annotated figures. Current `plot_type` values are:

```text
raw
generated
generated_extended
comparison
annotated
```

The main controls are:

```text
gas, components
pressures_bar, concentrations_percent
mosaic_rows, mosaic_cols, share_y
wavelength_min_nm, wavelength_max_nm
inset, inset_min_nm, inset_max_nm
inset_width_percent, inset_height_percent, inset_location
broken_x, log_y, title, output
```

Concentrations define panels and pressures define curves. The existing rows reproduce the historical 3×3 mosaics. Smaller or larger homogeneous mosaics only require changing panel lists and row/column counts.

An inset is deliberately simple: `inset=true`, its wavelength range, relative width/height and location. The main wavelength range remains independent. `broken_x=true` uses the inset range as the left VUV window and the main range as the right optical window. Spectral mosaics are homogeneous: concentrations define panels and pressures define curves; heterogeneous paper layouts remain intentionally outside the registry.

## Adding a new physical model

The plotting registry is the final step, not the first:

1. Map its Degrad population or Garfield hLevels population.
2. Implement the kinetic equation and expose stable component names.
3. Register parameters from a fit or literature table.
4. Add a fit only when parameters are actually fitted.
5. Register its primary/secondary adapter and normalisation/OCW contract.
6. Add a spectral shape when a wavelength-resolved spectrum is required.
7. Add rows to the appropriate plot CSV.

Examples:

- another Ar IR peak: expose `794`, add fit/prediction rows and optionally a spectral line;
- He IR family: add helium populations/model/parameters, then primary or secondary rows;
- another N2 continuum: expose `band_a|band_b|total`, then select those components in primary, secondary and spectra rows;
- Ar second continuum in Ar–CO2: reuse the additive-aware model after adding CO2 hLevels groups and quenching coefficients, then duplicate/adapt the existing Ar–CF4 or Ar–N2 rows.

## GUI

The Streamlit **Figure recipes** page edits these exact CSVs with an interactive table. Saving from the app and editing manually are equivalent. Regenerate with:

```bash
bash run_all.sh       # changed fits/data/models
bash run_products.sh  # changed plot rows or style
RECOMPUTE_BANDS=1 RECOMPUTE_TABLES=1 bash run_products.sh  # changed model/normalisation logic
```
