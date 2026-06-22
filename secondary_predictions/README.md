# Secondary predictions

This folder evaluates secondary scintillation predictions using the fit products
already exported by `primary_fits/`, but the population input is now the Garfield
summary CSV, not the primary Degrad CSV stored in the fit configuration.

Run first, if the parameters are not already present:

```bash
python primary_fits/run_primary_fits.py
```

Then run the secondary prediction test plot:

```bash
python secondary_predictions/run_secondary_predictions.py
```

Current default configuration
-----------------------------

`configs.py` defines a single test configuration:

- mixture: `ArCF4`
- curves: visible + total IR
- Garfield CSV: `data/Secondary_GarfieldData/ArCF4/populations/ArCF4_secondary.csv`
- masks: `gap_mm = 0.050` and `gain = 100 +- 20`
- because the current CSV has no explicit `gain` column, the runner creates `gain = ne`
- normalization column: `normalise_by="ne"`; it can be changed to `"ni"`

Outputs
-------

- `data/Predictions/Secondary/ArCF4_visible_ir_gap005_gain100pm20_selected_garfield_rows.csv`
- `data/Predictions/Secondary/ArCF4_visible_ir_gap005_gain100pm20_visible.csv`
- `data/Predictions/Secondary/ArCF4_visible_ir_gap005_gain100pm20_infrared_total.csv`
- `secondary_predictions/plots/secondary_predictions/ArCF4_visible_ir_gap005_gain100pm20.pdf`

Mask syntax
-----------

Masks are ordinary dictionaries over Garfield CSV columns. Supported forms:

```python
masks = {
    "gap_mm": {"value": 0.050, "atol": 5e-4},
    "gain": {"min": 80.0, "max": 120.0},
    "electric_field": {"min": 40.0, "max": 70.0},
    "pressure": [0.2, 1.0],
    "npe": 100,
}
```

Useful aliases are also accepted: `E` -> `electric_field`, `gap` -> `gap_mm`,
`pressure_bar` -> `pressure`, `Ne` -> `ne`, `Ni` -> `ni`.

Normalization
-------------

Before the model is evaluated, every population column is divided by
`npe * normalise_by`, where `normalise_by` is normally `ne` or `ni`.
The model output, which internally contains the primary X-ray-energy scaling,
is converted to secondary yield with

```text
raw_model * X_RAY_ENERGY / NORM
```

so the effective scale is

```text
population * X_RAY_ENERGY / (NORM * NPE * Ne)
```

or the same expression with `Ni` if `normalise_by="ni"`.

For each curve, `norm_fit_name` selects the fit whose first parameter is used as
`NORM`. This keeps IR safe: the IR fit does not have `Nnorm` as its first
parameter, so the current ArCF4 visible+IR test uses `norm_fit_name="ArCF4_primary"`
for both curves.

Future mixtures
---------------

The runner is mixture-generic. For a new mixture such as `ArN2` or `HeCF4`, add a
`SecondaryPlotConfig` with `mixture="ArN2"` or `mixture="HeCF4"`. The runner will
look under:

```text
data/Secondary_GarfieldData/<mixture>/populations/<mixture>_secondary.csv
```

If the population names differ, only the curve components/adapters in `configs.py`
need to be adjusted.
