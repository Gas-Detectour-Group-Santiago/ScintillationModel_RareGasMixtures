# Architecture

The project separates immutable inputs, reusable numerical caches and user-facing products.

- `data/raw`: original pickles, Degrad TXT and Garfield ROOT files.
- `data/processed`: compact aggregate input tables.
- `data/reference`: literature parameters, cross sections and hLevels catalogues.
- `data/cache`: fitted parameters, compressed toys, prediction bands, spectra and the secondary simulation catalogue.
- `outputs/figures`: PDF only.
- `outputs/tables`: LaTeX only.

Validated physical models execute through a generated `.runtime` compatibility workspace. New registries control input discovery, population grouping, exact secondary parameter sets, normalisations, OCW rules and all production figures.

Independent and joint models remain distinct. Shared comparison groups are descriptive only.

## Configuration-driven production figures

Every official figure is selected by one of four files:

```text
config/plots/fits.csv
config/plots/primary.csv
config/plots/secondary.csv
config/plots/spectra.csv
```

The physical equations and data transformations remain in Python. The CSV rows select registered models, stable components, datasets, normalisations, uncertainty policies, masks and presentation. Rows with the same `plot_id` form one PDF. This keeps family-specific conventions explicit without maintaining a plotting function for every figure.

The fit registry preserves the convention that experimental points show statistical error bars. Primary and secondary recipes share normalisation and uncertainty contracts. Spectral recipes additionally expose homogeneous mosaics, wavelength windows, inset and broken-axis controls.

## Graphical control layer

`app/main.py` is a Streamlit router with four pages: pipeline execution, figure-registry editing, output browsing and global style editing. The UI depends on public modules in `src/scintillation/gui/` and edits the exact CSV files used by command-line production.

The control layer does not implement physical equations. A GUI action either updates one of the four registries/style presets or calls `run_all.sh`/`run_products.sh`.

The active style is propagated to modern and legacy-compatible plotting code through `SCINTILLATION_PROJECT_ROOT`, `SCINTILLATION_STYLE_PRESET` and `SCINTILLATION_STYLE_FILE`. The compatibility workspace inherits these variables, so validated historical physics and the generic renderers share one profile.

Primary numerical products are cached separately from figures. `run_products.sh` reuses point tables and band CSVs by default; `run_all.sh` forces their regeneration after new fits. In-process point/toy evaluation is also memoised, preventing repeated evaluation of identical point/normalisation combinations within one run.
