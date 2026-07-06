# ScintillationModel Rare Gas Mixtures

Phenomenological scintillation-modeling framework for rare-gas mixtures, with emphasis on primary and secondary light production in Ar--CF$_4$, Ar--N$_2$ and related detector gases. The project combines microscopic population inputs from Degrad/Garfield++ with kinetic emission models, fit machinery, uncertainty propagation, spectral synthesis and paper-ready plots/tables for a TFM-style analysis.

<p align="center">
  <img src="docs/assets/readme/spectra_arcf4_extended.png" width="78%" alt="Extended generated Ar-CF4 spectra from 100 to 800 nm">
</p>

## What this repository does

The code is organized as a reproducible analysis pipeline:

1. **Read microscopic inputs** from Degrad and Garfield++ summaries.
2. **Fit primary kinetic models** to experimental X-ray scintillation yields.
3. **Export fitted parameters**, toy distributions, covariance and correlation matrices.
4. **Predict primary yields** in ph/MeV, including bands from statistical/systematic toys.
5. **Predict secondary scintillation** in ph/e$^-$ from Garfield++ avalanche populations.
6. **Generate spectra** from experimental raw spectra and kinetic-model predictions.
7. **Export publication-ready plots, CSVs and LaTeX tables** used in the thesis.

```mermaid
flowchart LR
    A[Degrad primary populations] --> B[Primary kinetic models]
    C[Experimental X-ray yields] --> D[Primary fits]
    B --> D
    D --> E[Parameters + toys + covariance]
    E --> F[Primary predictions ph/MeV]
    G[Garfield++ avalanche populations] --> H[Secondary predictions ph/e-]
    E --> H
    F --> I[Generated spectra]
    H --> I
    J[Raw experimental spectra] --> I
    I --> K[Figures, CSVs and LaTeX tables]
```

## Representative outputs

| Step | Example output |
|---|---|
| Cross sections | <img src="docs/assets/readme/cross_sections_ar.png" width="280" alt="Ar cross sections"> |
| Primary fit | <img src="docs/assets/readme/primary_fit_arcf4_uv.png" width="280" alt="Ar-CF4 UV primary fit"> |
| Primary prediction band | <img src="docs/assets/readme/primary_prediction_arn2_uv.png" width="280" alt="Ar-N2 primary UV prediction"> |
| Secondary comparison | <img src="docs/assets/readme/secondary_arcf4_visible.png" width="280" alt="Secondary Ar-CF4 comparison"> |
| Gaussian extraction study | <img src="docs/assets/readme/gaussian_ratio_grid.png" width="280" alt="Ar-N2 gaussian ratio study"> |
| Garfield++ channel catalogue | <img src="docs/assets/readme/ar_population_catalogue.png" width="280" alt="Ar Garfield++ level catalogue"> |

The corresponding PDFs remain in their native folders, for example `primary_predictions/plots/`, `secondary_predictions/plots/`, `spectra/plots/`, `integral_comparations/plots/` and `populations_histograms/pdf/`.

## Repository layout

```text
.
├── cross_sections/              # Magboltz/cross-section inspection and plots
├── data/                        # Experimental data, Degrad/Garfield inputs, parameters, tables
│   ├── Experimental/            # Experimental yields and spectra
│   ├── Primary_DegradData/      # Primary population tables from Degrad
│   ├── Secondary_GarfieldData/  # Secondary/avalanche population tables from Garfield++
│   ├── FitResults/              # Fitted parameters, toys, covariance/correlation outputs
│   ├── Parameters/              # Central parameter vectors consumed by predictions/spectra
│   ├── Predictions/             # CSV prediction outputs
│   └── Tables/                  # LaTeX tables exported by the pipeline
├── integral_comparations/       # Ar--N2 / Ar--CF4 integral-ratio and Gaussian extraction studies
├── models/                      # Kinetic models: ArCF4, ArN2, IR and Ar second continuum
├── populations_histograms/      # Garfield++ channel/level catalogue diagnostics
├── primary_fits/                # Primary fits, toy studies and parameter exports
├── primary_predictions/         # Primary yield predictions, bands, overlays and tables
├── secondary_predictions/       # Secondary scintillation predictions and experimental comparisons
├── spectra/                     # Raw/generated/comparison/annotated spectra pipeline
├── plot_style.py                # Shared matplotlib style helpers
└── run_all.sh                   # Project-level runner
```

## Installation

Use a Python virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas matplotlib scienceplots dill uproot awkward
```

Optional system tools:

- `pdftoppm` for generating README thumbnails from PDFs.
- CMake and a C++ compiler for `cross_sections/` if rebuilding the cross-section executable.

## Quick start

Run the complete default workflow:

```bash
bash run_all.sh
```

Or run each stage explicitly:

```bash
python primary_fits/run_primary_fits.py
python primary_predictions/run_primary_predictions.py
python secondary_predictions/run_secondary_predictions.py
python spectra/run_all_spectra.py
python integral_comparations/run_integral_comparisons.py
python populations_histograms/run_population_histograms.py
```

Most modules can also be run independently while editing a specific figure or table.

## Main configuration points

The project is deliberately config-driven. In normal use you should edit the configuration files, not the plotting or runner internals.

| Task | Main file |
|---|---|
| Change primary fit parameters, bounds, datasets or toy settings | `primary_fits/ArCF4_fit.py`, `primary_fits/ArN2_fit.py`, `primary_fits/ArCF4_IR_fit.py`, `primary_fits/ArN2_IR_fit.py` |
| Change primary prediction bands, normalizations or selected yields | `primary_predictions/configs.py` |
| Change secondary prediction masks, gain selections, OCW bands or comparisons | `secondary_predictions/configs.py`, `secondary_predictions/config_comparation.py`, `secondary_predictions/config_paper.py` |
| Change raw/generated spectra settings | `spectra/config.py` |
| Change Gaussian/hardcut integral comparisons | `integral_comparations/run_integral_comparisons.py` |
| Change Ar second-continuum parameters | `data/Parameters/Ar2nd_continium.csv` and `models/Ar2nd_continium.py` |

## Primary model workflow

Primary fits are defined by `FitConfig` objects. A typical fit contains:

- a model name,
- a Degrad population CSV,
- one or more experimental datasets,
- kinetic equations,
- fitted parameters with bounds/fixed flags,
- plot definitions,
- toy settings for statistical and systematic variations.

Example entry point:

```bash
python primary_fits/ArCF4_fit.py
```

The fit exports:

```text
data/FitResults/<fit_name>_central.csv
data/FitResults/<fit_name>_toys_stat.csv
data/FitResults/<fit_name>_toys_syst.csv
data/FitResults/<fit_name>_covariance.csv
data/FitResults/<fit_name>_correlation.csv
data/Parameters/<fit_name>.csv
data/Tables/<fit_name>_param_stat_syst.tex
primary_fits/plots/plot_fit/*.pdf
```

The fitted model output is converted to ph/MeV in the prediction layer through `NormalizationConfig`. The most common normalization modes are:

| Mode | Meaning |
|---|---|
| `own_norm` | Divide by the fitted normalization of the same model. |
| `reference_norm` | Divide by another fit's normalization, for example the Ar--CF$_4$ primary normalization. |
| `as_fit` | Keep the fitted normalization in the model output. |
| `fixed_norm` | Use an explicit numeric normalization. |

## Secondary model workflow

Secondary predictions use Garfield++ avalanche populations, then evaluate the same kinetic models with a secondary scaling. The selection is controlled by `SecondarySelection` objects, for example:

```python
SecondarySelection(
    id="gem_1bar_gain100",
    gas="ArCF4",
    pressure=1.0,
    gap_mm=0.05,
    gain_min=80,
    gain_max=120,
    normalize_by="pre_ne",
)
```

Selections can be made on pressure, gap, electric field, concentration, `npe`, `ne`, `ni`, gain or arbitrary CSV columns via `masks` / `extra_masks`. This keeps detector-specific logic out of the plotting code.

Secondary outputs are written to:

```text
data/Predictions/Secondary/
data/Tables/secondary_*.tex
secondary_predictions/plots/secondary_bands/
secondary_predictions/plots/secondary_comparation/
```

## Spectra workflow

The spectra module produces four families of plots:

1. raw experimental mosaics,
2. generated kinetic spectra,
3. raw-vs-generated comparisons,
4. annotated single spectra.

Run everything with:

```bash
python spectra/run_all_spectra.py
```

Useful switches:

```bash
python spectra/run_all_spectra.py --no-raw
python spectra/run_all_spectra.py --no-generated
python spectra/run_all_spectra.py --no-comparison
python spectra/run_all_spectra.py --no-annotated
```

The most important controls live in `spectra/config.py`:

```python
RAW_PLOT_SPECTRUM_COLUMN = "mean_spectrum"
GENERATED_PRESSURES_BAR = (1, 2, 3, 4, 5, 10)
GENERATED_CONCENTRATIONS_PERCENT = (0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0)
GENERATED_AMPLIED_INSET_ENABLED = True
```

## How to add a new gas mixture

A new mixture can be added without rewriting the whole project. The required work depends on whether it is primary-only, secondary-only or both.

### 1. Add microscopic input data

Primary Degrad input:

```text
data/Primary_DegradData/<Mixture>.csv
```

The table must contain a concentration column and one column per microscopic channel used by the model, for example neutral excitations, ionic excitations, molecular fragments or continuum precursors.

Secondary Garfield++ input:

```text
data/Secondary_GarfieldData/<Mixture>/populations/<Mixture>_secondary.csv
```

The secondary table should contain metadata columns such as pressure, concentration, electric field, gap, `npe`, `ne`, `ni`, plus population columns.

### 2. Implement a kinetic model

Create a new file in `models/`, for example:

```text
models/ArXe.py
```

A minimal model function should accept:

```python
def theory_yield_uv(params, degrad_data, concentration, pressure):
    # interpolate microscopic populations
    # evaluate transfer/quenching/emission probabilities
    # return yield in the same convention used by the fit layer
    return yield_array
```

For consistency with the existing models, keep these conventions:

- use concentration as a fraction internally, unless clearly documented otherwise;
- interpolate population columns with monotonic concentration grids;
- keep the first parameter as `Nnorm` if the model is fitted to arbitrary-unit data;
- divide by the X-ray energy inside the model only if the companion models do the same;
- keep component functions small and testable.

### 3. Add a primary fit config

Create a file such as:

```text
primary_fits/ArXe_fit.py
```

Define:

```python
PARAMETERS = [
    Parameter("Nnorm", r"$N_{\\mathrm{norm}}$", 1e-3, 0.0, 1.0),
    Parameter("P_channel", r"$\\mathcal{W}_{channel}$", 0.1, 0.0, 1.0),
]

DATASETS = [
    DatasetSpec(
        key="uv",
        csv_path=DATA_DIR / "Experimental" / "ArXe" / "csv" / "uv.csv",
        x_col="fXe",
        pressures=[1, 2, 3, 4, 5],
        output_concentration_name="fXe",
        w_function=W_ArXe,
    ),
]

CONFIG = FitConfig(
    name="ArXe_primary",
    model_name="ArXe",
    degrad_csv=DATA_DIR / "Primary_DegradData" / "ArXe.csv",
    datasets=DATASETS,
    equations={"uv": theory_yield_uv},
    parameters=PARAMETERS,
    plots=PLOTS,
    toy_spec=ToySpec(...),
)
```

Then register it in `primary_fits/run_primary_fits.py`.

### 4. Register the prediction adapter

Add the new fit and components to `primary_predictions/configs.py`:

```python
from primary_fits.ArXe_fit import CONFIG as ARXE_CONFIG

PRIMARY_ADAPTERS["ArXe_primary"] = PrimaryModelAdapter(
    fit_name="ArXe_primary",
    degrad_csv=ARXE_CONFIG.degrad_csv,
    components={
        "uv": ARXE_CONFIG.equations["uv"],
        "total": ARXE_CONFIG.equations["uv"],
    },
)
```

Then add `BandPlotConfig`, `PredictionPoint` or `MultiBandPlotConfig` entries depending on the plots/tables you want.

### 5. Register spectra, if needed

For generated spectra, add the wavelength-shape logic in `spectra/auxiliares/generated.py` and expose the mixture in `spectra/config.py`. Raw spectra require either a pickle/CSV reader candidate in `spectra/auxiliares/io.py` or a compatible long CSV in `data/Spectra/`.

### 6. Add secondary predictions, if needed

Add a `SecondaryPlotConfig` / `MultiBandPlotConfig` in `secondary_predictions/configs.py` or `secondary_predictions/config_comparation.py`, pointing to the correct Garfield++ table with `SecondarySelection(population_filename=...)`. If the microscopic population names differ, add or adapt the component mapping in the relevant secondary adapter/config.

## How to use your own parameters

There are three supported levels.

### Option A: edit a central parameter CSV

For quick tests, edit:

```text
data/Parameters/<fit_name>.csv
```

Then rerun only predictions:

```bash
python primary_predictions/run_primary_predictions.py
python spectra/run_all_spectra.py --no-raw --no-comparison
```

### Option B: change the fit configuration

For a reproducible change, edit the `PARAMETERS` list in the relevant `primary_fits/*_fit.py` file. You can change initial values, bounds, fixed flags or labels:

```python
Parameter("K_transfer", r"$K_{transfer}$", value=0.1, min_value=0.0, max_value=10.0)
```

Then rerun fits and predictions:

```bash
python primary_fits/run_primary_fits.py
python primary_predictions/run_primary_predictions.py
```

### Option C: define a new prediction-only component

For a phenomenological extension that should not be fitted, add a component in `primary_predictions/configs.py`, as done for the Ar second continuum and the CF$_4^{+,*}(D\rightarrow X)$ VUV branch. This is useful for testing spectral components with fixed branching ratios or external absolute references.

## Output conventions

| Quantity | Typical unit | Location |
|---|---|---|
| Primary yields | ph/MeV | `data/Predictions/`, `primary_predictions/plots/` |
| Secondary yields | ph/e$^-$ | `data/Predictions/Secondary/`, `secondary_predictions/plots/` |
| Raw spectra | arbitrary units | `spectra/csv/`, `spectra/plots/` |
| Generated spectra | ph MeV$^{-1}$ nm$^{-1}$ | `spectra/csv/`, `spectra/plots/` |
| Fit parameters | dimensionless / ns$^{-1}$ / model-specific | `data/FitResults/`, `data/Parameters/` |
| LaTeX tables | `.tex` | `data/Tables/`, module-specific `tables/` folders |

## Recommended development workflow

Use the runners in this order when changing model physics:

```bash
python primary_fits/run_primary_fits.py
python primary_predictions/run_primary_predictions.py
python secondary_predictions/run_secondary_predictions.py
python spectra/run_all_spectra.py
```

Use the smaller module runners when changing only style or plotting:

```bash
python spectra/run_raw_spectra.py
python spectra/run_generated_spectra.py
python primary_predictions/run_primary_multiband_predictions.py
python secondary_predictions/run_secondary_predictions.py
```

Before committing, remove generated caches that are not meant to be versioned:

```bash
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete
```

## Code organization notes

The current code is functional and already mostly modular, but the following cleanups would make it easier to maintain:

- Rename `auxiliares/` to `utils/` or `core/` once the thesis stabilizes.
- Move all generated files to a single top-level `outputs/` directory, while keeping `data/` for inputs and curated parameter tables.
- Remove `__pycache__/`, local build directories and notebook scratch files from version control.
- Standardize naming: `fiting.py` -> `fitting.py`, `ploting.py` -> `plotting.py`, `continium` -> `continuum`, `amplied` -> `amplified`.
- Split `models/` into small, documented components: interpolation, kinetic probabilities, spectral shapes and unit conversion.
- Add a small test suite for W-values, Gaussian integrals, normalization modes and selected prediction points.
- Keep all hard-coded physics constants in CSV/YAML/TOML config files, not inside plotting scripts.

These changes are not required to reproduce the present results, but they would make future mixtures and detector geometries much easier to add.

## Minimal reproducibility checklist

When publishing or sharing a result, keep together:

1. the model file in `models/`,
2. the fit configuration in `primary_fits/`,
3. the central parameter CSV in `data/Parameters/`,
4. the Degrad/Garfield++ population CSVs used,
5. the generated CSV behind the plot,
6. the plotting configuration that generated the PDF.

This ensures that each figure in the thesis can be traced back to a model equation, a parameter set and a microscopic population input.
