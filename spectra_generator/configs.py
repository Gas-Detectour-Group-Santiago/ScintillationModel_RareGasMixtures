from __future__ import annotations

from .bootstrap import bootstrap_project
from .auxiliares.spectra_types import (
    AnnotatedScriptConfig,
    ComparisonCurveConfig,
    ComparisonMosaicConfig,
    GeneratedSpectraConfig,
    RawReferenceConfig,
    RawSpectraConfig,
)


PROJECT_ROOT = bootstrap_project(__file__)
DATA_DIR = PROJECT_ROOT / "data"
SPECTRA_DIR = PROJECT_ROOT / "spectra_generator"


RAW_DIR = SPECTRA_DIR / "spectra_raw"
GENERATED_DIR = SPECTRA_DIR / "spectra_generated"
COMPARATION_DIR = SPECTRA_DIR / "spectra_comparation"
ANNOTATED_DIR = SPECTRA_DIR / "spectra_annotated"
ANNOTATED_PLOTS_DIR = ANNOTATED_DIR / "plots"


GENERATED_PRESSURES_BAR = (1, 2, 3, 4, 5, 10)
GENERATED_CONCENTRATIONS_PERCENT = (0.1, 1, 10, 100)

RAW_PRESSURES_BAR = (1, 2, 3, 4, 5)
RAW_GRID_CONCENTRATIONS_PERCENT = (0, 0.1, 0.5, 1, 5, 10, 20, 50, 100)
COMPARISON_CONCENTRATIONS_PERCENT = (0.1, 1, 5, 100)

# Por defecto se usa solo mean_spectrum. Cambiar estas tuplas permite probar
# C1/C2 sin tocar el runner ni el plotter.
RAW_ARCF4_SPECTRUM_COLUMNS = ("mean_spectrum",)
RAW_ARN2_SPECTRUM_COLUMNS = ("mean_spectrum",)


BLUE = "tab:blue"
RED = "tab:red"
GREEN = "tab:green"
ORANGE = "tab:orange"
PINK_9505 = "#ff66e8"


def raw_spectra_configs() -> tuple[RawSpectraConfig, ...]:
    arcf4_csv = DATA_DIR / "Spectra" / "ArCF4_raw_spectra.csv"
    arn2_csv = DATA_DIR / "Spectra" / "ArN2_raw_spectra.csv"
    return (
        RawSpectraConfig(
            name="ArCF4_raw_mean_spectrum_grid",
            gas_mixture="ArCF4",
            input_csv=arcf4_csv,
            output_csv=RAW_DIR / "csv" / "ArCF4_raw_mean_spectrum_grid.csv",
            output_pdf=RAW_DIR / "plots" / "experimental_ArCF4_grid.pdf",
            concentrations_percent=RAW_GRID_CONCENTRATIONS_PERCENT,
            pressures_bar=RAW_PRESSURES_BAR,
            spectrum_columns=RAW_ARCF4_SPECTRUM_COLUMNS,
            wavelength_range_nm=(200.0, 800.0),
            title=None,
            mosaic_shape=(3, 3),
            figsize=(12.0, 8.0),
            share_y=True,
            common_ylim=False,
            show_percent_in_titles=False,
        ),
        RawSpectraConfig(
            name="ArN2_raw_mean_spectrum_grid_with_ArCF4_9505_reference",
            gas_mixture="ArN2",
            input_csv=arn2_csv,
            output_csv=RAW_DIR / "csv" / "ArN2_raw_mean_spectrum_grid.csv",
            output_pdf=RAW_DIR / "plots" / "experimental_ArN2_grid_with_fixed_ArCF4_95_5_reference.pdf",
            concentrations_percent=RAW_GRID_CONCENTRATIONS_PERCENT,
            pressures_bar=RAW_PRESSURES_BAR,
            spectrum_columns=RAW_ARN2_SPECTRUM_COLUMNS,
            wavelength_range_nm=(200.0, 800.0),
            title=None,
            reference=RawReferenceConfig(
                raw_csv=arcf4_csv,
                gas_mixture="ArCF4",
                concentration_percent=5.0,
                pressure_bar=1.0,
                spectrum_columns=RAW_ARCF4_SPECTRUM_COLUMNS,
                label=r"Ar/CF$_4$ 95/5",
                color=PINK_9505,
                alpha=0.42,
                fill=True,
                linewidth=1.20,
            ),
            mosaic_shape=(3, 3),
            figsize=(12.0, 8.0),
            share_y=True,
            common_ylim=False,
            show_percent_in_titles=False,
        ),
    )


def generated_spectra_configs() -> tuple[GeneratedSpectraConfig, ...]:
    return (
        GeneratedSpectraConfig(
            name="ArCF4_generated",
            gas_mixture="ArCF4",
            degrad_csv=DATA_DIR / "Primary_DegradData" / "ArCF4.csv",
            degrad_ir_csv=DATA_DIR / "Primary_DegradData" / "ArCF4_IR.csv",
            parameter_csv=DATA_DIR / "Parameters" / "ArCF4_primary.csv",
            ir_parameter_csv=DATA_DIR / "Parameters" / "ArCF4_IR_primary.csv",
            norm_parameter_csv=DATA_DIR / "Parameters" / "ArCF4_primary.csv",
            output_csv=GENERATED_DIR / "csv" / "ArCF4_generated_spectra.csv",
            output_pdf=GENERATED_DIR / "plots" / "ArCF4_generated_spectra.pdf",
            output_summary_pdf=GENERATED_DIR / "plots" / "ArCF4_generated_summary.pdf",
            pressures_bar=GENERATED_PRESSURES_BAR,
            concentrations_percent=GENERATED_CONCENTRATIONS_PERCENT,
            wavelength_min_nm=200.0,
            wavelength_max_nm=800.0,
            wavelength_points=2000,
            wavelength_range_nm=(200.0, 800.0),
            title=r"Primary Ar--CF$_4$ spectra",
        ),
        GeneratedSpectraConfig(
            name="ArN2_generated",
            gas_mixture="ArN2",
            degrad_csv=DATA_DIR / "Primary_DegradData" / "ArN2.csv",
            degrad_ir_csv=DATA_DIR / "Primary_DegradData" / "ArN2_IR.csv",
            parameter_csv=DATA_DIR / "Parameters" / "ArN2_primary.csv",
            ir_parameter_csv=DATA_DIR / "Parameters" / "ArN2_IR_primary.csv",
            norm_parameter_csv=DATA_DIR / "Parameters" / "ArCF4_primary.csv",
            output_csv=GENERATED_DIR / "csv" / "ArN2_generated_spectra.csv",
            output_pdf=GENERATED_DIR / "plots" / "ArN2_generated_spectra.pdf",
            output_summary_pdf=GENERATED_DIR / "plots" / "ArN2_generated_summary.pdf",
            pressures_bar=GENERATED_PRESSURES_BAR,
            concentrations_percent=GENERATED_CONCENTRATIONS_PERCENT,
            wavelength_min_nm=300.0,
            wavelength_max_nm=800.0,
            wavelength_points=2000,
            wavelength_range_nm=(300.0, 800.0),
            title=r"Primary Ar--N$_2$ spectra",
        ),
    )


def comparison_spectra_configs() -> tuple[ComparisonMosaicConfig, ...]:
    arcf4_raw = DATA_DIR / "Spectra" / "ArCF4_raw_spectra.csv"
    arn2_raw = DATA_DIR / "Spectra" / "ArN2_raw_spectra.csv"
    arcf4_generated = GENERATED_DIR / "csv" / "ArCF4_generated_spectra.csv"
    arn2_generated = GENERATED_DIR / "csv" / "ArN2_generated_spectra.csv"
    arcf4_norm = DATA_DIR / "Parameters" / "ArCF4_primary.csv"

    return (
        ComparisonMosaicConfig(
            name="ArCF4_ArN2_raw_vs_generated_1bar",
            output_csv=COMPARATION_DIR / "csv" / "ArCF4_ArN2_raw_vs_generated_1bar.csv",
            output_pdf=COMPARATION_DIR / "plots" / "ArCF4_ArN2_raw_vs_generated_1bar.pdf",
            concentrations_percent=COMPARISON_CONCENTRATIONS_PERCENT,
            wavelength_range_nm=(200.0, 800.0),
            title=r"Raw and generated primary spectra at 1 bar",
            ylim=None,
            curves=(
                ComparisonCurveConfig(
                    name="ArCF4_generated_1bar",
                    gas_mixture="ArCF4",
                    kind="model",
                    pressure_bar=1.0,
                    generated_csv=arcf4_generated,
                    color=BLUE,
                    label=r"Ar--CF$_4$ generated, 1 bar",
                    linewidth=2.15,
                ),
                ComparisonCurveConfig(
                    name="ArCF4_raw_1bar",
                    gas_mixture="ArCF4",
                    kind="raw",
                    pressure_bar=1.0,
                    generated_csv=arcf4_generated,
                    raw_csv=arcf4_raw,
                    norm_parameter_csv=arcf4_norm,
                    spectrum_columns=RAW_ARCF4_SPECTRUM_COLUMNS,
                    color=ORANGE,
                    label=r"Ar--CF$_4$ raw, 1 bar",
                    linewidth=1.25,
                    alpha=0.86,
                    raw_normalisation="area_to_generated",
                    smooth_window=5,
                ),
                ComparisonCurveConfig(
                    name="ArN2_generated_1bar",
                    gas_mixture="ArN2",
                    kind="model",
                    pressure_bar=1.0,
                    generated_csv=arn2_generated,
                    color=RED,
                    label=r"Ar--N$_2$ generated, 1 bar",
                    linewidth=2.00,
                ),
                ComparisonCurveConfig(
                    name="ArN2_raw_1bar",
                    gas_mixture="ArN2",
                    kind="raw",
                    pressure_bar=1.0,
                    generated_csv=arn2_generated,
                    raw_csv=arn2_raw,
                    norm_parameter_csv=arcf4_norm,
                    spectrum_columns=RAW_ARN2_SPECTRUM_COLUMNS,
                    color=GREEN,
                    label=r"Ar--N$_2$ raw, 1 bar",
                    linewidth=1.20,
                    alpha=0.86,
                    raw_normalisation="area_to_generated",
                    smooth_window=5,
                ),
            ),
        ),
        ComparisonMosaicConfig(
            name="ArCF4_raw_vs_generated_1bar_4bar",
            output_csv=COMPARATION_DIR / "csv" / "ArCF4_raw_vs_generated_1bar_4bar.csv",
            output_pdf=COMPARATION_DIR / "plots" / "ArCF4_raw_vs_generated_1bar_4bar.pdf",
            concentrations_percent=COMPARISON_CONCENTRATIONS_PERCENT,
            wavelength_range_nm=(200.0, 800.0),
            title=r"Raw and generated Ar--CF$_4$ primary spectra",
            ylim=None,
            curves=(
                ComparisonCurveConfig(
                    name="ArCF4_generated_1bar",
                    gas_mixture="ArCF4",
                    kind="model",
                    pressure_bar=1.0,
                    generated_csv=arcf4_generated,
                    color=BLUE,
                    label=r"Generated, 1 bar",
                    linewidth=2.15,
                ),
                ComparisonCurveConfig(
                    name="ArCF4_raw_1bar",
                    gas_mixture="ArCF4",
                    kind="raw",
                    pressure_bar=1.0,
                    generated_csv=arcf4_generated,
                    raw_csv=arcf4_raw,
                    norm_parameter_csv=arcf4_norm,
                    spectrum_columns=RAW_ARCF4_SPECTRUM_COLUMNS,
                    color=ORANGE,
                    label=r"Raw, 1 bar",
                    linewidth=1.25,
                    alpha=0.86,
                    raw_normalisation="area_to_generated",
                    smooth_window=5,
                ),
                ComparisonCurveConfig(
                    name="ArCF4_generated_4bar",
                    gas_mixture="ArCF4",
                    kind="model",
                    pressure_bar=4.0,
                    generated_csv=arcf4_generated,
                    color=RED,
                    label=r"Generated, 4 bar",
                    linewidth=2.15,
                ),
                ComparisonCurveConfig(
                    name="ArCF4_raw_4bar",
                    gas_mixture="ArCF4",
                    kind="raw",
                    pressure_bar=4.0,
                    generated_csv=arcf4_generated,
                    raw_csv=arcf4_raw,
                    norm_parameter_csv=arcf4_norm,
                    spectrum_columns=RAW_ARCF4_SPECTRUM_COLUMNS,
                    color=GREEN,
                    label=r"Raw, 4 bar",
                    linewidth=1.25,
                    alpha=0.86,
                    raw_normalisation="area_to_generated",
                    smooth_window=5,
                ),
            ),
        ),
    )


def annotated_spectra_configs() -> tuple[AnnotatedScriptConfig, ...]:
    scripts_dir = ANNOTATED_DIR
    outputs = ANNOTATED_PLOTS_DIR
    return (
        AnnotatedScriptConfig("ar_pure", scripts_dir / "raw_spectra_Ar_annotated.py", outputs / "Ar_pure_raw_1bar.pdf"),
        AnnotatedScriptConfig("cf4_pure", scripts_dir / "raw_spectra_CF4_annotated.py", outputs / "CF4_pure_raw_1bar.pdf"),
        AnnotatedScriptConfig("cf4_pure_secondary_template", scripts_dir / "raw_spectra_CF4_sec.py", outputs / "CF4_pure_secondary_raw_1bar.pdf"),
        AnnotatedScriptConfig("arcf4_9505_primary", scripts_dir / "raw_spectra_ArCF4_9505_annotated.py", outputs / "ArCF4_9505_raw_1bar.pdf"),
        AnnotatedScriptConfig("arcf4_9505_secondary_template", scripts_dir / "raw_spectra_ArCF4_9505_sec_annotated.py", outputs / "ArCF4_9505_secondary_raw_1bar.pdf"),
        AnnotatedScriptConfig("arcf4_9901_primary", scripts_dir / "raw_spectra_ArCF4_9901_annotated.py", outputs / "ArCF4_9901_raw_1bar.pdf"),
        AnnotatedScriptConfig("arcf4_9901_secondary_template", scripts_dir / "raw_spectra_ArCF4_9901_sec_annotated.py", outputs / "ArCF4_9901_secondary_raw_1bar.pdf"),
        AnnotatedScriptConfig("arn2_9901_primary", scripts_dir / "raw_spectra_ArN2_annotated.py", outputs / "ArN2_9901_raw_1bar.pdf"),
        AnnotatedScriptConfig("hecf4_8020_primary_template", scripts_dir / "raw_spectra_HeCF4_8020_annotated.py", outputs / "HeCF4_8020_primary_raw_1bar.pdf"),
        AnnotatedScriptConfig("hecf4_8020_secondary_template", scripts_dir / "raw_spectra_HeCF4_8020_sec_annotated.py", outputs / "HeCF4_8020_secondary_raw_1bar.pdf"),
        AnnotatedScriptConfig("mosaic_cf4_arcf4_hecf4", scripts_dir / "raw_spectra_mosaic_CF4_ArCF4_HeCF4.py", outputs / "mosaic_CF4_ArCF4_HeCF4_2x3.pdf"),
    )
