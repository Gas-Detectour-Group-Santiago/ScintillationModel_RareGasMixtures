from __future__ import annotations

from dataclasses import dataclass


# =============================================================================
# Main switches
# =============================================================================
# Se leen y se procesan las tres columnas. Para los plots de producción se usa
# mean_spectrum, pero los CSVs agregados conservan mean_spectrum, C1_spectrum y
# C2_spectrum diferenciados.
SPECTRUM_COLUMNS = ("mean_spectrum", "C1_spectrum", "C2_spectrum")
RAW_PLOT_SPECTRUM_COLUMN = "mean_spectrum"
COMPARISON_SPECTRUM_COLUMN = "mean_spectrum"

MAKE_RAW_MOSAICS = True
MAKE_GENERATED_MOSAICS = True
MAKE_COMPARISON_MOSAICS = True
MAKE_ANNOTATED_FIGURES = True

# Comparisons mix raw spectra and generated spectra. Choose the unit conversion
# before the anchor matching:
#   "raw_to_generated" : raw -> generated units, ph MeV^-1 nm^-1
#   "generated_to_raw" : generated -> raw-like units, ph e^-1 nm^-1
#   "none"             : no unit conversion, only anchor scaling
COMPARISON_UNIT_SCALING = "raw_to_generated"
X_RAY_ENERGY_EV = 1.0e6
USE_NNORM_IN_RAW_UNIT_SCALING = True

# After unit conversion, force raw and generated to coincide at one reference
# peak. The reference is defined independently for ArCF4 and ArN2 below.
ANCHOR_MATCH_ENABLED = True
ANCHOR_SCALE_SIDE = "raw"  # "raw" or "generated"
RAW_SMOOTH_WINDOW = 3

# Comparison visual style: raw data is a thick translucent solid curve; the
# prediction is a thinner solid curve drawn on top.
COMPARISON_RAW_LINEWIDTH = 2.65
COMPARISON_RAW_ALPHA = 0.38
COMPARISON_GENERATED_LINEWIDTH = 1.15
COMPARISON_GENERATED_ALPHA = 0.98

# When several experimental spectra exist for the same gas/concentration/pressure
# and wavelength, produce one averaged intensity per condition. The original
# spectra are not plotted as one connected line anymore.
RAW_AGGREGATE_REPLICATES = True
RAW_WAVELENGTH_ROUND_DECIMALS = 6
RAW_PREFER_PICKLES = True
RAW_PICKLE_FILES = {
    "ArCF4": "data/Experimental/ArCF4/CF4_data.pkl",
    "ArN2": "data/Experimental/ArN2/N2_data.pkl",
}
RAW_CONCENTRATION_COLUMNS = {
    "ArCF4": ("concentracion", "concentration_CF4", "CF4 concentration (%)"),
    "ArN2": ("N2 concentration (%)", "concentration_N2", "concentracion"),
}
RAW_PRESSURE_COLUMNS = {
    "ArCF4": ("presion", "P (bar)"),
    "ArN2": ("P (bar)", "presion"),
}
RAW_SPECTRUM_COLUMN_CANDIDATES = {
    "mean_spectrum": ("mean_spectrum", "spectrum_new_cal", "data(norm)"),
    "C1_spectrum": ("C1_spectrum", "C1"),
    "C2_spectrum": ("C2_spectrum", "C2"),
}


# =============================================================================
# Grids
# =============================================================================
# Raw: paneles por concentración, curvas por presión.
RAW_CONCENTRATIONS_PERCENT = (0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0)
RAW_PRESSURES_BAR = (1, 2, 3, 4, 5)
RAW_REFERENCE_CONCENTRATION_PERCENT = 5.0
RAW_REFERENCE_PRESSURE_BAR = 1.0
RAW_REFERENCE_COLOR = "magenta"
RAW_REFERENCE_ALPHA = 0.22

# Generated: paneles por concentración, curvas por presión (6 curvas/panel).
GENERATED_PRESSURES_BAR = (1, 2, 3, 4, 5, 10)
GENERATED_CONCENTRATIONS_PERCENT = (0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0)

# Comparisons: paneles por concentración, curvas raw/generated a 1 y 4 bar.
COMPARISON_PRESSURES_BAR = (1, 4)
COMPARISON_CONCENTRATIONS_PERCENT = GENERATED_CONCENTRATIONS_PERCENT


# =============================================================================
# Wavelength ranges and anchors
# =============================================================================
WAVELENGTH_RANGE_RAW_NM = (180.0, 800.0)
WAVELENGTH_RANGE_GENERATED = {
    "ArCF4": (200.0, 800.0),
    "ArN2": (300.0, 800.0),
}
# Extended/generated VUV spectra.  These keep the normal generated spectra intact
# and optionally add the Ar second continuum plus, for Ar--CF4, the CF4 ionic VUV branch.
WAVELENGTH_RANGE_GENERATED_AMPLIED = {
    "ArCF4": (100.0, 800.0),
    "ArN2": (100.0, 800.0),
}
WAVELENGTH_RANGE_COMPARISON_NM = (180.0, 800.0)
WAVELENGTH_POINTS = 2000
WAVELENGTH_POINTS_AMPLIED = 2500

AR2ND_CONTINIUM_PARAMETER_CSV = "data/Parameters/Ar2nd_continium.csv"
AR2ND_DEGRAD_CSVS = {
    "ArCF4": "data/Primary_DegradData/ArCF4_Ar2nd.csv",
    "ArN2": "data/Primary_DegradData/ArN2_Ar2nd.csv",
}

# Extended/generated VUV switch. The amplied spectra include the current Ar
# second-continuum prediction directly.  The Ar second continuum is not anchored
# and is not divided by the fitted primary Nnorm; conversion to ph MeV^-1 uses
# the X-ray energy of each degradation calculation.
AMPLIED_INCLUDE_AR2ND_CONTINIUM = True

# The 0% additive panel is pure argon in both Ar--CF4 and Ar--N2 mosaics.
# By default we keep the two dedicated Degrad extractions separate, because the
# generated spectrum uses the total second-continuum emission in the
# extended VUV plots, while preserving the branch-dependent Degrad inputs for
# Ar--CF4 and Ar--N2.
# Set AR2ND_FORCE_COMMON_PURE_REFERENCE=True only for a debugging plot where the
# two pure-Ar panels are intentionally forced to use one identical input.
AR2ND_FORCE_COMMON_PURE_REFERENCE = False
AR2ND_PURE_REFERENCE_GAS = "ArN2"
AR2ND_PURE_REFERENCE_CONCENTRATION_PERCENT = 0.0

# Anchors at 95/5: ArCF4 visible band and ArN2 327 nm peak.
ANCHORS = {
    "ArCF4": {"concentration_percent": 5.0, "window_nm": (500.0, 750.0)},
    "ArN2": {"concentration_percent": 5.0, "window_nm": (320.0, 340.0)},
}

# Every panel in each mosaic can share the same y-limit. This is especially
# important for generated spectra, where otherwise each concentration panel
# rescales independently and comparisons become misleading.
RAW_SHARE_YLIM = True
GENERATED_SHARE_YLIM = True
COMPARISON_SHARE_YLIM = True

# The extended/generated VUV spectra may contain strong sub-200 nm structure. In
# linear scale this can compress the UV/VIS/IR region, so the amplied generated
# mosaics can be drawn in log-y scale when desired.
GENERATED_AMPLIED_LOG_SCALE = False
GENERATED_AMPLIED_LOG_YMIN = 1.0e0
GENERATED_AMPLIED_LOG_YMAX_FACTOR = 1.35

# Alternative layouts for the extended VUV generated mosaics. These are meant
# to keep the VUV branch visible without hiding the optical structure.
GENERATED_AMPLIED_INSET_ENABLED = True
GENERATED_AMPLIED_BROKENX_ENABLED = True
GENERATED_AMPLIED_VUV_WINDOW_NM = (110.0, 170.0)
# In the inset layout the main axes still show the full 100--800 nm range so the
# reader sees how dominant the VUV branch is. Their y-limits, however, are set
# from an optical-only sub-window so the weaker UV/VIS/IR lines remain visible
# and the VUV peak is intentionally clipped on the main axes.
GENERATED_AMPLIED_MAIN_WINDOW = {
    "ArCF4": WAVELENGTH_RANGE_GENERATED_AMPLIED["ArCF4"],
    "ArN2": WAVELENGTH_RANGE_GENERATED_AMPLIED["ArN2"],
}
GENERATED_AMPLIED_MAIN_YLIM_WINDOW = {
    "ArCF4": (200.0, 800.0),
    "ArN2": (300.0, 800.0),
}
GENERATED_AMPLIED_INSET_LOC = "upper right"
GENERATED_AMPLIED_INSET_WIDTH = "45%"
GENERATED_AMPLIED_INSET_HEIGHT = "45%"
# Extra headroom on the main axis so the narrow infrared lines do not sit right
# against the top frame when the VUV is moved to the inset.
GENERATED_AMPLIED_INSET_MAIN_YMAX_FACTOR = 1.32
GENERATED_AMPLIED_INSET_VUV_YMAX_FACTOR = 1.12
GENERATED_AMPLIED_BROKENX_WIDTH_RATIOS = (1.0, 3.2)


# =============================================================================
# Output
# =============================================================================
OUTPUT_DIRNAME = "spectra"
ANNOTATED_INPUT_DIRNAME = "annotated_input"


@dataclass(frozen=True)
class GasFiles:
    raw_csv: str
    degrad_csv: str
    degrad_ir_csv: str
    parameter_csv: str
    ir_parameter_csv: str
    norm_parameter_csv: str


GASES = {
    "ArCF4": GasFiles(
        raw_csv="data/Spectra/ArCF4_raw_spectra.csv",
        degrad_csv="data/Primary_DegradData/ArCF4.csv",
        degrad_ir_csv="data/Primary_DegradData/ArCF4_IR.csv",
        parameter_csv="data/Parameters/ArCF4_primary.csv",
        ir_parameter_csv="data/Parameters/ArCF4_IR_primary.csv",
        norm_parameter_csv="data/Parameters/ArCF4_primary.csv",
    ),
    "ArN2": GasFiles(
        raw_csv="data/Spectra/ArN2_raw_spectra.csv",
        degrad_csv="data/Primary_DegradData/ArN2.csv",
        degrad_ir_csv="data/Primary_DegradData/ArN2_IR.csv",
        parameter_csv="data/Parameters/ArN2_primary.csv",
        ir_parameter_csv="data/Parameters/ArN2_IR_primary.csv",
        norm_parameter_csv="data/Parameters/ArCF4_primary.csv",
    ),
}


COMPARISON_PLOTS = (
    {
        "name": "ArCF4_raw_generated_1bar_4bar",
        "title": r"Ar--CF$_4$: raw vs prediction, 1 and 4 bar",
        "gases": ("ArCF4",),
        "pressures_bar": COMPARISON_PRESSURES_BAR,
        "concentrations_percent": COMPARISON_CONCENTRATIONS_PERCENT,
        "output_pdf": "comparison_ArCF4_raw_generated_1bar_4bar.pdf",
        "output_csv": "comparison_ArCF4_raw_generated_1bar_4bar.csv",
    },
    {
        "name": "ArN2_raw_generated_1bar_4bar",
        "title": r"Ar--N$_2$: raw vs prediction, 1 and 4 bar",
        "gases": ("ArN2",),
        "pressures_bar": COMPARISON_PRESSURES_BAR,
        "concentrations_percent": COMPARISON_CONCENTRATIONS_PERCENT,
        "output_pdf": "comparison_ArN2_raw_generated_1bar_4bar.pdf",
        "output_csv": "comparison_ArN2_raw_generated_1bar_4bar.csv",
    },
)
