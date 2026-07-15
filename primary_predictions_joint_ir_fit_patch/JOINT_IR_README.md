# Joint IR fit (Ar-CF4 + Ar-N2)

This optional path is independent of the legacy `ArCF4_IR_primary` and
`ArN2_IR_primary` fits.

## Physics constraints

For each Ar IR line (696, 727, 750, 764 and 772 nm), the joint fit shares:

- `PAr_star`: optical weight / low-pressure amplitude;
- `K_Ar_Q_Ar`: Ar self-quenching.

It keeps independent:

- `K_Ar_Q_CF4`;
- `K_Ar_Q_N2`.

The model is evaluated at the exact experimental concentration. It does not
trim arrays to force equal lengths and does not apply a first-point anchor.

## Run

```bash
python primary_fits/ArJoint_IR_fit.py
python primary_predictions/run_joint_ir_predictions.py
```

The fit defaults to 300 statistical and 300 systematic toys. For a quick test:

```bash
JOINT_IR_N_TOYS=50 python primary_fits/ArJoint_IR_fit.py
```

The default fit window includes admixtures up to 20%:

```bash
JOINT_IR_MAX_CONCENTRATION_PERCENT=20 python primary_fits/ArJoint_IR_fit.py
```

## Main outputs

- `data/Parameters/ArJoint_IR_primary.csv`
- `data/Tables/ArJoint_IR_primary_param_stat_syst.tex`
- `data/Predictions/ArJoint_IR_primary_low_pressure_pure_ar.csv`
- `data/Tables/ArJoint_IR_primary_low_pressure_pure_ar.tex`
- `data/Predictions/Bands/ArJoint_IR/*.csv`
- `primary_predictions/plots/primary_bands/joint_ir/*.pdf`
- `primary_fits/plots/plot_fit/ArJoint_IR/*.pdf`

The products bundled in this patch were generated with 50 statistical and 50
systematic toys. The code itself keeps 300 as the default for a final run.
