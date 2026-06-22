# spectra_generator

Pipeline compartimentado para espectros.

Desde esta carpeta:

```bash
python run_spectra_raw.py
python run_spectra_generated.py
python run_spectra_comparation.py
python run_spectra_annotated.py
python run_all_spectra.py
```

Desde cualquier otra carpeta, instala una vez el proyecto en modo editable:

```bash
pip install -e /ruta/a/ScintillationModel_RareGasMixtures_2
sm-spectra all
sm-spectra raw
sm-spectra generated
sm-spectra comparation
sm-spectra annotated
```

En Jupyter:

```python
from spectra_generator.cli import main
main(["all"])
```

Antes de los mosaicos raw, regenera los CSV experimentales con la versión actual:

```bash
python data/Analysis_spectra.py
```

Esto es importante porque las concentraciones pequeñas de los pickles antiguos son porcentajes reales: `0.1` significa `0.1 %`, no una fracción `0.1 = 10 %`.

## Raw spectra

Por defecto se usa solo `mean_spectrum`. Para probar `C1_spectrum`, `C2_spectrum`, `spectrum_new_cal`, etc., cambia las tuplas en `spectra_generator/configs.py`:

```python
RAW_ARCF4_SPECTRUM_COLUMNS = ("mean_spectrum",)
RAW_ARN2_SPECTRUM_COLUMNS = ("mean_spectrum",)
```

Los mosaicos raw son 3x3:

- `spectra_raw/plots/experimental_ArCF4_grid.pdf`
- `spectra_raw/plots/experimental_ArN2_grid_with_fixed_ArCF4_95_5_reference.pdf`

El segundo incluye la referencia fija Ar/CF4 95/5 en magenta.

## Comparisons

La etapa `comparation` genera dos mosaicos limpios raw vs generated:

- `spectra_comparation/plots/ArCF4_ArN2_raw_vs_generated_1bar.pdf`
- `spectra_comparation/plots/ArCF4_raw_vs_generated_1bar_4bar.pdf`

En cada panel se dibujan cuatro curvas. Los raw se tratan como formas espectrales: se normalizan por área al espectro generado correspondiente, para que el tamaño vertical lo fije el yield del modelo y no la escala arbitraria del detector.

El CSV conserva también la conversión directa `raw * 1e6 / W / Nnorm` en la columna `intensity_raw_W_Nnorm` por trazabilidad.

## Annotated

`annotated` usa los scripts/plantillas de `spectra_annotated`. Los CSV secundarios no medidos se tratan como plantillas/reference shapes y no se mezclan con los raw experimentales de `data/Spectra`.

El launcher de la carpeta madre no se usa.
