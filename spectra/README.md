# spectra

Generador limpio de espectros raw, generated y comparación raw/generated.

## Ejecución

Desde la raíz del proyecto:

```bash
python3 spectra/run_all_spectra.py
```

También puedes correr partes separadas:

```bash
python3 spectra/run_raw_spectra.py
python3 spectra/run_generated_spectra.py
python3 spectra/run_comparison_spectra.py
```

En Jupyter, `run_all_spectra.py` usa `parse_known_args`, así que no debería fallar por los argumentos internos del kernel.

## Raw experimental

El raw se procesa correctamente como CSV largo y agregado:

```text
gas_mixture
concentration_percent
pressure_bar
spectrum_column          # mean_spectrum, C1_spectrum o C2_spectrum
wavelength_nm
intensity_mean
intensity_std
intensity_sem
n_replicates
n_points_averaged
intensity_raw            # alias de intensity_mean para plotting/comparison
```

Es decir: para cada gas, concentración, presión, columna espectral y longitud de onda queda **una sola intensidad**. Si hay varias réplicas experimentales para la misma condición, se promedian y se conserva la dispersión.

La configuración de columnas está en `config.py`:

```python
SPECTRUM_COLUMNS = ("mean_spectrum", "C1_spectrum", "C2_spectrum")
RAW_PLOT_SPECTRUM_COLUMN = "mean_spectrum"
COMPARISON_SPECTRUM_COLUMN = "mean_spectrum"
```

Los mosaicos raw usan `RAW_PLOT_SPECTRUM_COLUMN`, pero los CSVs agregados guardan las tres columnas.

## Generated

Los generated son mosaicos 3x3 por concentración. En cada panel hay seis curvas, una por presión:

```python
GENERATED_PRESSURES_BAR = (1, 2, 3, 4, 5, 10)
```

## Salidas

Todo queda dentro de la propia carpeta:

```text
spectra/csv/
spectra/plots/
```

## Annotated

También hay un cuarto runner para las figuras anotadas:

```bash
python3 spectra/run_annotated_spectra.py
```

Genera los PDFs en:

```text
spectra/plots/annotated/
```

Los CSVs externos necesarios para esas figuras están en:

```text
data/annotated_input/
```

Los espectros primarios que ya existen en los pickles principales se leen desde `data/Experimental/...`; los espectros secundarios/He--CF4 se leen desde `data/annotated_input/`.

`run_all_spectra.py` también ejecuta las figuras anotadas por defecto. Se puede desactivar con:

```bash
python3 spectra/run_all_spectra.py --no-annotated
```

## Comparaciones

Las comparaciones usan curvas sólidas en todos los casos. El raw se dibuja primero, grueso y transparente; la predicción se dibuja encima, más fina. La comparación de nitrógeno es ahora solo Ar--N2 a 1 y 4 bar:

```text
spectra/plots/comparison_ArN2_raw_generated_1bar_4bar.pdf
spectra/csv/comparison_ArN2_raw_generated_1bar_4bar.csv
```
