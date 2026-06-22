# Primary fits

Esta carpeta contiene los ajustes primarios del modelo de centelleo.

Scripts principales:

- `ArCF4_fit.py`: ajuste UV/VIS primario de Ar--CF4.
- `ArN2_fit.py`: ajuste UV primario de Ar--N2.
- `ArCF4_IR_fit.py`: ajuste IR primario de Ar--CF4.
- `ArN2_IR_fit.py`: ajuste IR primario de Ar--N2.
- `run_primary_fits.py`: ejecuta los cuatro.

Cada script define solo una configuración física/visual y delega la lógica común a `auxiliares/`.

## Salidas

Los plots de ajuste se guardan en:

```text
primary_fits/plots/plot_fit/
```

Los resultados reutilizables se guardan en `data/`:

```text
data/Parameters/
data/FitResults/
data/Tables/
```

`data/Parameters/<fit>.csv` contiene el resumen legible de parámetros:

```text
name,value,fit_uncertainty,stat_minus,stat_plus,syst_minus,syst_plus,total_minus,total_plus,fixed
```

Los fits leen directamente los CSVs experimentales ya generados, no los pickles.
Antes de ajustar, genera los CSVs con:

```bash
python data/Analysis_experimental.py
```

Ese script escribe un único CSV por observable con valor, error total, error
estadístico y error sistemático:

```text
data/Experimental/<gas>/csv/<observable>.csv
```

Por ejemplo:

```text
fCF4,1bar,Err 1bar,ErrStat 1bar,ErrSyst 1bar,2bar,Err 2bar,ErrStat 2bar,ErrSyst 2bar,...
```

`primary_fits` lee directamente ese CSV. Usa `Err <p>bar` para el ajuste central,
`ErrStat <p>bar` para los toys estadísticos y `ErrSyst <p>bar` para los toys
sistemáticos.

`data/FitResults/` contiene el ajuste central, toys, covarianza, correlación y metadatos:

```text
<fit>_central.csv
<fit>_toys_stat.csv
<fit>_toys_syst.csv
<fit>_covariance.csv
<fit>_correlation.csv
<fit>_metadata.json
```

`data/Tables/` contiene las tablas LaTeX finales con columnas separadas de estadístico y sistemático.

## Toys

Los toys se configuran desde `ToySpec` dentro de cada script:

```python
ToySpec(
    n_stat=10,
    n_syst=10,
    seed=...,
    n_jobs=-1,
)
```

Para acelerar, se usan los datos ya cargados en memoria, el ajuste central como `x0` de cada toy y, si está instalado, `joblib` para paralelizar.

## Predicciones

Esta carpeta no calcula predicciones ni bandas físicas. Solo guarda la información necesaria para que una futura carpeta `primary_predictions/` lea:

- parámetros centrales;
- toys estadísticos;
- toys sistemáticos;
- covarianzas/metadatos.

Con eso `primary_predictions/` podrá evaluar modelos primarios/secundarios y construir bandas sin repetir los fits.
