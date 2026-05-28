# Energy level diagram

Programa sencillo en Python para generar un diagrama de niveles de energía tipo paper.

## Requisitos

```bash
pip install numpy matplotlib
```

## Uso

```bash
python energy_level_diagram.py
```

El script genera:

```text
energy_level_diagram.pdf
```

## Qué tocar normalmente

- `level(...)` para añadir un nivel individual.
- `manifold(...)` para añadir muchas líneas próximas.
- `transition(...)` para añadir una flecha.
- Las variables `x_xe_liq`, `x_ar_liq`, etc. para mover columnas horizontalmente.
- Las energías, que son el segundo argumento de `level(...)` o los rangos `y_min`, `y_max` en `manifold(...)`.

La figura está pensada como plantilla: se puede adaptar fácilmente a Ar-CF4, CF4, CF3*, Ar2*, etc.
