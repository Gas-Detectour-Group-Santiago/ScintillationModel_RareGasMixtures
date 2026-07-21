# Validation status

Validated in this package:

- exactly four canonical production-figure registries;
- registry references to fit models, experimental datasets, secondary selections and outputs;
- statistical-only fit point error bars;
- optional X-ray scaling with the selected primary normalisation;
- compressed NPZ toy loading and cached point/band reuse;
- Ar–CF₄/Ar–N₂/He–CF₄ Garfield campaign discovery;
- Degrad population discovery, OCW loading and spectrum fallback tables;
- full cached `run_products.sh` execution with primary, secondary, spectra and LaTeX stages enabled;
- all configured secondary scans and legacy-equivalent UV/VIS/IR/VUV comparison families;
- primary electron/X-ray products, low-pressure products and joint-IR plots;
- all 22 registered spectral PDFs, including annotated figures, generated 3×3 mosaics, extended VUV views, insets and broken-x views;
- shell syntax, Python compilation and 17 unit tests.

The four distributed registries currently contain:

```text
fits.csv       30 rows / 30 figure IDs
primary.csv    57 rows / 31 figure IDs
secondary.csv  64 rows / 22 figure IDs
spectra.csv    22 rows / 22 figure IDs
```

Large spectral families run as isolated short-lived processes. Numerical-library thread counts and allocator arenas are bounded in the compatibility runtime, and generated spectral grids are read from compressed caches using compact dtypes. The compatibility workspace signature ignores bytecode and editor artefacts, so one stage no longer deletes plots produced by a preceding stage.

A completely fresh `run_all.sh` with every fit regenerated at 100 statistical and 100 systematic toys was not repeated during this packaging cycle. That remains the expensive end-to-end numerical validation. New fits automatically force `RECOMPUTE_BANDS=1` and `RECOMPUTE_TABLES=1`; ordinary product runs reuse compact caches.

## Graphical interface validation

The GUI uses four stable pages:

- Run pipeline;
- Figure recipes;
- Outputs;
- Plot style.

The figure page edits `config/plots/{fits,primary,secondary,spectra}.csv` directly. The output page reads only official PDFs and LaTeX tables. The style preview exercises primary/secondary lines, error bars, caps, uncertainty bands, spines, major/minor tick widths and lengths, grid and legend controls.

The interface calls the same two shell runners and contains no second copy of the physical equations. A complete fresh 100-toy fit was not launched through the browser during this packaging cycle.
