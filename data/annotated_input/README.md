# annotated_input

CSV inputs used by `spectra/run_annotated_spectra.py` for annotated spectra that are not read from the main experimental pickles.

Expected format for these files:

```text
wavelength_nm; intensity
```

The parser assumes semicolon-separated columns and decimal comma, matching the original experimental files.
