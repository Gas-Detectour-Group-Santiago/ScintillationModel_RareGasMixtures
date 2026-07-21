from __future__ import annotations

try:
    from .run_all_spectra import cli, main
except ImportError:
    from run_all_spectra import cli, main  # type: ignore


if __name__ == "__main__":
    cli()
