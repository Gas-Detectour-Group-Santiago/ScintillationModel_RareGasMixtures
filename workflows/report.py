#!/usr/bin/env python3
from __future__ import annotations
from _bootstrap import ROOT
from scintillation.reporting.reference_tables import export_second_continuum_parameter_table


def main() -> None:
    path = export_second_continuum_parameter_table(ROOT)
    print(f"[workflow] reference table: {path}")


if __name__ == "__main__":
    main()
