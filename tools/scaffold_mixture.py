#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the folders and templates for a new gas mixture")
    parser.add_argument("mixture_id", help="e.g. ArCO2")
    parser.add_argument("--additive", required=True, help="e.g. CO2")
    parser.add_argument("--base-gas", default="Ar")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    mixture = args.mixture_id
    for path in (
        root / "data" / "raw" / "degrad" / mixture / "txt",
        root / "data" / "raw" / "garfield" / mixture / "root",
        root / "data" / "raw" / "experimental" / mixture,
        root / "data" / "processed" / "experimental" / mixture,
    ):
        path.mkdir(parents=True, exist_ok=True)

    level_template = root / "data" / "reference" / "levels" / f"{mixture}_level_data.csv"
    if not level_template.exists():
        level_template.write_text("level,gas,state_name,type,energy_eV\n", encoding="utf-8")

    print(f"Created folders for {mixture}.")
    print("Next steps:")
    print(f"  1. Add Degrad TXT files to data/raw/degrad/{mixture}/txt/")
    print(f"  2. Add Garfield ROOT files to data/raw/garfield/{mixture}/root/")
    print(f"  3. Fill data/reference/levels/{mixture}_level_data.csv")
    print("  4. Add rows to config/mixtures.csv, primary_population_groups.csv,")
    print("     secondary_inputs.csv, population_groups.csv and, if needed, channels.csv.")
    print(f"  5. Add {args.additive} quenching rates to data/reference/parameters/Ar2nd_continium.csv.")


if __name__ == "__main__":
    main()
