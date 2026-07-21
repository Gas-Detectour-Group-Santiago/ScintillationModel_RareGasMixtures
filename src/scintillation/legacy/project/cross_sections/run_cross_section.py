#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compila y ejecuta cross_section.cxx (Garfield++)
"""

import subprocess
import os
from pathlib import Path
from multiprocessing import cpu_count


# ----------------------------------------
# CONFIGURACIÓN
# ----------------------------------------
SOURCE_FILE = "cross_section.cxx"
EXECUTABLE  = "cross_section"
BUILD_DIR   = "build"


# ----------------------------------------
# UTILIDADES
# ----------------------------------------
def run(cmd, cwd=None):
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


# ----------------------------------------
# MAIN
# ----------------------------------------
def main():

    # Comprobación básica
    if not Path(SOURCE_FILE).exists():
        raise FileNotFoundError(f"No existe {SOURCE_FILE}")

    # Limpieza
    if Path(BUILD_DIR).exists():
        run(["rm", "-rf", BUILD_DIR])

    os.makedirs(BUILD_DIR)

    # Configuración CMake
    run(["cmake", ".."], cwd=BUILD_DIR)

    # Compilación
    nproc = cpu_count()
    run(["make", f"-j{nproc}"], cwd=BUILD_DIR)

    # Comprobación del ejecutable
    exe_path = Path(BUILD_DIR) / EXECUTABLE
    if not exe_path.exists():
        raise RuntimeError(f"No se generó el ejecutable {EXECUTABLE}")

    # Ejecución
    run([f"./{EXECUTABLE}"], cwd=BUILD_DIR)

    run(["python","plot_cross_sections.py"])

    print("\n✔ cross_section.cxx compilado y ejecutado correctamente\n")


if __name__ == "__main__":
    main()