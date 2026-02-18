#!/bin/bash
set -e

echo "==============================="
echo " Ejecutando Read_Degrad.py"
echo "==============================="
python3 Read_Degrad.py

echo "==============================="
echo " Ejecutando Read_Experimental_Yield.py"
echo "==============================="
python3 Read_Experimental_Yield.py
python3 Read_Experimental_Yield_NoSistematic.py

echo "==============================="
echo " Ejecutando main.py"
echo "==============================="

cd NewModel_NoSistematic/
python3 main.py
cd ..


#cd PabloModel_NoSistematic/
#python3 main.py
#cd ..

#cd NewModel_All/
#python3 main.py
#cd ..


#cd NewModel_All_P1/
#python3 main.py
#cd ..


#cd NewModel_NoSistematic_P1/
#python3 main.py
#cd ..



echo "==============================="
echo "   ✔ Todos los programas ejecutados correctamente"
echo "==============================="
