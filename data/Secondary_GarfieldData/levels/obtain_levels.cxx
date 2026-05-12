#include <fstream>
#include <string>
#include <vector>
#include "Garfield/MediumMagboltz.hh"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdio>

using namespace Garfield;
using namespace std;

std::string CollisionTypeToString(const int type) {
  switch (type) {
    case 0: return "elastic";
    case 1: return "ionisation";
    case 2: return "attachment";
    case 3: return "inelastic";
    case 4: return "excitation";
    case 5: return "superelastic";
    default: return "unknown";
  }
}

void ExportAllLevelsCsv(Garfield::MediumMagboltz* gas,
                        const std::string& outCsv,
                        const std::vector<std::string>& gasNames) {
  std::ofstream fout(outCsv);
  fout << "level,gas,state_name,type,energy_eV,n_collisions\n";

  const int nLevels = gas->GetNumberOfLevels();
  for (int i = 0; i < nLevels; ++i) {
    int ngas = -1, type = -1;
    double e = 0.;
    std::string descr;

    if (!gas->GetLevel(i, ngas, type, descr, e)) continue;

    std::string gasName = "unknown";
    if (ngas >= 0 && ngas < (int)gasNames.size()) {
      gasName = gasNames[ngas];
    }

    const int nColl = gas->GetNumberOfElectronCollisions(i);

    fout << i << ","
         << "\"" << gasName << "\","
         << "\"" << descr << "\","
         << "\"" << CollisionTypeToString(type) << "\","
         << e << ","
         << nColl << "\n";
  }

  fout.close();
}

int main() {

  MediumMagboltz* gas = new MediumMagboltz();

  gas->SetComposition("he", 90., "cf4", 10.);
  gas->SetTemperature(293.15);
  gas->SetPressure(760.);

  gas->SetMaxElectronEnergy(400.0);

  gas->EnableDebugging();
  gas->PrintGas();

  
  gas->Initialise(true);

  gas->DisableDebugging();

  const int nLevels = gas->GetNumberOfLevels();
  std::cout << "# of levels: " << nLevels << std::endl;

  int ngas = 0;
  int type = 0;
  std::string description;
  double en = 0.;

  for (int il = 0; il < nLevels; ++il) {
    gas->GetLevel(il, ngas, type, description, en);
    
    std::cout << "level " << il
              << " | gas index = " << ngas
              << " | type = " << type
              << " (" << CollisionTypeToString(type) << ")"
              << " | energy = " << en << " eV"
              << " | description = " << description
              << std::endl;
  }

  const std::vector<std::string> gasNames = {"He", "CF4"};

  ExportAllLevelsCsv(gas, "HeCF4_level_data.csv", gasNames);

  delete gas;
  return 0;
}