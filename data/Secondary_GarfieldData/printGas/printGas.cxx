#include <fstream>
#include <iostream>
#include <cstdio>
#include <streambuf>
#include <string>
#include <vector>

#include "Garfield/MediumMagboltz.hh"

using namespace Garfield;

// -----------------------------------------------------------------------------
// Convert collision type to string
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Safe CSV string
// -----------------------------------------------------------------------------

std::string Csv(const std::string& s) {
  std::string out = "\"";

  for (char c : s) {
    if (c == '"') {
      out += "\"\"";
    } else {
      out += c;
    }
  }

  out += "\"";
  return out;
}

// -----------------------------------------------------------------------------
// Save gas.PrintGas() to a text file
// -----------------------------------------------------------------------------

void SavePrintGas(MediumMagboltz& gas,
                  const std::string& outTxt) {
  std::ofstream fout(outTxt);

  if (!fout.is_open()) {
    std::cerr << "Error: could not open " << outTxt << "\n";
    return;
  }

  // Redirect std::cout to file.
  std::streambuf* oldCout = std::cout.rdbuf();
  std::cout.rdbuf(fout.rdbuf());

  gas.PrintGas();

  // Restore std::cout.
  std::cout.rdbuf(oldCout);

  fout.close();

  std::cout << "Written: " << outTxt << "\n";
}

// -----------------------------------------------------------------------------
// Export all Magboltz/Garfield levels to CSV
// -----------------------------------------------------------------------------

void ExportAllLevelsCsv(MediumMagboltz* gas,
                        const std::string& outCsv,
                        const std::vector<std::string>& gasNames) {
  std::ofstream fout(outCsv);

  if (!fout.is_open()) {
    std::cerr << "Error: could not open " << outCsv << "\n";
    return;
  }

  fout << "level,"
       << "gas_index,"
       << "gas,"
       << "state_name,"
       << "collision_type_id,"
       << "collision_type,"
       << "energy_eV,"
       << "n_collisions\n";

  const int nLevels = gas->GetNumberOfLevels();

  for (int i = 0; i < nLevels; ++i) {
    int ngas = -1;
    int type = -1;
    double e = 0.;
    std::string descr;

    if (!gas->GetLevel(i, ngas, type, descr, e)) continue;

    std::string gasName = "unknown";

    if (ngas >= 0 && ngas < static_cast<int>(gasNames.size())) {
      gasName = gasNames[ngas];
    }

    const int nColl = gas->GetNumberOfElectronCollisions(i);

    fout << i << ","
         << ngas << ","
         << Csv(gasName) << ","
         << Csv(descr) << ","
         << type << ","
         << Csv(CollisionTypeToString(type)) << ","
         << e << ","
         << nColl << "\n";
  }

  fout.close();

  std::cout << "Written: " << outCsv << "\n";
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main() {
  MediumMagboltz gas;

  // ---------------------------------------------------------------------------
  // Gas mixture.
  // IMPORTANT: gasNames must follow the same order as SetComposition.
  // ---------------------------------------------------------------------------

  gas.SetComposition("ar", 90., "cf4", 10.);

  const std::vector<std::string> gasNames = {
    "Ar",
    "CF4"
  };

  const std::string prefix = "ArCF4";

  gas.SetTemperature(293.15);
  gas.SetPressure(760.);

  // Maximum electron energy used for the Magboltz tables.
  gas.SetMaxElectronEnergy(400.0);

  // Optional: writes Magboltz cross sections to cs.txt.
  // Must be called before Initialise().
  gas.EnableCrossSectionOutput();

  // Same initialisation block as in the avalanche code.
  gas.EnableDebugging();
  gas.PrintGas();

  if (!gas.Initialise()) {
    std::cerr << "Error: could not initialise gas.\n";
    gas.DisableDebugging();
    return 1;
  }

  gas.DisableDebugging();

  // ---------------------------------------------------------------------------
  // Read the Magboltz/Garfield levels explicitly.
  // ---------------------------------------------------------------------------

  const int nLevels = gas.GetNumberOfLevels();
  std::printf("# of levels: %i\n", nLevels);

  int ngas = 0;
  int type = 0;
  std::string description;
  double en = 0.;

  for (int il = 0; il < nLevels; ++il) {
    if (!gas.GetLevel(il, ngas, type, description, en)) continue;

    std::cout << "level " << il
              << " | gas index = " << ngas
              << " | type = " << type
              << " (" << CollisionTypeToString(type) << ")"
              << " | energy = " << en << " eV"
              << " | description = " << description
              << "\n";
  }

  // Save PrintGas output.
  SavePrintGas(gas, prefix + "_PrintGas.txt");

  // Export all levels to CSV.
  ExportAllLevelsCsv(&gas, prefix + "_all_levels.csv", gasNames);

  std::cout << "\nDone.\n";
  std::cout << "Generated files:\n";
  std::cout << "  " << prefix << "_PrintGas.txt\n";
  std::cout << "  " << prefix << "_all_levels.csv\n";
  std::cout << "  cs.txt\n";

  return 0;
}