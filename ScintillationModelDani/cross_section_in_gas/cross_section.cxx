#include <iostream>
#include "Garfield/MediumMagboltz.hh"

#include "TApplication.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TPad.h"
#include "TObject.h"
#include "TLegend.h"
#include "TH1.h"
#include "TAxis.h"

static void SetPaperStyle() {
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  gStyle->SetTextFont(42);
  gStyle->SetLabelFont(42, "XYZ");
  gStyle->SetTitleFont(42, "XYZ");

  gStyle->SetTitleSize(0.050, "XYZ");
  gStyle->SetLabelSize(0.045, "XYZ");

  gStyle->SetTitleOffset(1.10, "X");
  gStyle->SetTitleOffset(1.25, "Y");

  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);

  gStyle->SetLineWidth(2);
  gStyle->SetFrameLineWidth(2);
}

static void BeautifyAfterPlot(TCanvas* c) {
  c->SetLeftMargin(0.14);
  c->SetRightMargin(0.04);
  c->SetBottomMargin(0.12);
  c->SetTopMargin(0.06);

  c->SetGridx(0);
  c->SetGridy(0);

  // ROOT suele dibujar una "histo base" para los ejes. La buscamos y retocamos.
  TH1* h = nullptr;
  for (auto* obj : *c->GetListOfPrimitives()) {
    h = dynamic_cast<TH1*>(obj);
    if (h) break;
  }
  if (h) {
    h->GetXaxis()->CenterTitle(true);
    h->GetYaxis()->CenterTitle(true);
    h->GetXaxis()->SetNdivisions(510);
    h->GetYaxis()->SetNdivisions(510);

    // Ajusta rangos si quieres (en Mbarn)
    // h->GetXaxis()->SetRangeUser(0.05, 2000.0);
    // h->SetMinimum(1e-2);
    // h->SetMaximum(3e3);
  }

  // Si hay leyenda, la recolocamos y la dejamos limpia.
  for (auto* obj : *c->GetListOfPrimitives()) {
    auto* leg = dynamic_cast<TLegend*>(obj);
    if (!leg) continue;
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextFont(42);
    leg->SetTextSize(0.040);

    // Reposiciona (x1,y1,x2,y2) en coordenadas NDC
    leg->SetX1NDC(0.15);
    leg->SetY1NDC(0.15);
    leg->SetX2NDC(0.35);
    leg->SetY2NDC(0.35);
  }

  c->Modified();
  c->Update();
}

int main(int argc, char** argv) {
  TApplication app("app", &argc, argv);
  gROOT->SetBatch(kTRUE);

  SetPaperStyle();

  using namespace Garfield;

  MediumMagboltz gas;
  gas.SetComposition("ar", 100.0);
  gas.SetTemperature(293.15);
  gas.SetPressure(760.0);
  gas.SetMaxElectronEnergy(20000.0);
  gas.EnableCrossSectionOutput();
  gas.Initialise(true);

  auto* c1 = new TCanvas("c1", "Electron cross sections", 900, 750);
  gas.PlotElectronCrossSections(0, c1);

  BeautifyAfterPlot(c1);
  
  gas.PrintGas();

  c1->SaveAs("../cross_sections_ar_pretty.pdf");

  return 0;
}
