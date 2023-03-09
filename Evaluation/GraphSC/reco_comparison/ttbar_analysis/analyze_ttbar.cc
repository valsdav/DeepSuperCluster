#include "ROOT/RVec.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RResultPtr.hxx"
#include <TGraph.h>
#include <TChain.h>
#include <TCanvas.h>
#include <TTree.h>
#include <TFile.h>
#include <TObject.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "TString.h"
#include "TMath.h"
#include "TH1D.h"

using namespace std;
using namespace ROOT;
using namespace ROOT::VecOps;
using namespace ROOT::RDF;


std::vector<std::vector<double>> getNxtal(RVec<std::vector<int>> ind , RVec<double> nxtals){
  std::vector<std::vector<double>> result;
  result.reserve(ind.size());
    for ( const auto & is : ind){
      std:vector<double> ol;
      ol.reserve(is.size());
      for (const auto & i : is){
        ol.push_back(nxtals[i]);
      }
      result.push_back(ol);
    }
    return result;
}

int main(){
  ROOT::EnableImplicitMT(5); // Tell ROOT you want to go parallel
  ROOT::RDataFrame d("recosimdumper/caloTree", "/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/TTbar_14TeV_TuneCP5_Pythia8/RawDumper_DeepSC_12_5_0/ttbar_Run3_DeepSC_algoA.root"); // Interface to TTree and TChain
  
  d.Define("nxtals", getNxtal, {"superCluster_pfClustersIndex", "pfCluster_nXtals"})\
    .Snapshot("recosimdumper/caloTree", "/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/TTbar_14TeV_TuneCP5_Pythia8/RawDumper_DeepSC_12_5_0/ttbar_Run3_DeepSC_algoA_withRechit_onlySC.root", {"nxtals"});
  
}
