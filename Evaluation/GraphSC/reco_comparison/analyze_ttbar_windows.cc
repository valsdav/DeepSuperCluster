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
#include <array>
#include <algorithm>
#include "TString.h"
#include "TMath.h"
#include "TH1D.h"

using namespace std;
using namespace ROOT;
using namespace ROOT::VecOps;
using namespace ROOT::RDF;


float deltaPhi(float phi1,float  phi2){
  auto dphi = phi1-phi2;
  if (dphi > TMath::Pi()) dphi -= 2*TMath::Pi();
  if (dphi < -TMath::Pi()) dphi += 2*TMath::Pi();
  return dphi;
}


float deltaR(float phi1,float  eta1,float  phi2,float  eta2){
  auto dphi = deltaPhi(phi1, phi2);
  auto deta = eta1 - eta2;
  auto deltaR = (deta*deta) + (dphi*dphi);
  return TMath::Sqrt(deltaR);
}

array<float, 3> get_dynamic_window(float eta){
  auto aeta = std::abs(eta);
  array<float, 3> dim;
  if (aeta <= 1.5){
    dim[0] = (0.1/1.5)*aeta + 0.1;
    dim[1] = -0.1;
  }else{
    dim[0] = (0.1/1.5)*(aeta-1.5) + 0.2;
    dim[1] = (-0.1/1.5)*(aeta-1.5) -0.1;
  }
  dim[2]= 0.7 + (-0.1/3)*aeta;
  return dim;
}

bool in_window(float seed_eta, float seed_phi, float seed_iz,
               float eta, float phi, int iz,
               array<float,3> dim){
  if (seed_iz != iz) return false;
  auto etaw = eta-seed_eta;
  if (seed_eta < 0) etaw = -etaw;
  auto phiw = std::abs(deltaPhi(seed_phi, phi));
  auto [wup,wdown,wphi] = dim;
  if (etaw>=wdown && etaw<=wup && phiw <= wphi ) return true;
  else return false;
                     
}

float Et(float en, float eta){
  return en/TMath::CosH(eta);
}

auto getNxtal(RVec<float> cl_energy, RVec<float> cl_eta, RVec<float> cl_phi,RVec<int> cl_iz, RVec<double> nrechits){
  std::vector<std::vector<double>> result;
  result.reserve(40);

  for (auto icl_seed=0; icl_seed < cl_eta.size(); icl_seed++){
    if (Et(cl_energy[icl_seed], cl_eta[icl_seed]) < 1) continue;
    std::vector<double> rec_per_seed;
    rec_per_seed.reserve(10);
    auto dim = get_dynamic_window(cl_eta[icl_seed]);

    for (auto icl=0; icl < cl_eta.size(); icl++){
      if (in_window(cl_eta[icl_seed], cl_phi[icl_seed], cl_iz[icl_seed],
                    cl_eta[icl], cl_phi[icl], cl_iz[icl], dim) || icl_seed==icl){
        rec_per_seed.push_back(nrechits[icl]);
      }
    }
    result.push_back(rec_per_seed);
  }
  return result;
}


int main(){
  ROOT::EnableImplicitMT(6); // Tell ROOT you want to go parallel
  ROOT::RDataFrame d("recosimdumper/caloTree", "/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/TTbar_14TeV_TuneCP5_Pythia8/RawDumper_DeepSC_12_5_0/ttbar_Run3_DeepSC_algoA.root"); // Interface to TTree and TChain
  
  d.Define("nxtals", getNxtal, {"pfCluster_energy","pfCluster_eta", "pfCluster_phi", "pfCluster_iz", "pfCluster_nXtals"}) \
    .Snapshot("recosimdumper/caloTree", "/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/TTbar_14TeV_TuneCP5_Pythia8/RawDumper_DeepSC_12_5_0/ttbar_Run3_DeepSC_algoA_withRechit.root", {"nxtals"});
  
}
