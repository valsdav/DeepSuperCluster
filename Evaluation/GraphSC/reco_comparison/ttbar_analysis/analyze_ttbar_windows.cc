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
  return en/TMath::CosH(std::abs(eta));
}

using mytuple_t = std::tuple<std::vector<std::vector<double>>, std::vector<float>, std::vector<float>, std::vector<std::vector<double>>>;

mytuple_t getNxtal(RVec<float> cl_energy, RVec<float> cl_eta, RVec<float> cl_phi,RVec<int> cl_iz, RVec<double> nrechits){
  std::vector<std::vector<double>> result;
  std::vector<std::vector<double>> cl_et;
  std::vector<float> seed_ets;
  std::vector<float> seed_eta;
  result.reserve(100);
  cl_et.reserve(100);
  seed_ets.reserve(100);
  seed_eta.reserve(100);

  for (auto icl_seed=0; icl_seed < cl_eta.size(); icl_seed++){
    // std::cout << "seed: " <<  icl_seed << std::endl;
    auto seed_et = Et(cl_energy[icl_seed], cl_eta[icl_seed]);
    if ( seed_et < 1) continue;
    seed_ets.push_back(seed_et);
    seed_eta.push_back(cl_eta[icl_seed]);
    std::vector<double> rechit_per_cl;
    std::vector<double> et_per_cl;
    rechit_per_cl.reserve(30);
    et_per_cl.reserve(30);
    
    auto dim = get_dynamic_window(cl_eta[icl_seed]);
    // std::cout << "seed eta "<< cl_eta[icl_seed] << " seed et " << seed_et <<
      // "Window " << dim[0] << " " << dim[1] << " " << dim[2] << std::endl;

    for (auto icl=0; icl < cl_eta.size(); icl++){
      // std::cout << "\tcluster eta,phi,iz "<< cl_eta[icl] << " "<<cl_phi[icl] << " " <<
        // cl_iz[icl] << std::endl;
      if (in_window(cl_eta[icl_seed], cl_phi[icl_seed], cl_iz[icl_seed],
                    cl_eta[icl], cl_phi[icl], cl_iz[icl], dim) || icl_seed==icl){
        // std::cout << "\t\tInwindow" << std::endl;
        rechit_per_cl.push_back(nrechits[icl]);
        et_per_cl.push_back(Et(cl_energy[icl], cl_eta[icl]));
      }
    }
    result.push_back(rechit_per_cl);
    cl_et.push_back(et_per_cl);
  }
  return std::make_tuple(result, seed_ets, seed_eta, cl_et);
}



int main(){
  ROOT::EnableImplicitMT(8); // Tell ROOT you want to go parallel
  ROOT::RDataFrame d("recosimdumper/caloTree", "/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/TTbar_14TeV_TuneCP5_Pythia8/RawDumper_DeepSC_12_5_0/ttbar_Run3_DeepSC_algoA.root"); // Interface to TTree and TChain
  
  d.Define("data", getNxtal, {"pfCluster_energy","pfCluster_eta", "pfCluster_phi", "pfCluster_iz", "pfCluster_nXtals"}) \
    .Define("nxtals",  [](const mytuple_t &data) -> std::vector<std::vector<double>> { return std::get<0>(data); } , {"data"})\
    .Define("seed_et",[](const mytuple_t &data) -> std::vector<float> { return std::get<1>(data); } , {"data"}) \
op    .Define("seed_eta",[](const mytuple_t &data) -> std::vector<float> { return std::get<2>(data); } , {"data"}) \
    .Define("cls_et", [](const mytuple_t &data) -> std::vector<std::vector<double>> { return std::get<3>(data); } , {"data"}) \
    .Snapshot("recosimdumper/caloTree",
              "/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/TTbar_14TeV_TuneCP5_Pythia8/RawDumper_DeepSC_12_5_0/ttbar_Run3_DeepSC_algoA_withRechit_withClEt.root",
              {"nxtals", "seed_et", "seed_eta", "cls_et"});
  
}
