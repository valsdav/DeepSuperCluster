import ROOT  as R
import calo_association
import sys
from pprint import pprint
import pandas as pd
import math

file = "/eos/user/r/rdfexp/ecal/cluster/raw_files/FourElectronsGunPt1-100_pythia8_PU_Flat55To75_14TeV_112X_mcRun3_2021_realistic_v15_Dumper/cluster_job{}_noOOTPU_step2_output.root"
nfiles = int(sys.argv[1])
starting = int(sys.argv[2])

data= [ ] 

for i in range(nfiles):
    try:
        f = R.TFile.Open(file.format(starting+i))
        t = f.Get("recosimdumper/caloTree")
    except:
        print("problems with file")
        continue

    i = 0 
    for ev in t:
        cluster_calo_assoc, cluster_calo_assoc_score, sorted_calo_cluster_assoc, cluster_calo_PU_assoc = \
            calo_association.get_calo_association_withpu(ev.pfCluster_sim_fraction,  ev.caloParticle_isPU,ev.caloParticle_isOOTPU, 
                                            sort_calo_cl=True, debug=False, min_sim_fraction=1e-4)

        # df = {}
        # print("\nClusters - Caloparticle PU association")
        # for cl, calopuscores in cluster_calo_PU_assoc.items():
        #     print("Cluster: ", cl, " nXtals: ", ev.pfCluster_nXtals[cl], " raw En: ",  ev.pfCluster_rawEnergy[cl])
        #     for calo, score in calopuscores:
        #         simenergy_PU = score * ev.caloParticle_simEnergy[calo]
        #         print("\tCaloPU: {}  nSharedXtals: {:.0f}, simFraction:{:.4f}, simenergy_PU:{:.4f}".format(calo, ev.pfCluster_sim_nSharedXtals[cl][calo], score,simenergy_PU ))
        
        # print("\nClusters in signal caloparticle analysis")
        # for calo, clusters in sorted_calo_cluster_assoc.items():
        #     print("calo: ", calo)
        #     for cl, score in clusters:
        #         simenergy_true = score * ev.caloParticle_simEnergy[calo]
        #         simenergy_pu = sum( [  scorepu * ev.caloParticle_simEnergy[calopu]  for calopu,scorepu in cluster_calo_PU_assoc[cl]] )
        #         print("\tcluster: {}, simfraction: {:.4f}, simenergy_signal: {:.3f}, simenergy PU: {:.3f}".format(cl, score, simenergy_true, simenergy_pu) )
        cl_energy =ev.pfCluster_rawEnergy
        cl_eta =ev.pfCluster_eta
        cl_nxtals = ev.pfCluster_nXtals
        calo_simenergy = ev.caloParticle_simEnergy
        calo_simeta = ev.caloParticle_simEta
        #  print("\nClusters in signal caloparticle analysis")
        for calo, clusters in sorted_calo_cluster_assoc.items():
            # print(clusters)
            for cl, score in clusters:
                simenergy_signal = score * calo_simenergy[calo]
                n_calopu = len(cluster_calo_PU_assoc[cl])
                simenergy_pu = sum( [  scorepu * calo_simenergy[calopu]  for calopu,scorepu in cluster_calo_PU_assoc[cl]] )
                data.append({"en": cl_energy[cl],
                            "et": cl_energy[cl]/ math.cosh(cl_eta[cl]),
                            "xtals": cl_nxtals[cl],
                            "simfrac_sig": score, 
                            "simen_sig":simenergy_signal,
                            "simen_pu":simenergy_pu,
                            "simen_sig_frac":simenergy_signal/cl_energy[cl],
                            "simen_pu_frac":simenergy_pu/cl_energy[cl],
                            "eta": cl_eta[cl],
                            "n_calopu": n_calopu,
                            "calo_en": calo_simenergy[calo],
                            "calo_et": calo_simenergy[calo]/math.cosh(calo_simeta[calo]) })
                
        print(".", end="")


df = pd.DataFrame(data)
df.to_csv(sys.argv[3], sep=';', index=False)