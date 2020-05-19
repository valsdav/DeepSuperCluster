from __future__ import print_function
from math import pi, sqrt, cosh
import random
import string
from collections import OrderedDict, defaultdict
from operator import itemgetter, attrgetter
import calo_association
import random

'''
This script extracts the windows and associated clusters from events
coming from RecoSimDumper
'''


def DeltaR(phi1, eta1, phi2, eta2):
        dphi = phi1 - phi2
        if dphi > pi: dphi -= 2*pi
        if dphi < -pi: dphi += 2*pi
        deta = eta1 - eta2
        deltaR = (deta*deta) + (dphi*dphi)
        return sqrt(deltaR)

def DeltaPhi(phi1, phi2):
    dphi = phi1 - phi2
    if dphi > pi: dphi -= 2*pi
    if dphi < -pi: dphi += 2*pi
    return dphi


def dynamic_window(eta,iz):
    if iz == 0:
        return 0.2, 0.6  
    else:
        if abs(eta)< 2.25:
            deta = 0.2
            x = abs(eta)
            dphi =   0.2197*(x**2) - 1.342*x + 2.195
            return deta, dphi 
        elif abs(eta) >= 2.25:
            deta = 0.2
            x = 2.25
            dphi =  0.2197*(x**2) - 1.342*x + 2.195
            return deta, dphi 

# def dynamic_window(eta):
#     return 0.3, 0.7


# Check if a xtal is in the window
def in_window(seed_eta, seed_phi, seed_iz, eta, phi, iz, window_eta, window_phi):
    if seed_iz != iz: return False, (-1,-1)
    etaw = seed_eta - eta
    phiw = DeltaPhi(seed_phi, phi)
    if abs(etaw) <= window_eta and abs(phiw) <= window_phi: 
        return True, (etaw, phiw)
    else:
        return False,(-1,-1)

# Check if cluster has an hit in the window
# def cluster_in_window(window, clhits_eta, clhits_phi, clhits_iz):
#     for eta, phi, iz in zip(clhits_eta, clhits_phi, clhits_iz):
#         hit_in_wind, (etaw, phiw) = in_window(window["seed"][0],window["seed"][1],window["seed"][2],eta, phi, iz)
#         #print((eta,phi,iz), (window["seed"][0],window["seed"][1],window["seed"][2]), etaw, phiw)
#         if hit_in_wind:
#             return True
#     return False


def get_windows(event, assoc_strategy,  nocalowNmax=0, min_et_seed=1, debug=False):
    # Branches
    pfCluster_energy = event.pfCluster_energy
    pfCluster_rawEnergy = event.pfCluster_rawEnergy
    pfCluster_eta = event.pfCluster_eta
    pfCluster_phi = event.pfCluster_phi
    pfCluster_iz = event.pfCluster_iz
    calo_simenergy = event.caloParticle_simEnergy
    calo_simeta = event.caloParticle_simEta
    calo_simphi = event.caloParticle_simPhi
    pfcl_f5_r9 = event.pfCluster_full5x5_r9
    pfcl_f5_sigmaIetaIeta = event.pfCluster_full5x5_sigmaIetaIeta
    pfcl_f5_sigmaIetaIphi = event.pfCluster_full5x5_sigmaIetaIphi
    pfcl_f5_sigmaIphiIphi = event.pfCluster_full5x5_sigmaIphiIphi
    pfcl_f5_swissCross = event.pfCluster_full5x5_swissCross
    pfcl_nxtals = event.pfCluster_nXtals
    pfcl_etaWidth = event.pfCluster_etaWidth
    pfcl_phiWidth = event.pfCluster_phiWidth

    clusters_scores = getattr(event, "pfCluster_"+assoc_strategy)
    # Get Association between pfcluster and calo
    pfcluster_calo_map, calo_pfcluster_map = calo_association.get_calo_association(clusters_scores, pfCluster_eta, False)

    if debug:
        print(">>>Cluster_calo map")
        for cluster, calo in pfcluster_calo_map.items():
            if calo == -1: continue
            print("cl: {} | calo: {}".format(cluster,calo))
        print("\n>>>Cluster_calo map")
        for calo, clusters in calo_pfcluster_map.items():
            print("calo: {} | clusters: {}".format(calo, clusters))
    
    #Mustache info
    mustacheseed_pfcls = [s for s in event.superCluster_seedIndex]
    pfcl_in_mustache = event.superCluster_pfClustersIndex
   
    # map of windows, key=pfCluster seed index
    windows_map = OrderedDict()
    windows_nocalomatched = []
    calo_windows = defaultdict(list)
    clusters_event = []
    seed_clusters = []
    

    # 1) Look for highest energy cluster (corrected energy)
    clenergies_ordered = sorted([ (ic , et) for ic, et in enumerate(
                             map ( lambda k: k[0]/cosh(k[1]), zip( pfCluster_energy, pfCluster_eta) )
                            )], key=itemgetter(1), reverse=True)


    # Now iterate over clusters in order of energies
    for icl, clenergy_T in clenergies_ordered:
        #print(icl, clenergy_T)

        # No seeds with Et< 1 GeV
        if clenergy_T < min_et_seed: continue

        cl_eta = pfCluster_eta[icl]
        cl_phi = pfCluster_phi[icl]
        cl_iz =  pfCluster_iz[icl]

        is_in_window = False
        # Check if it is already in one windows
        for window in windows_map.values():
            is_in_this_window, (etaw, phiw) = in_window(*window["seed"], cl_eta, cl_phi, cl_iz, 
                                                 *dynamic_window(window["seed"][0], window["seed"][2])) 
            if is_in_this_window:
                is_in_window = True
                break

        # If is not already in some window 
        if not is_in_window: 
            # Check the associated calo to the seed
            caloseed = pfcluster_calo_map[icl]  # -1 if not associated
            # Save the cluster in the list of associated clusters
            seed_clusters.append(icl)
            # Check if it is a mustache seed
            if icl in mustacheseed_pfcls:
                mustache_seed_index = mustacheseed_pfcls.index(icl)
            else:
                mustache_seed_index = -1

            # Create a unique index
            windex = "".join([ random.choice(string.ascii_lowercase) for _ in range(9)])
            # Let's create  new window:
            new_window = {
                "window_index": windex,
                "seed": (cl_eta, cl_phi, cl_iz),
                "calo" : caloseed,
                "metadata": {
                    "is_calo_matched": caloseed != -1,
                    "calo_seed_index": caloseed,
                    "is_mustached_matched": mustache_seed_index != -1,
                    "mustache_seed_index": mustache_seed_index,

                    "seed_eta": cl_eta,
                    "seed_phi": cl_phi, 
                    "seed_iz": cl_iz,

                    "en_seed": pfCluster_rawEnergy[icl],
                    "et_seed": pfCluster_rawEnergy[icl] / cosh(cl_eta),
                    "en_seed_calib": pfCluster_energy[icl],
                    "et_seed_calib": pfCluster_energy[icl] / cosh(cl_eta),
                    "en_true": calo_simenergy[caloseed] if caloseed!=-1 else 0, 
                    "et_true": calo_simenergy[caloseed]/cosh(calo_simeta[caloseed]) if caloseed!=-1 else 0, 

                    "seed_f5_r9": pfcl_f5_r9[icl],
                    "seed_f5_sigmaIetaIeta" : pfcl_f5_sigmaIetaIeta[icl],
                    "seed_f5_sigmaIetaIphi" : pfcl_f5_sigmaIetaIphi[icl],
                    "seed_f5_sigmaIphiIphi" : pfcl_f5_sigmaIphiIphi[icl],
                    "seed_f5_swissCross" : pfcl_f5_swissCross[icl],
                    "seed_etaWidth" : pfcl_etaWidth[icl],
                    "seed_phiWidth" : pfcl_phiWidth[icl],
                    "seed_nxtals" : pfcl_nxtals[icl],
                }
            }
            # Save the window
            windows_map[windex] = new_window
            if caloseed == -1:  
                windows_nocalomatched.append(windex)
            else:
                # Save also the window index
                calo_windows[caloseed].append(windex)
           
            # Save also seed cluster for cluster_masks
            clusters_event.append({
                    "window_index": windex,
                    "is_seed": True,
                    "in_scluster":  new_window["calo"] != -1,
                    "in_mustache" :  new_window["metadata"]["mustache_seed_index"] != -1,

                    "cluster_deta": 0.,
                    "cluster_dphi": 0., 
                    "cluster_iz" : cl_iz,
                    "en_cluster": pfCluster_rawEnergy[icl],
                    "et_cluster": pfCluster_rawEnergy[icl] / cosh(cl_eta),
                    "en_cluster_calib": pfCluster_energy[icl],
                    "et_cluster_calib": pfCluster_energy[icl] / cosh(cl_eta),
                    # Shower shape variables
                    "cl_f5_r9": pfcl_f5_r9[icl],
                    "cl_f5_sigmaIetaIeta" : pfcl_f5_sigmaIetaIeta[icl],
                    "cl_f5_sigmaIetaIphi" : pfcl_f5_sigmaIetaIphi[icl],
                    "cl_f5_sigmaIphiIphi" : pfcl_f5_sigmaIphiIphi[icl],
                    "cl_f5_swissCross" : pfcl_f5_swissCross[icl],
                    "cl_etaWidth" : pfcl_etaWidth[icl],
                    "cl_phiWidth" : pfcl_phiWidth[icl],
                    "cl_nxtals" : pfcl_nxtals[icl],
                })

    if debug:
        print("ALL windows")
        print("N windows:", len(windows_map))
        calo_match = []
        for iw,window in enumerate(windows_map.values()):
            m = window["metadata"]
            if m["is_calo_matched"]: calo_match.append(iw)
            print(iw, ") Window: ", window["window_index"], "  Calo Matched: ",  m["is_calo_matched"])
            print("\t Seed: Eta:{:.3f}, Phi:{:.3f}, Iz: {:.3f}, En:{:.3f}".format( m["seed_eta"], m["seed_phi"],m["seed_iz"], m["en_seed"]))
            print("\t Calo: Index:{}, Eta:{:.3f}, Phi:{:.3f}, En:{:.3f}".format(m["calo_seed_index"],
                                    calo_simeta[m["calo_seed_index"]],calo_simphi[m["calo_seed_index"]], m["en_true"]))
        print(">>> TOT windows calomatched: ", len(calo_match), calo_match)

           
    # Now that all the seeds are inside let's add the non seed
    for icl_noseed, clenergy_T in clenergies_ordered:
        # exclude seed clusters
        if icl_noseed in seed_clusters: continue

        cl_iz = pfCluster_iz[icl_noseed]
        cl_eta = pfCluster_eta[icl_noseed]
        cl_phi = pfCluster_phi[icl_noseed]
    
        # Fill all the windows
        for window in windows_map.values():
            isin, (etaw, phiw) = in_window(*window["seed"], cl_eta, cl_phi, cl_iz,
                                             *dynamic_window(window["seed"][0], window["seed"][2]))
            if isin:

                # If the window is not associated to a calo then in_scluster is always false for the cluster
                if window["calo"] == -1 :   
                    in_scluster = False
                else: 
                    in_scluster = pfcluster_calo_map[icl_noseed] == window["calo"]
                # check if the cluster is inside the same mustache
                if window["metadata"]["mustache_seed_index"] != -1:
                    in_mustache = icl_noseed in pfcl_in_mustache[window["metadata"]["mustache_seed_index"]]
                else:
                    in_mustache = False
               
                cevent = {  
                    "window_index": window["window_index"],
                    "is_seed": False,
                    "in_scluster": in_scluster,
                    "in_mustache" : in_mustache,

                    "cluster_dphi":phiw ,
                    "cluster_iz" : cl_iz,
                    "en_cluster": pfCluster_rawEnergy[icl_noseed],
                    "et_cluster": pfCluster_rawEnergy[icl_noseed] / cosh(cl_eta),
                    "en_cluster_calib": pfCluster_energy[icl_noseed],
                    "et_cluster_calib": pfCluster_energy[icl_noseed] /cosh(cl_eta),
                    
                    # Shower shape variables
                    "cl_f5_r9": pfcl_f5_r9[icl_noseed],
                    "cl_f5_sigmaIetaIeta" : pfcl_f5_sigmaIetaIeta[icl_noseed],
                    "cl_f5_sigmaIetaIphi" : pfcl_f5_sigmaIetaIphi[icl_noseed],
                    "cl_f5_sigmaIphiIphi" : pfcl_f5_sigmaIphiIphi[icl_noseed],
                    "cl_f5_swissCross" : pfcl_f5_swissCross[icl_noseed],
                    "cl_etaWidth" : pfcl_etaWidth[icl_noseed],
                    "cl_phiWidth" : pfcl_phiWidth[icl_noseed],
                    "cl_nxtals" : pfcl_nxtals[icl_noseed]
                }
                if window["metadata"]["seed_eta"] > 0:
                    cevent["cluster_deta"] = cl_eta - window["metadata"]["seed_eta"]
                else:
                    cevent["cluster_deta"] = window["metadata"]["seed_eta"] - cl_eta
                
                clusters_event.append(cevent)
                # Save the cluster only in the first window encountered by Et
                break


    ###############################
    #### Some metadata
    
    # for window in windows_map.values():
    #     calo_seed = window["calo"]
    #     # Check the type of events
    #     # - Number of pfcluster associated, 
    #     # - deltaR of the farthest cluster
    #     # - Energy of the pfclusters
    #     if calo_seed != -1:
    #         # Get number of associated clusters
    #         assoc_clusters =  calo_pfcluster_map[calo_seed]
    #         max_en_pfcluster = max([pfCluster_energy[i] for i in assoc_clusters])
    #         max_dr = max( [ DeltaR(calo_simphi[calo_seed], calo_simeta[calo_seed], 
    #                         pfCluster_phi[i], pfCluster_eta[i]) for i in assoc_clusters])
    #         window["metadata"]["nclusters"] = len(assoc_clusters)
    #         window["metadata"]["max_en_cluster"] = max_en_pfcluster
    #         window["metadata"]["max_dr_cluster"] = max_dr

    
    # Save metadata in the cluster items
    for clw in clusters_event:
        clw.update(windows_map[clw["window_index"]]["metadata"])

    # Check if there are more than one windows associated with the same caloparticle
    windows_calomatched = []
    for calo, ws in calo_windows.items():
        # Take only the first windows, by construction the most energetic one
        windows_calomatched.append(ws[0])
        # if len(ws)>1:
        #     print("A lot of windows!")
        #     for w in ws:
        #         print("Windex: {}, seed et: {}, calo en: {}, calo eta: {}".format(w, 
        #                     windows_map[w]["metadata"]["et_seed"], calo_simenergy[calo], calo_simeta[calo]))


    ## Now save only a number of window nocalowindow extracting by random
    if len(windows_nocalomatched)> nocalowNmax:
        windows_to_keep_index = windows_calomatched + random.sample(windows_nocalomatched, nocalowNmax)
    else:
        windows_to_keep_index = windows_calomatched + windows_nocalomatched

    if debug: print("Windows to keep: ", windows_to_keep_index)
    
    windows_to_keep = list(filter(lambda w: w["window_index"] in windows_to_keep_index, windows_map.values()))
        
    clusters_events_final = list(filter(lambda cl: cl["window_index"] in windows_to_keep_index, clusters_event))

    return windows_to_keep, clusters_events_final
