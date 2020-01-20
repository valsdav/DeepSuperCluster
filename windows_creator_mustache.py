from math import pi, sqrt, cosh
import random
import string
from operator import itemgetter, attrgetter

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

def transform_ieta(ieta):
    if ieta > 0:  return ieta +84
    elif ieta < 0: return ieta + 85

def iphi_distance(iphiseed, iphi, iz):
    if iz == 0:
        if abs(iphiseed-iphi)<= 180: return iphi-iphiseed
        if iphiseed < iphi:
            return iphi-iphiseed - 360
        else:
            return iphi - iphiseed + 360
    else :
        return iphi - iphiseed



def ieta_distance(ietaseed, ieta, iz):
    if iz == 0:
        return transform_ieta(ieta) - transform_ieta(ietaseed)
    else:
        return ieta-ietaseed

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
def cluster_in_window(window, clhits_eta, clhits_phi, clhits_iz):
    for eta, phi, iz in zip(clhits_eta, clhits_phi, clhits_iz):
        hit_in_wind, (etaw, phiw) = in_window(window["seed"][0],window["seed"][1],window["seed"][2],eta, phi, iz)
        #print((eta,phi,iz), (window["seed"][0],window["seed"][1],window["seed"][2]), etaw, phiw)
        if hit_in_wind:
            return True
    return False


def get_windows(event, window_eta, window_phi, nocalowNmax=0, assoc_strategy="sim_fraction_min1", debug=False):
    # Branches
    # Supercluster refs
    mustacheseed_pfcls = event.superCluster_seedIndex
    pfcl_in_mustache = event.superCluster_pfClustersIndex

    pfCluster_energy = event.pfCluster_energy
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
    pfcl_swissCross = event.pfCluster_swissCross
    pfcl_nxtals = event.pfCluster_nXtals
    

    # Load associations from dumper
    pfcluster_calo_map = getattr(event, "pfCluster_{}_MatchedIndex".format(assoc_strategy))
    calo_pfcluster_map = getattr(event, "caloParticle_pfCluster_{}_MatchedIndex".format(assoc_strategy))
   
    # map of windows, key=pfCluster seed index
    windows_map = {}
    clusters_event = []
    nocalowN = 0

    # Create windows looking at supercluster seeds
    for scindex, icl in enumerate(mustacheseed_pfcls):
        cl_iz = pfCluster_iz[icl]
        cl_eta = pfCluster_eta[icl]
        cl_phi = pfCluster_phi[icl]

        caloseed = pfcluster_calo_map[icl]
        if caloseed == -1:
            nocalowN+=1
            # Not creating too many windows of noise
            if nocalowN> nocalowNmax: continue
        # Let's create  new window:
        new_window = {
            "seed": (cl_eta, cl_phi, cl_iz),
            "calo" : caloseed,
            "mustache_index": scindex,
            "metadata": {
                "seed_eta": cl_eta,
                "seed_phi": cl_phi, 
                "seed_iz": cl_iz,
                "en_seed": pfCluster_energy[icl],
                "en_true": calo_simenergy[caloseed] if caloseed!=-1 else 0, 
                "is_calo": caloseed != -1
            }
        }
        
        # Create a unique index
        windex = "".join([ random.choice(string.ascii_lowercase) for _ in range(8)])
        new_window["metadata"]["index"] = windex
        # Save the window
        windows_map[windex] = new_window
        # isin, mask = fill_window_cluster(new_window, clxtals_ieta, clxtals_iphi, clxtals_iz, 
        #                     clxtals_energy, clxtals_rechitEnergy, pfcluster_calo_map[icl], fill_mask=True)
        # Save also seed cluster for cluster_masks
        clusters_event.append({
                "window_index": new_window["metadata"]["index"],
                "cluster_deta": 0.,
                "cluster_dphi": 0., 
                "cluster_iz" : cl_iz,
                "en_cluster": pfCluster_energy[icl],
                "is_seed": True,
                "in_scluster":  pfcluster_calo_map[icl] == new_window["calo"],
                # Save the information that is in the mustache selection
                "in_mustache": True,
                # Shower shape variables
                "f5_r9": pfcl_f5_r9[icl],
                "f5_sigmaIetaIeta" : pfcl_f5_sigmaIetaIeta[icl],
                "f5_sigmaIetaIphi" : pfcl_f5_sigmaIetaIphi[icl],
                "f5_sigmaIphiIphi" : pfcl_f5_sigmaIphiIphi[icl],
                "swissCross" : pfcl_swissCross[icl],
                "nxtals" : pfcl_nxtals[icl]
            })


           
    # Now that all the seeds are inside let's add the non seed
    for icl_noseed in range(len(pfCluster_eta)):
        if icl_noseed in mustacheseed_pfcls: continue  # it's a window

        cl_iz = pfCluster_iz[icl_noseed]
        cl_eta = pfCluster_eta[icl_noseed]
        cl_phi = pfCluster_phi[icl_noseed]

        # Fill all the windows
        for window in windows_map.values():
            isin, (etaw, phiw) = in_window(*window["seed"], cl_eta, cl_phi, cl_iz,
                                            window_eta[cl_iz], window_phi[cl_iz])
            if isin:
                cevent = {  
                    "window_index": window["metadata"]["index"],
                    "cluster_dphi": phiw,
                    "cluster_iz" : cl_iz,
                    "en_cluster": pfCluster_energy[icl_noseed],
                    "is_seed": False,
                    "in_scluster":  pfcluster_calo_map[icl_noseed] == window["calo"],
                    # Now check if the cluster is selected by mustache
                    "in_mustache" : icl_noseed in pfcl_in_mustache[window["mustache_index"]],
                    # Shower shape variables
                    "f5_r9": pfcl_f5_r9[icl_noseed],
                    "f5_sigmaIetaIeta" : pfcl_f5_sigmaIetaIeta[icl_noseed],
                    "f5_sigmaIetaIphi" : pfcl_f5_sigmaIetaIphi[icl_noseed],
                    "f5_sigmaIphiIphi" : pfcl_f5_sigmaIphiIphi[icl_noseed],
                    "swissCross" : pfcl_swissCross[icl_noseed],
                    "nxtals" : pfcl_nxtals[icl_noseed],
                }
                if window["metadata"]["seed_eta"] > 0:
                    cevent["cluster_deta"] = cl_eta - window["metadata"]["seed_eta"]
                else:
                    cevent["cluster_deta"] = window["metadata"]["seed_eta"] - cl_eta
                            
                clusters_event.append(cevent)

    ###############################
    #### Some metadata
    
    for window in windows_map.values():
        calo_seed = window["calo"]
        # Check the type of events
        # - Number of pfcluster associated, 
        # - deltaR of the farthest cluster
        # - Energy of the pfclusters
        if calo_seed != -1:
            # Get number of associated clusters
            assoc_clusters =  calo_pfcluster_map[calo_seed]
            max_en_pfcluster = max([pfCluster_energy[i] for i in assoc_clusters])
            max_dr = max( [ DeltaR(calo_simphi[calo_seed], calo_simeta[calo_seed], 
                            pfCluster_phi[i], pfCluster_eta[i]) for i in assoc_clusters])
            window["metadata"]["nclusters"] = len(assoc_clusters)
            window["metadata"]["max_en_cluster"] = max_en_pfcluster
            window["metadata"]["max_dr_cluster"] = max_dr

    
    # Save metadata in the cluster items
    for clw in clusters_event:
        clw.update(windows_map[clw["window_index"]]["metadata"])

    return windows_map, clusters_event