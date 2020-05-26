from collections import defaultdict
from operator import itemgetter


def get_calo_association(clusters_scores, clusters_eta, debug=False):
    '''
    pfCluster_scores is a list. For each pfCluster, there is a list of scores for each calo. 
    Each cluster is associated with the calo with the highest score. 
    Each calo is assocciated with the list of cluster for which it has the highest score.
    '''
    calo_cluster_assoc_scores = defaultdict(list)
    calo_cluster_assoc_map = defaultdict(list)
    cluster_calo_assoc = {}
    cluster_calo_assoc_score = {}

    for (clid, scores), cl_eta in zip(enumerate(clusters_scores), clusters_eta):
        # order caloparticle by fraction
        caloids =  list(sorted( enumerate(scores), key=itemgetter(1), reverse=True))
    
        # Associate to che cluster the list of caloparticles ordered by fraction 
        if caloids and caloids[0][1]> 1e-5:
            cluster_calo_assoc[clid] = caloids[0][0]
            cluster_calo_assoc_score[clid] = caloids[0][1]
            # save for the calocluster in the caloparticle if it is the one with more fraction
            # This is necessary in case one caloparticle is linked with more than one cluster
            calo_cluster_assoc_map[caloids[0][0]].append(clid)
            if debug:  calo_cluster_assoc_scores[caloids[0][0]].append((clid, caloids[0][1]))
        else:
            # Save -1 index for caloparticle absent
            cluster_calo_assoc[clid] = -1
            cluster_calo_assoc_score[clid] = -1
        
    # Now sort the clusters associated to a caloparticle with the fraction 
    if debug:
        sorted_calo_cluster_assoc = {}
        for caloid, clinfo in calo_cluster_assoc_scores.items():
            sorted_calo_cluster_assoc[caloid] = sorted(clinfo, key=itemgetter(1), reverse=True)
    
        for calo, cls in sorted_calo_cluster_assoc.items():
            print("Calo: ", calo)
            for cl in cls:
                print("\tCluster: {} , fraction: {}".format(cl[0], cl[1]))

    return cluster_calo_assoc, cluster_calo_assoc_score, calo_cluster_assoc_map