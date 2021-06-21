import json
import random
import os
import tensorflow as tf
import numpy as np
import collections
import time
import sys
import glob
import gzip
import argparse 
import ROOT as R

parser = argparse.ArgumentParser()
parser.add_argument("-n","--name", type=str, help="Job name", required=True)
parser.add_argument("-i","--inputfiles", type=str, help="inputfile", required=True)
parser.add_argument("-o","--outputdir", type=str, help="Outputdirectory",required=True)
parser.add_argument("-w","--weights", type=str, help="Weights",required=False)
parser.add_argument("-f","--flag", type=int, help="flag to add")
args = parser.parse_args()


def load_iter(files):
    for filename in files:
        with gzip.open(filename, "rt") as file:
            for line in file:
                content = line[line.rfind('{"window_index"'):]
                if content:
                    try:
                        data1 = json.loads(content)
                        yield data1
                    except Exception as  e:
                        print(e)
                        continue
                    
               

#https://stackoverflow.com/questions/47861084/how-to-store-numpy-arrays-as-tfrecord
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def _int64_features(values):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_features(values):
    """Returns an float32_list from a bool / enum / int / uint."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def _float_feature(value):
    """Returns an float32_list from a bool / enum / int / uint."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _tensor_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()]))


def make_example_window(window, weights=None):
    
    seed_features = ["seed_eta","seed_phi", "seed_ieta","seed_iphi", "seed_iz", 
                     "en_seed", "et_seed","en_seed_calib","et_seed_calib",
                    "seed_f5_r9","seed_f5_sigmaIetaIeta", "seed_f5_sigmaIetaIphi",
                    "seed_f5_sigmaIphiIphi","seed_f5_swissCross",
                    "seed_r9","seed_sigmaIetaIeta", "seed_sigmaIetaIphi",
                    "seed_sigmaIphiIphi","seed_swissCross",
                    "seed_nxtals","seed_etaWidth","seed_phiWidth",
                    ]

    seed_labels =   [ "is_seed_calo_matched", "is_seed_calo_seed", "is_seed_mustache_matched"]
    seed_metadata = [ "seed_score", "seed_simen_sig", "seed_simen_PU", "seed_PUfrac"]

    
    # features that can be used in the training
    window_features = [  "max_en_cluster","max_et_cluster","max_deta_cluster","max_dphi_cluster","max_den_cluster","max_det_cluster",
                         "min_en_cluster","min_et_cluster","min_deta_cluster","min_dphi_cluster","min_den_cluster","min_det_cluster",
                         "mean_en_cluster","mean_et_cluster","mean_deta_cluster","mean_dphi_cluster","mean_den_cluster","mean_det_cluster" ]
    # Metadata about the window like true energy, true calo position, useful info
    window_metadata = ["nVtx", "rho", "obsPU", "truePU",
                         "sim_true_eta", "sim_true_phi",  
                        "en_true_sim","et_true_sim", "en_true_gen", "et_true_gen",
                        "en_true_sim_good", "et_true_sim_good",
                        "sim_true_eta","sim_true_phi","gen_true_eta","gen_true_phi",
                        "en_mustache_raw", "et_mustache_raw","en_mustache_calib", "et_mustache_calib", "nclusters_insc",
                        "max_en_cluster_insc","max_deta_cluster_insc","max_dphi_cluster_insc",
                        "event_tot_simen_PU","wtot_simen_PU","wtot_simen_sig" ]

    cls_features = [    "en_cluster","et_cluster",
                        "cluster_eta", "cluster_phi", 
                        "cluster_ieta","cluster_iphi","cluster_iz",
                        "cluster_deta", "cluster_dphi",
                        "cluster_den_seed","cluster_det_seed",
                        "en_cluster_calib", "et_cluster_calib",
                        "cl_f5_r9", "cl_f5_sigmaIetaIeta", "cl_f5_sigmaIetaIphi",
                        "cl_f5_sigmaIphiIphi","cl_f5_swissCross",
                        "cl_r9", "cl_sigmaIetaIeta", "cl_sigmaIetaIphi",
                        "cl_sigmaIphiIphi","cl_swissCross",
                        "cl_nxtals", "cl_etaWidth","cl_phiWidth",
                    ]

    cls_labels = ["is_seed","is_calo_matched","is_calo_seed", "in_scluster","in_geom_mustache","in_mustache"]
    cls_metadata = [ "calo_score", "calo_simen_sig", "calo_simen_PU", "cluster_PUfrac","calo_nxtals_PU",
                    "noise_en","noise_en_uncal","noise_en_nofrac","noise_en_uncal_nofrac"]

    seed_f = np.array( [window[f] for f in seed_features],dtype='float32')
    seed_l = np.array( [window[f] for f in seed_labels],dtype='int')
    seed_m = np.array( [window[f] for f in seed_metadata],dtype='float32')
    window_f = np.array( [window[f] for f in window_features],dtype='float32')
    window_m = np.array( [window[f] for f in window_metadata],dtype='float32')
    seed_hits = np.array([ [r[0],r[1],r[2],r[4]] for r in  window['seed_hits']], dtype='float32')

    # Class division
    if not window['is_seed_calo_matched']:
        class_ = 0
    elif window['is_seed_calo_matched'] and not window["is_seed_calo_seed"]:
        class_ = 1
    elif window['is_seed_calo_matched'] and window["is_seed_calo_seed"]:
        class_ = 2

    weight = 1.0
    if weights != None:
        weight = weights.GetBinContent(weights.FindBin(window["et_seed"], window["ncls"]))

    #print(_int64_feature(window["ncls"])
    #Using short labels because they are repeated a lot of times
    context_features = {
        # Seed features (for training)
        's_f': _float_features(seed_f),
        # Seed labels
        's_l': _int64_features(seed_l),
        #Seed metadata
        's_m': _float_features(seed_m),
        # Seed hits
        's_h': _tensor_feature(seed_hits),
        # window features (for training)
        'w_f': _float_features(window_f),
        # window metadata and truth info
        'w_m': _float_features(window_m),
        # window class
        'w_cl' : _int64_feature(class_),
        # number of clusters
        'n_cl' : _int64_feature(window["ncls"]),
        # Weight
        'wi' :  _float_feature(weight)
    }
    # flag for flavour 
    if args.flag != None:
        if window["is_seed_calo_matched"]:
            context_features['f'] = _int64_feature(args.flag) 
        else:
            context_features['f'] = _int64_feature(0) 

    # Now clusters features as a list
    clusters_features = [ _float_features(np.array([ cl[feat] for feat in cls_features],dtype='float32'))  for cl in window["clusters"] ]
    clusters_metadata = [ _float_features(np.array([ cl[m] for m in cls_metadata],dtype='float32'))  for cl in window["clusters"] ]
    clusters_labels =   [ _int64_features(np.array([ cl[l] for l in cls_labels],dtype='int'))  for cl in window["clusters"] ]
    
    clusters_hits0 =     [ _float_features(np.array([r[0] for r in cl['cl_hits']],dtype="float32"))  for cl in window["clusters"] ]
    clusters_hits1 =     [ _float_features(np.array([r[1] for r in cl['cl_hits']],dtype="float32"))  for cl in window["clusters"] ]
    clusters_hits2 =     [ _float_features(np.array([r[2] for r in cl['cl_hits']],dtype="float32"))  for cl in window["clusters"] ]
    clusters_hits4 =     [ _float_features(np.array([r[4] for r in cl['cl_hits']],dtype="float32"))  for cl in window["clusters"] ]
    
    
    clusters_list = tf.train.FeatureLists(
        feature_list={
            "cl_f" : tf.train.FeatureList(feature=clusters_features),
            "cl_m" : tf.train.FeatureList(feature=clusters_metadata),
            "cl_l" : tf.train.FeatureList(feature=clusters_labels),
            "cl_h0" : tf.train.FeatureList(feature=clusters_hits0),
            "cl_h1" : tf.train.FeatureList(feature=clusters_hits1),
            "cl_h2" : tf.train.FeatureList(feature=clusters_hits2),
            "cl_h4" : tf.train.FeatureList(feature=clusters_hits4)
        }
    )

    # print(clusters_list)
    
    example = tf.train.SequenceExample(context=tf.train.Features(feature=context_features), 
                                       feature_lists=clusters_list)
    # print(example)
    return example, class_


if __name__ == "__main__":
    
    report_dt = 5
    outputdir =args.outputdir

    os.makedirs(outputdir+"/calo_matched", exist_ok=True)
    os.makedirs(outputdir+"/no_calo_matched", exist_ok=True)
    
    if "#_#" in args.inputfiles: 
        inputfiles = args.inputfiles.split("#_#")
    else:
        inputfiles = [args.inputfiles]

    print("Start reading files")
    it_files = load_iter(inputfiles)

    writers= {0: tf.io.TFRecordWriter(os.path.join(outputdir,"no_calo_matched","nocalomatch_" + args.name+".proto")),
              1: tf.io.TFRecordWriter(os.path.join(outputdir,"calo_matched", "calomatch_" + args.name+".proto"))}
    # for class 2 use the same as class1
    writers[2] = writers[1]

    t0 = time.time()
    counter = {0:0,1:0,2:0}

    weights = None
    if args.weights:
        f = R.TFile(args.weights,"READ")
        weights= f.Get("weight")
        weights.SetDirectory(0)
        f.Close()
    

    try:
        for i, row in enumerate(it_files):
            # print(row)
            example, class_ = make_example_window(row, weights)
            writers[class_].write(example.SerializeToString())
            counter[class_] += 1

            if time.time() - t0 > report_dt:
                print('processed %d' % i)
                t0 = time.time()
    except:
        pass

    for writer in writers.values():
        writer.close()    

    with open(os.path.join(outputdir, args.name + "_metadata.txt"), "w") as mf:
        for cl, count in counter.items():
            mf.write("{};{};{}\n".format(args.name, cl, count))
