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

parser = argparse.ArgumentParser()
parser.add_argument("-n","--name", type=str, help="Job name", required=True)
parser.add_argument("-i","--inputfiles", type=str, help="inputfile", required=True)
parser.add_argument("-o","--outputdir", type=str, help="Outputdirectory",required=True)
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
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    """Returns an float32_list from a bool / enum / int / uint."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _tensor_feature(value):
    return _bytes_feature(tf.io.serialize_tensor(value))


def make_example_window(window):

    seed_features = ["seed_eta","seed_phi", "seed_ieta","seed_iphi", "seed_iz", 
                     "en_seed", "et_seed","en_seed_calib","et_seed_calib",
                     "en_true","et_true",
                    "seed_f5_r9","seed_f5_sigmaIetaIeta", "seed_f5_sigmaIetaIphi",
                    "seed_f5_sigmaIphiIphi","seed_f5_swissCross",
                    "seed_r9","seed_sigmaIetaIeta", "seed_sigmaIetaIphi",
                    "seed_sigmaIphiIphi","seed_swissCross",
                    "seed_nxtals","seed_etaWidth","seed_phiWidth",
                    ]

    seed_labels = [ "is_seed_calo_matched","is_seed_calo_seed","is_seed_mustach_matched"]
    seed_metadata = ["nclusters_insc","max_en_cluster_insc","max_deta_cluster_insc",
                        "max_dphi_cluster_insc", "max_en_cluster","max_deta_cluster","max_dphi_cluster","seed_score" ]

    cls_features = [  "cluster_ieta","cluster_iphi","cluster_iz",
                     "cluster_deta", "cluster_dphi",
                     "en_cluster","et_cluster", 
                     "en_cluster_calib", "et_cluster_calib",
                    "cl_f5_r9", "cl_f5_sigmaIetaIeta", "cl_f5_sigmaIetaIphi",
                    "cl_f5_sigmaIphiIphi","cl_f5_swissCross",
                    "cl_r9", "cl_sigmaIetaIeta", "cl_sigmaIetaIphi",
                    "cl_sigmaIphiIphi","cl_swissCross",
                    "cl_nxtals", "cl_etaWidth","cl_phiWidth",
                    ]

    cls_labels = ["is_seed","is_calo_matched","is_calo_seed", "in_scluster","in_geom_mustache","in_mustache"]
    cls_metadata = [ "calo_score" ]

    seed_f = np.array( [window[f] for f in seed_features],dtype='float32')
    seed_l = np.array( [window[f] for f in seed_labels],dtype='int')
    seed_m = np.array( [window[f] for f in seed_metadata],dtype='float32')
    seed_hits = np.array([ [r[0],r[1],r[2],r[4]] for r in  window['seed_hits']], dtype='float32')

    # Class division
    if not window['is_seed_calo_matched']:
        class_ = 0
    elif window['is_seed_calo_matched'] and not window["is_seed_calo_seed"]:
        class_ = 1
    elif window['is_seed_calo_matched'] and window["is_seed_calo_seed"]:
        class_ = 2

    #Using short labels because they are repeated a lot of times
    context_features = {
        's_f': _tensor_feature(seed_f),
        's_l': _tensor_feature(seed_l),
        's_m': _tensor_feature(seed_m),
        's_h': _tensor_feature(seed_hits),
        # window class
        'w_cl' : _int64_feature(class_),
        # number of clusters
        'n_cl' : _int64_feature(window["ncls"]),
    }
    # flag for flavour or other info
    if args.flag != None:
        context_features['f'] = _int64_feature(args.flag) 


    # Now clusters features as a list
    clusters_features = [ _tensor_feature(np.array([ cl[feat] for feat in cls_features],dtype='float32'))  for cl in window["clusters"] ]
    clusters_metadata = [ _tensor_feature(np.array([ cl[m] for m in cls_metadata],dtype='float32'))  for cl in window["clusters"] ]
    clusters_labels =   [ _tensor_feature(np.array([ cl[l] for l in cls_labels],dtype='int'))  for cl in window["clusters"] ]
    clusters_hits =     [ _tensor_feature(np.array([[r[0],r[1],r[2],r[4]] for r in  cl['cl_hits']],dtype="float32"))  for cl in window["clusters"] ]

    clusters_list = tf.train.FeatureLists(
        feature_list={
            "cl_f" : tf.train.FeatureList(feature=clusters_features),
            "cl_m" : tf.train.FeatureList(feature=clusters_metadata),
            "cl_l" : tf.train.FeatureList(feature=clusters_labels),
            "cl_h" : tf.train.FeatureList(feature=clusters_hits)
        }
    )
    
    example = tf.train.SequenceExample(context=tf.train.Features(feature=context_features), 
                                       feature_lists=clusters_list)

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

    try:
        for i, row in enumerate(it_files):
            example, class_ = make_example_window(row)
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
