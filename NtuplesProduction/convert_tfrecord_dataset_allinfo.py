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
                    except:
                        break
                else:
                    break
                    
               

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


def make_example_window(window):

    clusters_features = []

    seed_features = ["seed_eta","seed_phi", "seed_ieta","seed_iphi", "seed_iz", 
                     "en_seed", "et_seed","en_seed_calib","et_seed_calib",
                     "en_true","et_true",
                    "seed_f5_r9","seed_f5_sigmaIetaIeta", "seed_f5_sigmaIetaIphi",
                    "seed_f5_sigmaIphiIphi","seed_f5_swissCross",
                    "seed_r9","seed_sigmaIetaIeta", "seed_sigmaIetaIphi",
                    "seed_sigmaIphiIphi","seed_swissCross",
                    "seed_nxtals","seed_etaWidth","seed_phiWidth",
                    ]

    seed_metadata_int = [ "is_seed_calo_matched","is_seed_calo_seed","is_seed_mustached_matched"]
    seed_metadata_float = ["seed_score"]

    
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

    cls_metadata_int = ["is_seed","is_calo_matched","is_calo_seed",
                        "in_scluster","in_geom_mustache","in_mustache"]
    cls_metadata_float = ["calo_score"]

    seed_f = np.array( [window[f] for f in seed_features],dtype='float32')
    seed_m_i = np.array( [window[f] for f in seed_metadata_int],dtype='int')
    seed_m_f = np.array( [window[f] for f in seed_metadata_float],dtype='float32')
    seed_rechits = np.array([ [r[0],r[1],r[2],r[4]] for r in  window['seed_hits']], dtype='float32')

    clusters_features = np.transpose(np.array([ window["clusters"][feat] for feat in cls_features],dtype='float32'))
    clusters_m_f = np.transpose(np.array([ window["clusters"][feat] for feat in cls_metadata_float],dtype='float32')) 
    clusters_m_i = np.transpose(np.array([ window["clusters"][feat] for feat in cls_metadata_int],dtype='float32'))

    # Class division
    if not window['is_seed_calo_matched']:
        class_ = 0
    elif window['is_seed_calo_matched'] and not window["is_seed_calo_seed"]:
        class_ = 1
    elif window['is_seed_calo_matched'] and window["is_seed_calo_seed"]:
        class_ = 2

    #Using short labels because they are repeated a lot of times
    feature = {
        's_f': _bytes_feature(tf.io.serialize_tensor(seed_f)),
        's_m_i': _bytes_feature(tf.io.serialize_tensor(seed_m_i)),
        's_m_f': _bytes_feature(tf.io.serialize_tensor(seed_m_f)),
        'cl_f': _bytes_feature(tf.io.serialize_tensor(clusters_features)),
        'cl_m_f': _bytes_feature(tf.io.serialize_tensor(clusters_m_f)),
        'cl_m_i': _bytes_feature(tf.io.serialize_tensor(clusters_m_i)),
        # window class
        'w_cl' : _int64_feature(class_),
        # number of clusters
        'n_cl' : _int64_feature(len(window["clusters"]))
    }


    example = tf.train.Example(features=tf.train.Features(feature=feature))
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

    it_files = load_iter(inputfiles)

    writers= {0: tf.io.TFRecordWriter(os.path.join(outputdir,"no_calo_matched","nocalomatch_" + args.name+".proto")),
              1: tf.io.TFRecordWriter(os.path.join(outputdir,"calo_matched", "calomatch_" + args.name+".proto"))}
    # for class 2 use the same as class1
    writers[2] = writers[1]

    t0 = time.time()
    counter = {0:0,1:0,2:0}

    try:
        for i, row in enumerate(it_files):
            example, class_ = make_example_window(next(it_files))
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
