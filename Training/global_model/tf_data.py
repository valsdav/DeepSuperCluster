import numpy as np
import glob
import multiprocessing
import os

import tensorflow as tf

####################################

def parse_single_window(element, read_hits=False):
    context_features = {
        's_f': tf.io.FixedLenFeature([24], tf.float32),
        's_l': tf.io.FixedLenFeature([3], tf.int64),
        's_m': tf.io.FixedLenFeature([8], tf.float32),
        # window class
        'w_cl' : tf.io.FixedLenFeature([], tf.int64),
        # number of clusters
        'n_cl' : tf.io.FixedLenFeature([], tf.int64),
        # flag (pdgid id)
        'f' :  tf.io.FixedLenFeature([], tf.int64)
    }
    
    clusters_features = {
        "cl_f" : tf.io.FixedLenSequenceFeature([22], dtype=tf.float32),
        "cl_m" : tf.io.FixedLenSequenceFeature([1], dtype=tf.float32),
        "cl_l" : tf.io.FixedLenSequenceFeature([6], dtype=tf.int64),
    }

    if read_hits:
        context_features['s_h'] = tf.io.FixedLenFeature([], tf.string)
        clusters_features["cl_h0"] = tf.io.RaggedFeature(dtype=tf.float32)
        clusters_features["cl_h1"] = tf.io.RaggedFeature(dtype=tf.float32)
        clusters_features["cl_h2"] = tf.io.RaggedFeature(dtype=tf.float32)
        clusters_features["cl_h4"] = tf.io.RaggedFeature(dtype=tf.float32)
    
    ex = tf.io.parse_single_sequence_example(element, context_features=context_features, sequence_features=clusters_features)
    
    if read_hits:
        seed_hits = tf.io.parse_tensor(ex[0]['s_h'], out_type=tf.float32)
        seed_hits.set_shape(tf.TensorShape((None, 4))) 
        ex[0]['s_h'] = seed_hits
        
        cluster_hits = tf.ragged.stack([ex[1]['cl_h0'], ex[1]['cl_h1'],ex[1]['cl_h2'],ex[1]['cl_h4']],axis=2)
        ex[1]['cl_h'] = cluster_hits 
        ex[1].pop("cl_h0")
        ex[1].pop("cl_h1")
        ex[1].pop("cl_h2")
        ex[1].pop("cl_h4")
    
    return ex

########################
def parse_windows_batch(elements, read_hits=False):
    context_features = {
        's_f': tf.io.FixedLenFeature([24], tf.float32),
        's_l': tf.io.FixedLenFeature([3], tf.int64),
        's_m': tf.io.FixedLenFeature([8], tf.float32),
        # window class
        'w_cl' : tf.io.FixedLenFeature([], tf.int64),
        # number of clusters
        'n_cl' : tf.io.FixedLenFeature([], tf.int64),
        # flag (pdgid id)
        'f' :  tf.io.FixedLenFeature([], tf.int64)
    }
    clusters_features = {
        "cl_f" : tf.io.FixedLenSequenceFeature([22], dtype=tf.float32),
        "cl_m" : tf.io.FixedLenSequenceFeature([1], dtype=tf.float32),
        "cl_l" : tf.io.FixedLenSequenceFeature([6], dtype=tf.int64),
    }
    if read_hits:
        # context_features['s_h'] = tf.io.FixedLenFeature([1], tf.string)
        clusters_features["cl_h0"] = tf.io.RaggedFeature(dtype=tf.float32)
        clusters_features["cl_h1"] = tf.io.RaggedFeature(dtype=tf.float32)
        clusters_features["cl_h2"] = tf.io.RaggedFeature(dtype=tf.float32)
        clusters_features["cl_h4"] = tf.io.RaggedFeature(dtype=tf.float32)

    ex = tf.io.parse_sequence_example(elements, context_features=context_features, sequence_features=clusters_features,name="input")
    
    if read_hits:
        # seed_hits = tf.io.parse_tensor(ex[0]['s_h'], out_type=tf.float32)
        # seed_hits.set_shape(tf.TensorShape((None, 4))) 
        # ex[0]['s_h'] = seed_hits 
        cluster_hits = tf.ragged.stack([ex[1]['cl_h0'], ex[1]['cl_h1'],ex[1]['cl_h2'],ex[1]['cl_h4']],axis=3)
        ex[1]['cl_h'] = cluster_hits 
        ex[1].pop("cl_h0")
        ex[1].pop("cl_h1")
        ex[1].pop("cl_h2")
        ex[1].pop("cl_h4")
    
    return ex

########################

def get_cluster_features_indexes(feats):
    '''
    Utils function to get the index of the requested features from the cluster full features list
    '''
    cl_feat = ["cluster_ieta","cluster_iphi","cluster_iz",
                "cluster_deta", "cluster_dphi",
                "en_cluster","et_cluster", 
                "en_cluster_calib", "et_cluster_calib",
                "cl_f5_r9", "cl_f5_sigmaIetaIeta", "cl_f5_sigmaIetaIphi",
                "cl_f5_sigmaIphiIphi","cl_f5_swissCross",
                "cl_r9", "cl_sigmaIetaIeta", "cl_sigmaIetaIphi",
                "cl_sigmaIphiIphi","cl_swissCross",
                "cl_nxtals", "cl_etaWidth","cl_phiWidth"]
    output = [] 
    for f in feats: 
        if f in cl_feat:
            output.append(feats.index(f))
        else:
            print("Missing branch! ", f)
    return output

####################################################
# Function to prepare tensors for training 

def get_cluster_features_and_hits(feat_index): 
    def process(*kargs):
        cl_f = kargs[1]['cl_f']
        cl_l = kargs[1]['cl_l']
        cl_X = tf.gather(cl_f, indices=feat_index,axis=2)
        cl_hits = kargs[1]['cl_h']
        is_seed = tf.gather(cl_l,indices=[0],axis=2)
        in_sc = tf.gather(cl_l,indices=[3],axis=2)
        cl_X = tf.concat([ cl_X,tf.cast(is_seed, tf.float32),], axis=-1)
        return cl_X, cl_hits, is_seed, in_sc,  kargs[2]["cl_f"]
    return process


##############################################
  
def load_dataset_batch(path, batch_size, options):
    '''
    options = { "read_hits" }
    '''
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(path))
    dataset = dataset.batch(batch_size).map(
                lambda el: parse_windows_batch(el, options['read_hits']),
                num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
    return dataset


def cluster_features_and_hits(dataset, features):
    return dataset.map( get_cluster_features_and_hits(get_cluster_features_indexes(features)),
           num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False )


def get(dataset):
    el = next(iter(dataset.take(1)))
    return el

