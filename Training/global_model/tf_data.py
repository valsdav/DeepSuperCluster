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


####################################################
# Function to prepare tensors for training 

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
            output.append(cl_feat.index(f))
        else:
            print("Missing branch! ", f)
    return output


def get_cluster_features_and_hits(feat_index): 
    def process(*kargs):
        cl_f = kargs[1]['cl_f']
        cl_l = kargs[1]['cl_l']
        cl_X = tf.gather(cl_f, indices=feat_index,axis=-1)
        cl_hits = kargs[1]['cl_h']
        is_seed = tf.gather(cl_l,indices=[0],axis=-1)
        in_sc = tf.gather(cl_l,indices=[3],axis=-1)
        cl_X = tf.concat([ cl_X,tf.cast(is_seed, tf.float32),], axis=-1)
        n_cl = kargs[0]["n_cl"]
        return cl_X, cl_hits, is_seed, in_sc,n_cl
    return process


def cluster_features_and_hits(dataset, features):
    return dataset.map( get_cluster_features_and_hits(get_cluster_features_indexes(features)),
           num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False )


##############################################
# Loading functions
  
def load_dataset_batch(path, batch_size, options):
    '''
    options = { "read_hits" }
    '''
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(path))
    dataset = dataset.batch(batch_size).map(
                lambda el: parse_windows_batch(el, options['read_hits']),
                num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
    return dataset

def load_dataset_single(path, options):
    '''
    options = { "read_hits" }
    '''
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(path))
    dataset = dataset.map(
                lambda el: parse_single_window(el, options['read_hits']),
                num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
    return dataset


def load_balanced_dataset_batch(data_paths, features, batch_size, weights=None):
    datasets = {}
    for n, p in data_paths.items():
        df = load_dataset_single(p, options={"read_hits":True})
        df = cluster_features_and_hits(df, features)
        datasets[n] = df
    if weights:
        total_ds = tf.data.experimental.sample_from_datasets(list(datasets.values()), weights=weights)
    else:
        total_ds = tf.data.experimental.sample_from_datasets(list(datasets.values()), weights=[1/len(datasets)]*len(datasets))
    # Now we can shuffle and batch
    def batch_features(cl_X, cl_hits, is_seed, in_sc, ncls):
        '''This function is used to create padded batches together for dense features and ragged ones'''
        return tf.data.Dataset.zip((cl_X.padded_batch(batch_size), 
                                    cl_hits.batch(batch_size), 
                                is_seed.padded_batch(batch_size), 
                                in_sc.padded_batch(batch_size),
                                ncls.padded_batch(batch_size)))
    #total_ds = total_ds.shuffle(1000, reshuffle_each_iteration=True)
    total_ds_batched = total_ds.window(batch_size).flat_map(batch_features)
    return total_ds_batched



######################### 
#Utils for debugging

def get(dataset):
    el = next(iter(dataset.take(1)))
    return el

