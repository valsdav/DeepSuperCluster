import numpy as np
import glob
import multiprocessing
import os
import numpy as np
import tensorflow as tf

####################################

def parse_single_window(element, read_hits=False, read_metadata=False):
    context_features = {
        's_f': tf.io.FixedLenFeature([22], tf.float32),
        's_l': tf.io.FixedLenFeature([3], tf.int64),
        #Window features
        'w_f': tf.io.FixedLenFeature([18], tf.float32),
        # window class
        'w_cl' : tf.io.FixedLenFeature([], tf.int64),
        # number of clusters
        'n_cl' : tf.io.FixedLenFeature([], tf.int64),
        # flag (pdgid id)
        'f' :  tf.io.FixedLenFeature([], tf.int64)
    }
    
    clusters_features = {
        "cl_f" : tf.io.FixedLenSequenceFeature([26], dtype=tf.float32),
        "cl_l" : tf.io.FixedLenSequenceFeature([6], dtype=tf.int64),
    }

    if read_metadata:
        # seed metadata
        context_features["s_m"] = tf.io.FixedLenFeature([5], tf.float32) 
        # window metadata
        context_features["w_m"] = tf.io.FixedLenFeature([22], tf.float32)
        # Cluster metadata
        clusters_features["cl_m"]= tf.io.FixedLenSequenceFeature([6], dtype=tf.float32)

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
def parse_windows_batch(elements, read_hits=False, read_metadata=False):
    context_features = {
        's_f': tf.io.FixedLenFeature([22], tf.float32),
        's_l': tf.io.FixedLenFeature([3], tf.int64),
        #Window features
        'w_f': tf.io.FixedLenFeature([18], tf.float32),
        # window class
        'w_cl' : tf.io.FixedLenFeature([], tf.int64),
        # number of clusters
        'n_cl' : tf.io.FixedLenFeature([], tf.int64),
        # flag (pdgid id)
        'f' :  tf.io.FixedLenFeature([], tf.int64)
    }
    clusters_features = {
        "cl_f" : tf.io.FixedLenSequenceFeature([26], dtype=tf.float32),
        "cl_l" : tf.io.FixedLenSequenceFeature([6], dtype=tf.int64),
    }

    if read_metadata:
        # seed metadata
        context_features["s_m"] = tf.io.FixedLenFeature([5], tf.float32) 
        # window metadata
        context_features["w_m"] = tf.io.FixedLenFeature([22], tf.float32)
        # Cluster metadata
        clusters_features["cl_m"]= tf.io.FixedLenSequenceFeature([6], dtype=tf.float32)

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
# Functions to get the features indexes from names 

def get_cluster_features_indexes(feats):
    '''
    Utils function to get the index of the requested features from the cluster full features list
    '''
    cl_feat = ["en_cluster","et_cluster",
            "cluster_eta", "cluster_phi", 
            "cluster_ieta","cluster_iphi","cluster_iz",
            "cluster_deta", "cluster_dphi",
            "cluster_den_seed","cluster_det_seed",
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

def get_seed_features_indexes(feats):
    seed_feat =  ["seed_eta","seed_phi", "seed_ieta","seed_iphi", "seed_iz", 
                     "en_seed", "et_seed","en_seed_calib","et_seed_calib",
                    "seed_f5_r9","seed_f5_sigmaIetaIeta", "seed_f5_sigmaIetaIphi",
                    "seed_f5_sigmaIphiIphi","seed_f5_swissCross",
                    "seed_r9","seed_sigmaIetaIeta", "seed_sigmaIetaIphi",
                    "seed_sigmaIphiIphi","seed_swissCross",
                    "seed_nxtals","seed_etaWidth","seed_phiWidth",
                    ]
    output = [] 
    for f in feats:
        if f in seed_feat:
            output.append(seed_feat.index(f))
        else:
            print("Missing branch! ", f)
    return output

def get_window_features_indexes(feats):
    window_features = ["max_en_cluster","max_et_cluster","max_deta_cluster","max_dphi_cluster","max_den_cluster","max_det_cluster",
                         "min_en_cluster","min_et_cluster","min_deta_cluster","min_dphi_cluster","min_den_cluster","min_det_cluster",
                         "mean_en_cluster","mean_et_cluster","mean_deta_cluster","mean_dphi_cluster","mean_den_cluster","mean_det_cluster" ]
    output = [] 
    for f in feats:
        if f in window_features:
            output.append(window_features.index(f))
        else:
            print("Missing branch! ", f)
    return output



def get_window_metadata_indexes(feats):
    window_metadata = ["nVtx", "rho", "obsPU", "truePU",
                         "sim_true_eta", "sim_true_phi",  
                        "en_true_sim","et_true_sim", "en_true_gen", "et_true_gen",
                        "en_mustache_raw", "et_mustache_raw","en_mustache_calib", "et_mustache_calib",
                        "nclusters_insc",
                        "max_en_cluster_insc","max_deta_cluster_insc","max_dphi_cluster_insc",
                        "event_tot_simen_PU","wtot_simen_PU","wtot_recoen_PU","wtot_simen_sig"  ]
    output = [] 
    for f in feats:
        if f in window_metadata:
            output.append(window_metadata.index(f))
        else:
            print("Missing branch! ", f)
    return output


####################################################
# Function to prepare tensors for training 

def prepare_features(dataset,  features, window_features, metadata): 
    ''' The tensor containing the requested featues follow the requested order '''
    feat_index = get_cluster_features_indexes(features)
    seed_feat_index = get_seed_features_indexes(["seed_eta","et_seed"])
    window_feat_index = get_window_features_indexes(window_features)
    metadata_index = get_window_metadata_indexes(metadata)
    def process(*kargs):
        cl_f = kargs[1]['cl_f']
        cl_l = kargs[1]['cl_l']
         # get requested cluster features
        cl_X = tf.gather(cl_f, indices=feat_index,axis=-1)
        cl_hits = kargs[1]['cl_h']
        is_seed = tf.gather(cl_l,indices=[0],axis=-1)
        in_sc = tf.gather(cl_l,indices=[3], axis=-1) 
        # HACK to include in the supercluster all the seeds also for unmatched windows
        #in_sc = tf.cast( tf.math.logical_or(tf.cast(is_seed, tf.bool), tf.cast(in_sc, tf.bool)), tf.int64)
        n_cl = kargs[0]["n_cl"]
        # get requested window metadata
        seed_feat = tf.gather(kargs[0]["s_f"], indices=seed_feat_index,axis=-1)
        wind_X =  tf.gather(kargs[0]["w_f"], indices=window_feat_index, axis=-1)
        w_metadata =  tf.gather(kargs[0]["w_m"], indices=metadata_index, axis=-1)
        # Add window class , flavour
        # Add also some seed information
        wind_meta = tf.concat( [  w_metadata , 
                                  seed_feat,
                                  tf.stack( [ tf.cast(kargs[0]['w_cl'],tf.float32),
                                              tf.cast(kargs[0]['f'],tf.float32)], axis=-1),
                               ], axis=-1) 
        return  cl_X, wind_X, cl_hits, is_seed, n_cl, in_sc, wind_meta

    return dataset.map( process, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False )


# def delta_energy_seed(dataset, en_index, et_index):
#     '''
#     Add to the cluster featrues the delta energy and Et wrt the seed cluster.
#     N.B. to be applied on batched dataset
#     '''
#     def process(*kargs):
#         # The expected kargs are: cl_X, cl_hits, is_seed, n_cl, in_sc, wind_meta
#         # but we stay generic in order to be able to add more elements without changing this code
#         mask_seed = tf.squeeze(kargs[2])
#         mask_seed.set_shape([None,None])
#         seed_en = tf.boolean_mask(kargs[0][:,:,en_index], mask_seed)[:,tf.newaxis]
#         seed_et = tf.boolean_mask(kargs[0][:,:,et_index], mask_seed)[:,tf.newaxis]
#         delta_seed_en = ((seed_en - kargs[0][:,:,en_index]) * tf.cast(kargs[0][:,:,en_index]!=0, tf.float32))[:,:,tf.newaxis]
#         delta_seed_et = ((seed_et - kargs[0][:,:,et_index]) * tf.cast(kargs[0][:,:,et_index]!=0, tf.float32))[:,:,tf.newaxis]
#         cl_X = tf.concat([kargs[0], delta_seed_en, delta_seed_et], axis=-1)
#         output = [cl_X] 
#         for k in kargs[1:]: output.append(k)
#         return output
#     return dataset.map(process, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False )

###########
## Features normalization, to be applied at the same level as the normalization parameters have been calculated
def normalize_features(dataset, file, file_win):
    params = np.load(file)
    m = tf.convert_to_tensor(params["mean"], dtype=tf.float32)
    s = tf.convert_to_tensor(params["sigma"], dtype=tf.float32)
    paramsW = np.load(file_win)
    mw = tf.convert_to_tensor(paramsW["mean"], dtype=tf.float32)
    sw = tf.convert_to_tensor(paramsW["sigma"], dtype=tf.float32)

    def process(cl_X, wind_W, *kargs):
        # Remove mean and divide by sigma
        cl_X_norm = (cl_X - m) / s
        wind_W_norm = (wind_W - mw) / sw
        # Pass through both
        output = [cl_X, cl_X_norm, wind_W, wind_W_norm]
        for k in kargs: output.append(k)
        return output

    return dataset.map(process, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False )

########################################
# Final shapes that are used in the training loop
# Split the tensors in X and Y tuples

def training_format(dataset, norm=True):
    if norm:
        def process(*kargs):
            ''' Function needed to divide the dataset tensors in X,Y for the training loop'''
            cl_X, cl_X_norm, wind_X, wind_X_norm, cl_hits, is_seed, n_cl, in_sc, wind_meta = kargs
            # get window classification target, total number of true clusters, total simenergy and genenergy
            w_flavour = tf.one_hot( tf.cast(wind_meta[:,-1] / 11, tf.int32) , depth=3)
            return (cl_X_norm, wind_X_norm, cl_hits, is_seed, n_cl), (in_sc, w_flavour, cl_X, wind_X, wind_meta)
        return dataset.map(process,num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
    else:
        def process(*kargs):
            ''' Function needed to divide the dataset tensors in X,Y for the training loop'''
            cl_X, wind_X, cl_hits, is_seed, n_cl, in_sc, wind_meta = kargs
            # get window classification target, total number of true clusters, total simenergy and genenergy
            w_flavour = tf.one_hot( tf.cast(wind_meta[:,-1] / 11, tf.int32) , depth=3)
            return (cl_X, wind_X, cl_hits, is_seed, n_cl), (in_sc, w_flavour, wind_meta)
        return dataset.map(process,num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)


 
##############################################
# Loading functions
  
def load_dataset_batch(path, batch_size, options):
    '''
    options = { "read_hits" }
    '''
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(path))
    dataset = dataset.batch(batch_size).map(
                lambda el: parse_windows_batch(el, options.get('read_hits', False), options.get('read_metadata', False)),
                num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
    return dataset

def load_dataset_single(path, options):
    '''
    options = { "read_hits" }
    '''
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(path))
    dataset = dataset.map(
                lambda el: parse_single_window(el, options.get('read_hits', False), options.get('read_metadata', False)),
                num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
    return dataset


# Function to load and prepare a batched dataset with prepared tensors
def load_balanced_dataset_batch(data_paths, features, wind_features, metadata, batch_size, filter=None, weights=None, options={"read_hits":True, "read_metadata":True}):
    datasets = {}
    for n, p in data_paths.items():
        df = load_dataset_single(p, options)
        if filter:
            df = df.filter(filter)
        df = prepare_features(df, features, wind_features, metadata)
        df = df.shuffle(buffer_size=batch_size*30) # Shuffle elements for 15 times sample the batch size
        datasets[n] = df
    if weights:
        ws = [ ]
        for d in datasets.keys():
             ws.append(weights[d])
        total_ds = tf.data.experimental.sample_from_datasets(list(datasets.values()), weights=ws)
    else:
        total_ds = tf.data.experimental.sample_from_datasets(list(datasets.values()), weights=[1/len(datasets)]*len(datasets))
    # Now we can shuffle and batch
    def batch_features(cl_X, wind_X, cl_hits, is_seed, n_cl, in_sc, wind_meta):
        '''This function is used to create padded batches together for dense features and ragged ones'''
        return tf.data.Dataset.zip((cl_X.padded_batch(batch_size), 
                                wind_X.batch(batch_size),
                                cl_hits.batch(batch_size), 
                                is_seed.padded_batch(batch_size), 
                                n_cl.padded_batch(batch_size),
                                in_sc.padded_batch(batch_size),
                                wind_meta.padded_batch(batch_size)))
    #total_ds = total_ds.shuffle(1000, reshuffle_each_iteration=True)
    total_ds_batched = total_ds.window(batch_size).flat_map(batch_features)
    return total_ds_batched


######################### 
#Utils for debugging

def get(dataset):
    el = next(iter(dataset.take(1)))
    return el

