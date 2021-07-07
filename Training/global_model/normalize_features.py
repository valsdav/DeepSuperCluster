import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas


import tensorflow as tf
import tf_data

data_path_train = {"ele_match": "/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/electrons/recordio_allinfo_v11/training/calo_matched/*.proto",
                  "gamma_match": "/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/gammas/recordio_allinfo_v11/training/calo_matched/*.proto",
                #  "nomatch": "/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/electrons/recordio_allinfo_v10/training/no_calo_matched/*.proto",
                  #"gamma_nomatch": "/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/gammas/recordio_allinfo_v2/training/no_calo_matched/*.proto"
                  }

feat = {
 "cl_features" : [ "en_cluster","et_cluster",
                        "cluster_eta", "cluster_phi", 
                        "cluster_ieta","cluster_iphi","cluster_iz",
                        "cluster_deta", "cluster_dphi",
                        "cluster_den_seed","cluster_det_seed",
                        "en_cluster_calib", "et_cluster_calib",
                        "cl_f5_r9", "cl_f5_sigmaIetaIeta", "cl_f5_sigmaIetaIphi",
                        "cl_f5_sigmaIphiIphi","cl_f5_swissCross",
                        "cl_r9", "cl_sigmaIetaIeta", "cl_sigmaIetaIphi",
                        "cl_sigmaIphiIphi","cl_swissCross",
                        "cl_nxtals", "cl_etaWidth","cl_phiWidth"],

 "window_features" : [ "max_en_cluster","max_et_cluster","max_deta_cluster","max_dphi_cluster","max_den_cluster","max_det_cluster",
                    "min_en_cluster","min_et_cluster","min_deta_cluster","min_dphi_cluster","min_den_cluster","min_det_cluster",
                    "mean_en_cluster","mean_et_cluster","mean_deta_cluster","mean_dphi_cluster","mean_den_cluster","mean_det_cluster" ],

}

# Load a balanced dataset from the list of paths given to the function. Selected only the requestes features from clusters and prepare batches
train_ds = tf_data.load_balanced_dataset_batch(data_path_train, feat, batch_size= 300, 
                                                weights={"ele_match":0.5,"gamma_match":0.5})
# the indexes for energy and et are from the features list we requestes
train_ds = tf_data.training_format(train_ds, norm=False)

# Create training and validation
ds_train = train_ds.take(30000)

def parameters(ds, features):
    '''
    Function to calculate the parameters (mean, sigma) for features' distributions.
    
    Return: 
    - mean (dim: n_features): mean of the features' distributions.
    - sigma (dim: n_features): sigma of the features' distributions. 
    
    Args:
    - ds: tensorflow dataset (in the format after tf_data.training_format)
    - features: list of all the features recorded in the dataset. 
    '''
    # initialize counting variables
    n_features = len(features)
    
    total_cl = 0.
    m = tf.zeros(shape=n_features)
    s = tf.zeros(shape=n_features)
    
    # iterate through dataset to calculate mean 
    for el in ds:
        (cl_X, _, _, _, n_cl), (*_) = el
        cl_X = cl_X[:,:,0:n_features]
        m += tf.reduce_sum(cl_X, axis=(0,1)).numpy()
        total_cl += tf.reduce_sum(n_cl).numpy()
    
    # calculate mean for each feature, create dictionary with feature labels
    #ind = tf_data.get_cluster_features_indexes(features)
    m = m/total_cl
    mean = dict(zip(features, m))
    m = tf.reshape(m, shape=[1,1,-1])
    
    # iterate through dataset to calculate sigma
    for el in ds: 
        (cl_X, _,_, _, n_cl), (*_) = el
        cl_X = cl_X[:,:,0:n_features]
        # create mask to eliminate the padded values from calculation
        mask = tf.expand_dims(tf.cast(tf.reduce_sum(cl_X, axis=-1) != 0., tf.float32), axis=-1)
        
        dif_masked = mask*(cl_X-m)
        s +=tf.reduce_sum(tf.math.pow(dif_masked, 2), axis=(0,1)).numpy()
    s = tf.math.sqrt(s/total_cl)
    sigma = dict(zip(features, s))
    return m,s , mean, sigma



def parameters_wind(ds, features):
    '''
    Function to calculate the parameters (mean, sigma) for features' distributions.
    
    Return: 
    - mean (dim: n_features): mean of the features' distributions.
    - sigma (dim: n_features): sigma of the features' distributions. 
    
    Args:
    - ds: tensorflow dataset (in the format after tf_data.training_format)
    - features: list of all the features recorded in the dataset. 
    '''
    
    # initialize counting variables
    n_features = len(features)
    
    total_cl = 0.
    m = tf.zeros(shape=n_features)
    s = tf.zeros(shape=n_features)
    
    # iterate through dataset to calculate mean 
    for el in ds:
        (_, wind_X,_, _, n_cl), (*_) = el
        wind_X = wind_X[:,0:n_features]

        m += tf.reduce_sum(wind_X, axis=(0)).numpy()
        total_cl += len(wind_X)
    
    # calculate mean for each feature, create dictionary with feature labels
    #ind = tf_data.get_cluster_features_indexes(features)
    m = m/total_cl
    mean = dict(zip(features, m))
    m = tf.reshape(m, shape=[1,-1])
    
    # iterate through dataset to calculate sigma
    for el in ds: 
        (_, wind_X,_, _, n_cl), (*_) = el
        wind_X = wind_X[:,0:n_features]
        # create mask to eliminate the padded values from calculation
        #mask = tf.expand_dims(tf.cast(tf.reduce_sum(cl_X, axis=-1) != 0., tf.float32), axis=-1)
        
#         dif_masked = mask*(cl_X-m)
        s +=tf.reduce_sum(tf.math.pow(wind_X-m, 2), axis=(0)).numpy()
    s = tf.math.sqrt(s/total_cl)
    sigma = dict(zip(features, s))
    return m,s , mean, sigma


################################

m ,s , mean, sigma = parameters(ds_train, feat["cl_features"])
print(mean)
print(sigma)
np.savez("normalization.npz", mean=m, sigma=s)

mw ,sw , meanw, sigmaw = parameters_wind(ds_train, feat["window_features"])
print(meanw)
print(sigmaw)
np.savez("normalization_wind_features.npz", mean=mw, sigma=sw)