import tensorflow as tf
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
import tf_data

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
        (cl_X, _, _, n_cl), (*_) = el
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
        (cl_X, _, _, n_cl), (*_) = el
        cl_X = cl_X[:,:,0:n_features]
        # create mask to eliminate the padded values from calculation
        mask = tf.expand_dims(tf.cast(tf.reduce_sum(cl_X, axis=-1) != 0., tf.float32), axis=-1)
        
        dif_masked = mask*(cl_X-m)
        s +=tf.reduce_sum(tf.math.pow(dif_masked, 2), axis=(0,1)).numpy()
    s = tf.math.sqrt(s/total_cl)
    sigma = dict(zip(features, s))
    return mean, sigma

def convert_df(data_path, features, n_samples=100):
    '''
    Function to convert tensorflow dataset to pandas dataframe.
    
    Return: 
    - df: pandas dataframe with columns=features + is_seed.
    
    Args: 
    - data_path: dict with path to the .proto file.
    - features: features to be saved in the dataframe.
    - n_samples (default=100): number of samples to save. 
    '''
    # prepare the required variables
    dataframes = []
    ds = tf_data.load_balanced_dataset_batch(data_path, features, 1).take(n_samples)
    features_ext = features["cl_features"]
    
    # iterate through the dataset
    for el in ds: 
        cl_X, _,_, is_seed, n_cl, in_sc, wind_meta,_ = el
        # choose only features that were passed to the function
        X_features = [cl_X[0,:,features_ext.index(feature)] for feature in features_ext]
        d = dict(zip(features_ext, X_features))
        
        # save each entry in dataset
        df_el = pd.DataFrame(data=d)
        df_el['is_seed'] = is_seed[0].numpy()
        df_el['n_cl'] = n_cl[0].numpy()
        df_el['in_sc'] = in_sc[0].numpy()
        
        # window metadata info  
        for i, meta in enumerate(features['window_metadata']):
            df_el[meta] = wind_meta[0,i].numpy()
           
        df_el['seed_eta'] = wind_meta[0,-3].numpy()
        
        dataframes.append(df_el)
    
    # aggregate all the samples
    df = pd.concat(dataframes, keys=np.arange(0,n_samples))
    return df