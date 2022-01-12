import tensorflow as tf
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
import tf_data_jet as tf_data
from tqdm import tqdm
import time

from collections import defaultdict

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
       #(cl_X, _, _, n_cl), (*_) = el
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
     #   (cl_X, _, _, n_cl), (*_) = el
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
    df = defaultdict(list)
    ind = []
    
    ds = tf_data.load_balanced_dataset_batch(data_path, features, 300, jet=True).take(n_samples)
    features_ext = features["cl_features"]
    
    t = 0
    # iterate through the dataset
    for el in tqdm(ds): 
        
        # define dictionary to collect the features
        data = defaultdict(list)
        
        cl_X, wind_X, cl_hits, is_seed, n_cl, weight, in_sc, wind_meta, cl_labels = el
        
        # create a multiindex for the dataset
        index_1 = np.repeat(np.arange(t+0, t+cl_X.shape[0]), cl_X.shape[1])
        index_2 = np.array([np.arange(0, cl_X.shape[1])] * cl_X.shape[0]).flatten()
        mindex = list(zip(index_1, index_2))
        
        t += cl_X.shape[0]
        
        index = pd.MultiIndex.from_tuples(mindex)
        
        # choose only features that were passed to the function
        X_features = [cl_X[:,:,features_ext.index(feature)].numpy().flatten() for feature in features_ext]
        
        
        for k, v in zip(features_ext, X_features): 
            data[k].append(v)
        

        # add extra info
        data['is_seed'].append(is_seed.numpy().flatten())
        data['in_sc'].append(in_sc.numpy().flatten())
        
        data['n_cl'].append(np.repeat(n_cl.numpy().flatten(), cl_X.shape[1]))

        data['weight'].append(np.repeat(weight.numpy().flatten(), cl_X.shape[1]))
        
        # window metadata info  
        for i, meta in enumerate(features['window_metadata']):
            data[meta].append(np.repeat(wind_meta[:,i].numpy().flatten(), cl_X.shape[1]))
           
        data['seed_eta'].append(np.repeat(wind_meta[:,-3].numpy().flatten(), cl_X.shape[1]))

        data_final = {}  
        
        
        data_final = {k:np.concatenate(v) for (k,v) in data.items()}
#         for k,v in data.items():
#             data_final[k] = np.concatenate(v)
        #t1 = time.time()
        #batch_df = pd.DataFrame(index=index, data=data_final)
        #batch_df = batch_df.loc[batch_df['en_cluster'] != 0]
        #t2 = time.time()
        #print(batch_df)
        for k, v in data_final.items():
            df[k].append(v)
        #df.append(data_final)
        ind.extend(index)
#     # aggregate all the samples
    #df = {}
    #print(df)
    #df = {k:np.concatenate(v) for (k,v) in df.items()}
    df = {k:np.concatenate(v) for (k,v) in df.items()}

    ind = pd.MultiIndex.from_tuples(ind)
    df_final = pd.DataFrame(data=df, index=ind)
    
    df_final = df_final.loc[df_final['en_cluster'] != 0]
    
#     if len(df) != 1: 
#         df_final = pd.concat(df)
#     else: 
#         df_final = df
    return df_final
