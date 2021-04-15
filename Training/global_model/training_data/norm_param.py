import tensorflow as tf
import numpy as np
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