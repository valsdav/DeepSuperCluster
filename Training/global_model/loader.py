import os 
import importlib.util
import tensorflow as tf 
from collections import namedtuple
from pprint import pprint
import json
import tf_data

    

def load_dataset(path_dict, batch_size, normalization_files, nevents=None, features_dict=None, weights=None,  training=False ):
    # Load a balanced dataset from the list of paths given to the function. Selected only the requestes features from clusters and prepare batches
    test_ds = tf_data.load_balanced_dataset_batch(path_dict, features_dict, batch_size, weights=weights, training=training)
    # the indexes for energy and et are from the features list we requestes
    test_ds = tf_data.normalize_features(test_ds, normalization_files[0], normalization_files[1],
                                         features_dict["cl_features"], features_dict["window_features"])
    test_ds = tf_data.training_format(test_ds)
    # Create training and validation
    ds_test  = test_ds.prefetch(100)
    if nevents:
        ds_test = ds_test.take(nevents// batch_size)
    return ds_test


def get_model(config_path, definition_path, weights_path, X):
    #Load args
    args = json.load(open(config_path))
    print("Model options: ")
    pprint(args)
    # Convert the activation function
    args['activation'] = tf.keras.activations.get(args['activation'])

    # Load model modules
    spec = importlib.util.spec_from_file_location("model", definition_path)
    model_lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_lib)

    tf.keras.backend.clear_session()
    # Construction of the model
    model = model_lib.DeepClusterGN( **args)
    model.set_metrics()
    #Call the model once
    y = model(X)    
    model.summary()
    #Loading weights
    model.load_weights(weights_path)
    return model 
 