import os 
import importlib.util
import tensorflow as tf 
from collections import namedtuple
from pprint import pprint
import json

    
def load_dataset(tf_data_lib, path_dict, batch_size, normalization_files, features_dict=None, weights=None, training=False,  nevents=None,):
    # Load a balanced dataset from the list of paths given to the function. Selected only the requestes features from clusters and prepare batches
    dataset = tf_data_lib.load_balanced_dataset_batch(path_dict, features_dict, batch_size, weights=weights, training=training)
    # the indexes for energy and et are from the features list we requestes
    dataset = tf_data_lib.normalize_features(dataset, normalization_files[0], normalization_files[1],
                                         features_dict["cl_features"], features_dict["window_features"])
    dataset = tf_data_lib.training_format(dataset)
    # Create training and validation
    dataset  = dataset.prefetch(300)
    if nevents:
        dataset = dataset.take(nevents// batch_size)
    return dataset


def get_model(args, model_definition_path, weights_path, X):
    #Load args
    # Convert the activation function
    args['activation'] = tf.keras.activations.get(args['activation'])

    # Load model modules
    spec = importlib.util.spec_from_file_location("model", model_definition_path)
    model_lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_lib)

    tf.keras.backend.clear_session()
    # Construction of the model
    model = model_lib.DeepClusterGN(**args)
    # model.set_metrics()
    #Call the model once
    y = model(X)    
    model.summary()
    #Loading weights
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    return model 
 

def get_model_and_dataset(config_path, weights_path, training=False, fixed_X=None):
    # Load configs
    args = json.load(open(config_path))
    print("Model options: ")
    pprint(args)
    # Load the tf_data lib
    spec = importlib.util.spec_from_file_location("tf_data", args["data_definition_path"])
    tf_data_lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tf_data_lib)
    # Features and dataset
    data_path = {} 
    for name,path in args["data_path"].items():
        if training:
            data_path[name] = path
        else:
            data_path[name] = path.replace("training","testing")
    

    print(">> Load the dataset ")
    dataset = load_dataset(tf_data_lib, data_path, args["batch_size"], args["normalizations"],
                             args["features_dict"], training=training, weights=None, nevents=None )

    # Get one instance
    print(">> Load the model")
    if fixed_X == None:
        X,y,W = tf_data_lib.get(dataset)
        
        model = get_model(args, args["model_definition_path"],
                weights_path=weights_path, X=X)
    else:
        model = get_model(args, args["model_definition_path"],
                weights_path=weights_path, X=fixed_X)


    return model, dataset

