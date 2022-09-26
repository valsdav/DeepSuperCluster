import os
import awk_data
import importlib.util
import tensorflow as tf 
from collections import namedtuple
from pprint import pprint
import json


def get_model(args, model_definition_path, weights_path, X):
    #Load args
    # Convert the activation function
    args['activation'] = tf.keras.activations.get(args['activation'])

    # Load model modules
    print(f"Loading module from:  {model_definition_path}")
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
 

def get_model_and_dataset(config_path, weights_path,
                          training=False, fixed_X=None, overwrite=None):
    # Load configs
    args = json.load(open(config_path))
    print("Model options: ")
    if overwrite!=None:
        args.update(overwrite)
    pprint(args)
    
    # Load the dataset
    print(">> Load the dataset ")
    if training:
        dataset = awk_data.load_dataset(awk_data.LoaderConfig(**args["dataset_conf"]["training"])) 
    else:
        dataset = awk_data.load_dataset(awk_data.LoaderConfig(**args["dataset_conf"]["validation"]))
  
    # Get model instance
    print(">> Load the model")
    if fixed_X == None:
        X,y,W = awk_data.get(dataset)

        model = get_model(args, args["model_definition_path"],
                          weights_path=os.path.join(args["models_path"],
                                                    weights_path), X=X)
    else:
        model = get_model(args, args["model_definition_path"],
                          weights_path=os.path.join(args["models_path"],weights_path),
                          X=fixed_X)


    return model, dataset



