import loader_awk
import awk_data
import tensorflow as tf
import cmsml
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("-c","--model-config", type=str, help="Model config", required=True)
parser.add_argument("-m","--model-python", type=str, help="Model code to use for export", required=False)
parser.add_argument("-w", "--model-weights", type=str, help="Weights h5", required=True)
parser.add_argument("-o", "--output", type=str, help="Output model", required=True)
parser.add_argument("-os", "--output-scaler", type=str, help="Output scalers", required=True)
parser.add_argument("--max-ncls", type=int, help="Max number of clusters", required=True)
parser.add_argument("--max-nrechits", type=int, help="Max number of rechits", required=True)
parser.add_argument("--conf-overwrite", type=str, help="Validation config overwrite", required=False)
args = parser.parse_args()


if args.conf_overwrite != None and args.conf_overwrite!= "" and args.conf_overwrite!="None":
    config_overwrite = json.load(open(args.conf_overwrite))
else:
    config_overwrite = None

if args.model_python!=None:
  if config_overwrite == None: config_overwrite = {}
  config_overwrite["model_definition_path"] = args.model_python

#preloading the configuration
ccf = json.load(open(args.model_config))
cfg = awk_data.LoaderConfig(**ccf["dataset_conf"]["validation"])

ncls = args.max_ncls
nrec = args.max_nrechits

if cfg.include_rechits:
  X = ( 
    tf.zeros((32, ncls, len(cfg.columns["cl_features"]))), #to be made more generic
    tf.zeros((32, len(cfg.columns["window_features"]))),
    tf.zeros((32, ncls, nrec, 4)), # if we have rechits
    tf.zeros((32, ncls)),   #is_seed
    tf.zeros((32, ncls)),   #clsmask
    tf.zeros((32, ncls, nrec)) #rechitmask
  )
else:
    X = ( 
    tf.zeros((32, ncls, len(cfg.columns["cl_features"]))), #to be made more generic
    tf.zeros((32, len(cfg.columns["window_features"]))),
    tf.zeros((32, ncls)), 
    tf.zeros((32, ncls)),
  )
    
  
print("Loading model and dataset")
model, dataset, cfg = loader_awk.get_model_and_dataset(args.model_config, args.model_weights,
                                                       training=False,
                                                       awk_dataset=False,
                                                       overwrite=config_overwrite, fixed_X=X)



#cmsml.tensorflow.save_graph(args.output, model, variables_to_constants=True)

# Doing the export by hand since we have to change the names
from tensorflow.python.keras.saving import saving_utils
model_func = saving_utils.trace_model_call(model)
obj = model_func.get_concrete_function()
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
frozen_func = convert_variables_to_constants_v2(obj)
graph_def = frozen_func.graph.as_graph_def()

for node in graph_def.node:
  if node.name == "Identity":
    node.name = "cl_class"
  if node.name == "Identity_1":
    node.name = "wind_class"

tf.io.write_graph(graph_def, "./", args.output, as_text=".txt" in args.output)

# Export scaler files
# Read the norm files from json
norm_cfg = json.load(open(cfg.norm_factors_file))

with open(args.output_scaler+ "_cls_norm.txt","w") as o:
  clnorm = norm_cfg['cluster']
  for col in cfg.columns["cl_features"]:
    o.write(f"{col} MeanRms {clnorm['mean'][col]} {clnorm['std'][col]}\n")

    
with open(args.output_scaler+ "_wind_norm.txt","w") as o:
  clnorm = norm_cfg['window']
  for col in cfg.columns["window_features"]:
    o.write(f"{col} MeanRms {clnorm['mean'][col]} {clnorm['std'][col]}\n")


    
