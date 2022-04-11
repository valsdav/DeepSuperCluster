import loader 
import cmsml
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-c","--config", type=str, help="Model config", required=True)
parser.add_argument("-w", "--model-weights", type=str, help="Weights h5", required=True)
parser.add_argument("-o", "--output", type=str, help="Output model", required=True)
parser.add_argument("-os", "--output-scaler", type=str, help="Output scalers", required=True)
parser.add_argument("--max-ncls", type=int, help="Max number of clusters", required=True)
parser.add_argument("--max-nrechits", type=int, help="Max number of rechits", required=True)
args = parser.parse_args()


ncls = args.max_ncls
nrec = args.max_nrechits
X = ( 
  tf.zeros((2, ncls, 12)), #to be made more generic
  tf.zeros((2, 18)),
  tf.zeros((2, ncls, nrec, 4)),
  tf.zeros((2, ncls, 1)),
  tf.zeros((2)),
  )

model,dataset, _, norm_tensors = loader.get_model_and_dataset(args.config,  args.model_weights, training=False, fixed_X=X)

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
m ,s = norm_tensors["cl_features"]
m = m.numpy().squeeze()
s = s.numpy().squeeze()

with open(args.output_scaler+ "_cls_norm.txt","w") as o:
  for i in range(m.shape[0]):
    o.write("{} {}\n".format(m[i], s[i]))

m ,s = norm_tensors["wind_features"]
m = m.numpy().squeeze()
s = s.numpy().squeeze()

with open(args.output_scaler+ "_wind_norm.txt","w") as o:
  for i in range(m.shape[0]):
    o.write("{} {}\n".format(m[i], s[i]))
