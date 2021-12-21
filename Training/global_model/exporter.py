import loader 
import cmsml
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-c","--config", type=str, help="Model config", required=True)
parser.add_argument("-w", "--model-weights", type=str, help="Weights h5", required=True)
parser.add_argument("-o", "--output", type=str, help="Output", required=True)
parser.add_argument("--max-ncls", type=int, help="Max number of clusters", required=True)
parser.add_argument("--max-nrechits", type=int, help="Max number of rechits", required=True)
args = parser.parse_args()


# args = "/eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/models_archive/gcn_models/gcn_models_SA_v19/run_03_testexport/args_load.json"

# weights = "/eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/models_archive/gcn_models/gcn_models_SA_v19/run_03_testexport/weights.best.hdf5"

ncls = args.max_ncls
nrec = args.max_nrechits
X = ( 
  tf.zeros((50, ncls, 12)), #to be made more generic
  tf.zeros((50, 18)),
  tf.zeros((50, ncls, nrec, 4)),
  tf.zeros((50, ncls, 1)),
  tf.zeros((50)),
  )

model,dataset, _ = loader.get_model_and_dataset(args.config,  args.model_weights, training=False, fixed_X=X)

cmsml.tensorflow.save_graph("graph.pb.txt", model, variables_to_constants=True)