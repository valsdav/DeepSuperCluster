import argparse
import tensorflow as tf
from loader_awk import get_model_and_dataset
import os

if __name__=="__main__":
  """Example:
  >>> python3 inference_profiling.py --config_path /eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/models_archive/gcn_models/ACAT2022/tests/run_27/training_config.json --weights_name weights.89-10.394977.hdf5 --log_folder /tensorflow_logs/
  """

  parser = argparse.ArgumentParser()
  parser.add_argument('--config_path', type=str, help="Model configuration path.")
  parser.add_argument('--weights_name', type=str, help="Model weights name.")
  parser.add_argument('--log_folder', type=str, help="Folder for saving tensorboard logs.")
  parser.add_argument("--conf-overwrite", type=str, help="Validation config overwrite", required=False)
  args = parser.parse_args()
    
  config_path = args.config_path
  weights_name = args.weights_name
  log_folder = args.log_folder
  
if args.conf_overwrite != None and args.conf_overwrite!= "" and args.conf_overwrite!="None":
    config_overwrite = json.load(open(args.conf_overwrite))
else:
    config_overwrite = None
    
  os.makedirs(log_folder, exist_ok=True)
  
  # Single CPU thread
  num_threads = 1
  
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
  os.environ["TF_NUM_INTEROP_THREADS"] = "1"
  
  tf.config.threading.set_inter_op_parallelism_threads(
      num_threads
  )
  tf.config.threading.set_intra_op_parallelism_threads(
      num_threads
  )
  tf.config.set_soft_device_placement(True)

  print("Starting to load the model...")
  model, dataset, cfg = get_model_and_dataset(config_path, weights_name,
                                              training=False, fixed_X=None,
                                              config_overwrite=config_overwrite)
  print("Model loaded")
  print("Preload the dataset")
  d = dataset.take(50).cache("/tmp/cache"+weights_name)

  print("Profiling the prediction")
  
  with tf.profiler.experimental.Profile(log_folder):
    for x,y,w, in d :
      predictions = model.predict(x)
  print("Done")
