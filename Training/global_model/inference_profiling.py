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
  args = parser.parse_args()
    
  config_path = args.config_path
  weights_name = args.weights_name
  log_folder = args.log_folder
  
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
  model, dataset = get_model_and_dataset(config_path, weights_name,
                            training=False, fixed_X=None, overwrite=None)
  print("Model loaded")
  print("Preload the dataset")
  d0 = dataset.take(1)
  d1 = dataset.take(100)

  print("Tracing the model graph")
  writer = tf.summary.create_file_writer(logdir)
  tf.summary.trace_on(graph=True, profiler=True)

  for x,y,w in d0:
    y = model(x)
    
  with writer.as_default():
    tf.summary.trace_export(
      name="model_trace",
      step=0,
      profiler_outdir=log_folder)

  print("Profiling the prediction")
  
  with tf.profiler.experimental.Profile(log_folder):
    for x,y,w, in d:
      predictions = model.predict(x)
  print("Done")
