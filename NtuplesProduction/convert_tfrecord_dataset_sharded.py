import json
import random
import os
import tensorflow as tf
import numpy as np
import collections
import time
import sys
import glob

def load_iter(files):
    for file in files:
        print(file)
        for line in open(file):
            if '}{"window_index"' in line:
                lines = line.split('}{"window_index"')
                data1 = json.loads(lines[0]+"}")
                yield data1
                data2 = json.loads('{"window_index"' +  lines[1])
                yield data2
            else:
                data = json.loads(line)
                yield data


# The code in this cell simply takes a list of iterators and then
# randomly distributes the values returned by these iterators into sharded
# datasets (e.g. a train/eval/test split).

def rand_key(counts):
  """Returns a random key from "counts", using values as distribution."""
  r = random.randint(0, sum(counts.values()))
  for key, count in counts.items():
    if r > count or count == 0:
      r -= count
    else:
      counts[key] -= 1
      return key

def get_split(i, splits):
  """Returns key from "splits" for iteration "i"."""
  i %= sum(splits.values())
  for split in sorted(splits):
    if i < splits[split]:
      return split
    i -= splits[split]

def make_counts(labels, total):
  """Generates counts for "labels" totaling "total"."""
  counts = {}
  for i, name in enumerate(labels):
    counts[name] = total // (len(labels) - i)
    total -= counts[name]
  return counts


def make_sharded_files(make_example, path, input_iter, splits,
                       shards=10, overwrite=False, report_dt=10):
    """Create sharded dataset from "iters".

    Args:
        make_example: Converts object returned by elements of "iters"
            to tf.train.Example() proto.
        path: Directory that will contain recordio files.
        labels: Names of labels, will be written to "labels.txt".
        iters: List of iterables returning drawing objects.
        splits: Dictionary mapping filename to multiple examples. For example,
            splits=dict(a=2, b=1) will result in two examples being written to "a"
            for every example being written to "b".
        shards: Number of files to be created per split.
        overwrite: Whether a pre-existing directory should be overwritten.
        report_dt: Number of seconds between status updates (0=no updates).

    Returns:
        Total number of examples written to disk per split.
    """
    # Prepare output.
    if not os.path.exists(path):
        os.makedirs(path)
    paths = {
        split: ['%s/%s-%05d-of-%05d' % (path, split, i, shards)
                for i in range(shards)]
        for split in splits
    }
    writers = {
        split: [tf.io.TFRecordWriter(ps[i]) for i in range(shards)]
        for split, ps in paths.items()
    }
    t0 = time.time()
    examples_per_split = collections.defaultdict(int)
    
    i = 0
    for row in input_iter:
        split = get_split(i, splits)
        writer = writers[split][examples_per_split[split] % shards]
        try:
            example = make_example(next(input_iter))
            writer.write(example.SerializeToString())
        
            examples_per_split[split] += 1
            i += 1

            if report_dt > 0 and time.time() - t0 > report_dt:
                print('processed %d' % i)
                t0 = time.time()
        except StopIteration:
            print("ended files")

    # Store results.
    for split in splits:
        for writer in writers[split]:
            writer.close()
    
    return dict(**examples_per_split)

#https://stackoverflow.com/questions/47861084/how-to-store-numpy-arrays-as-tfrecord
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  """Returns an float32_list from a bool / enum / int / uint."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def scale_features_clusters(X):
    '''
    'is_seed',"cluster_deta", "cluster_dphi", "en_cluster", "et_cluster",
    "cl_f5_r9", "cl_f5_sigmaIetaIeta","cl_f5_sigmaIetaIphi","cl_f5_sigmaIphiIphi",
    "cl_f5_swissCross", "cl_nxtals", "cl_etaWidth", "cl_phiWidth
    '''
    x_mean = np.array( 
        [   0.,  -7.09402501e-04, -1.27142875e-04,  1.30375508e+00,  5.67249500e-01, 
            1.92096066e+00,  1.31476120e-02,  1.62948213e-05,  1.42948806e-02,
            5.92920497e-01,  1.49597644e+00,  3.36213188e-03,  3.06446267e-03]
        )

    x_scale = np.array(
        [  1.,  1.10279784e-01, 3.30488055e-01, 2.62605247e+00, 1.16284769e+00,
            7.81094814e+00, 1.70392176e-02, 3.05995567e-04, 1.80176053e-02,
            1.99316624e+00, 1.88845046e+00, 4.12315715e-03, 4.79639033e-03]       
        )
    return (X-x_mean)/ x_scale

def scale_features_seed(X):
    '''
     "seed_eta", "seed_iz","en_seed","et_seed",
     "seed_f5_r9", "seed_f5_sigmaIetaIeta","seed_f5_sigmaIetaIphi","seed_f5_sigmaIphiIphi",
     "seed_f5_swissCross","seed_nxtals", "seed_etaWidth", "seed_phiWidth",
    '''
    x_mean = np.array( 
        [   6.84241156e-03,  1.62242679e-03,  5.81495577e+01,  2.57215845e+01, 
            1.00772582e+00,  1.35803461e-02, -4.29317013e-06,  1.71072024e-02,
            4.90466869e-01,  5.10511982e+00,  8.82101138e-03,  1.04095965e-02 ]
    
        )

    x_scale = np.array(
        [   1.31333380e+00, 5.06988411e-01, 9.21157365e+01, 2.98580765e+01, 
            1.17047757e-01, 1.11969442e-02, 1.86572967e-04, 1.31036359e-02,
            4.01511744e-01, 5.67007350e+00, 6.14304203e-03, 7.24808860e-03]       
        )
    return (X-x_mean)/ x_scale

def make_example_window(window):

    clusters_features = []

    seed_features = ["seed_eta","seed_iz", "en_seed", "et_seed",
                    # "et_seed_calib", 
                    "seed_f5_r9","seed_f5_sigmaIetaIeta", "seed_f5_sigmaIetaIphi","seed_f5_sigmaIphiIphi",
                    "seed_f5_swissCross","seed_nxtals","seed_etaWidth","seed_phiWidth",
                    ]

    cls_features = [   "is_seed", "cluster_deta", "cluster_dphi","en_cluster","et_cluster",
                      # "en_cluster_calib", "et_cluster_calib",
                       "cl_f5_r9", "cl_f5_sigmaIetaIeta", "cl_f5_sigmaIetaIphi","cl_f5_sigmaIphiIphi",
                       "cl_f5_swissCross","cl_nxtals", "cl_etaWidth","cl_phiWidth",]
    
    for feat in cls_features:
        clusters_features.append(np.array(window["clusters"][feat],dtype='float32'))
    
    X = np.stack(clusters_features).T

    X_seed = np.expand_dims(np.array( [window[f] for f in seed_features],dtype='float32'), axis=0)

    # # the truth array is in_scluster and then true energy
    y_array = window["clusters"]['in_scluster']
    
    #true_energy = sum([ window['clusters']['en_cluster'][i]*isin for i,isin in enumerate(window["clusters"]['in_scluster'])])
    true_energy = window['en_true']


    y_array = [true_energy] + y_array
    y = np.array(y_array, dtype='float32')   

    # print("X: ", X.shape)
    # print("X_seed: ", X_seed.shape)
    # print("y: ", y.shape)

    feature = {
        'X': _bytes_feature(tf.io.serialize_tensor(X)),
        'X_seed' : _bytes_feature(tf.io.serialize_tensor(X_seed)),
        'y': _bytes_feature(tf.io.serialize_tensor(y)),
        'y_energy' :  _float_feature(true_energy),
        'n_clusters':_int64_feature(len(window['clusters']['is_seed']))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example



if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if len(sys.argv) > 3:
        nlimit_files = int(sys.argv[3])
    else:
        nlimit_files = None 

    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(input_dir+"/*.ndjson")
    print(files)
    if nlimit_files == None:
        it = load_iter(files)
    else:
        it = load_iter(files[:nlimit_files])

    make_sharded_files(make_example_window, output_dir, it, splits=dict(training=1), shards=100,report_dt=5)
