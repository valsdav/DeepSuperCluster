import json
import random
import os
import tensorflow as tf
import numpy as np
import collections
import time
import sys
import glob
import gzip

def load_iter(files):
    for filename in files:
        with gzip.open(filename, "rt") as file:
            for line in file:
                start =  line.rfind('{"window_index"')
                try:
                    data1 = json.loads(line[start:])
                except:
                    print("parse error, next")
                    continue
                yield data1
               


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


def make_example_window(window):

    clusters_features = []

    seed_features = ["seed_eta","seed_iz", "en_seed", "et_seed",
                    "seed_ieta","seed_iphi",
                    # "et_seed_calib", 
                    "seed_f5_r9","seed_f5_sigmaIetaIeta", "seed_f5_sigmaIetaIphi","seed_f5_sigmaIphiIphi",
                    "seed_f5_swissCross","seed_nxtals","seed_etaWidth","seed_phiWidth",
                    ]

    X = np.array( [window[f] for f in seed_features],dtype='float32')
    X_rechits = np.array([ [r[0],r[1],r[2],r[4]] for r in  window['seed_hits']], dtype='float32')

    # # the truth array is in_scluster and then true energy
    y = window["is_seed_calo_seed"]
    calo_match = window["is_seed_calo_matched"]
    

    feature = {
        'X': _bytes_feature(tf.io.serialize_tensor(X)),
        'X_hits' : _bytes_feature(tf.io.serialize_tensor(X_rechits)),
        'y': _int64_feature(y),
        'calo_match': _int64_feature(calo_match),
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
    files = glob.glob(input_dir+"/*.ndjson.tar.xz")
    print(files)
    if nlimit_files == None:
        it = load_iter(files)
    else:
        it = load_iter(files[:nlimit_files])

    make_sharded_files(make_example_window, output_dir, it, splits=dict(training=1), shards=100,report_dt=5)
