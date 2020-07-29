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
        example = make_example(next(input_iter))
        writer.write(example.SerializeToString())
        
        examples_per_split[split] += 1
        i += 1

        if report_dt > 0 and time.time() - t0 > report_dt:
            print('processed %d' % i)
            t0 = time.time()

    # Store results.
    for split in splits:
        for writer in writers[split]:
            writer.close()
    
    return dict(**examples_per_split)



def make_example_window(window):
    example = tf.train.Example()

    float_features = ["seed_eta", "seed_phi", "seed_iz", "en_seed", "et_seed",
                    "en_seed_calib", "et_seed_calib", 
                    "seed_f5_r9","seed_f5_sigmaIetaIeta", "seed_f5_sigmaIetaIphi",
                    "seed_f5_sigmaIphiIphi","seed_f5_swissCross","seed_etaWidth",
                    "seed_phiWidth","seed_nxtals"]

    for float_f in float_features:
        example.features.feature[float_f].float_list.value.append(window[float_f])
    
    int_list_features = ["is_seed", "in_scluster","cl_nxtals"]
    for ilist_f in int_list_features:
        example.features.feature[ilist_f].int64_list.value.extend(window["clusters"][ilist_f])

    float_list_features = ["cluster_dphi","en_cluster","et_cluster",
                       "en_cluster_calib", "et_cluster_calib",
                       "cl_f5_r9", "cl_f5_sigmaIetaIeta", "cl_f5_sigmaIetaIphi",
                        "cl_f5_sigmaIphiIphi","cl_f5_swissCross","cl_etaWidth",
                        "cl_phiWidth"]

    for flist_f in float_list_features:
        example.features.feature[flist_f].float_list.value.extend(window["clusters"][flist_f])
    

    return example



if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(input_dir+"/*.ndjson")
    print(files)
    it = load_iter(files)

    make_sharded_files(make_example_window, output_dir, it, splits=dict(training=3, validation=2, test=1), shards=50,report_dt=5)
