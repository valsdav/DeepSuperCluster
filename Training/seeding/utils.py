import tensorflow as tf


def parse_window_all(element, read_hits=False):
    context_features = {
        's_f': tf.io.FixedLenFeature([24], tf.float32),
        's_l': tf.io.FixedLenFeature([3], tf.int64),
        's_m': tf.io.FixedLenFeature([8], tf.float32),
        # window class
        'w_cl' : tf.io.FixedLenFeature([], tf.int64),
        # number of clusters
        'n_cl' : tf.io.FixedLenFeature([], tf.int64),
        # flag (pdgid id)
        'f' :  tf.io.FixedLenFeature([], tf.int64)
    }
    clusters_features = {
        "cl_f" : tf.io.FixedLenSequenceFeature([22], dtype=tf.float32),
        "cl_m" : tf.io.FixedLenSequenceFeature([1], dtype=tf.float32),
        "cl_l" : tf.io.FixedLenSequenceFeature([6], dtype=tf.int64),
    }
    if read_hits:
        context_features['s_h'] = tf.io.FixedLenFeature([], tf.string)
        clusters_features["cl_h0"] = tf.io.RaggedFeature(dtype=tf.float32)
        clusters_features["cl_h1"] = tf.io.RaggedFeature(dtype=tf.float32)
        clusters_features["cl_h2"] = tf.io.RaggedFeature(dtype=tf.float32)
        clusters_features["cl_h4"] = tf.io.RaggedFeature(dtype=tf.float32)

    ex = tf.io.parse_single_sequence_example(element, context_features=context_features, sequence_features=clusters_features)
    
    if read_hits:
        seed_hits = tf.io.parse_tensor(ex[0]['s_h'], out_type=tf.float32)
        seed_hits.set_shape(tf.TensorShape((None, 4))) 
        ex[0]['s_h'] = seed_hits
        
        cluster_hits = tf.ragged.stack([ex[1]['cl_h0'], ex[1]['cl_h1'],ex[1]['cl_h2'],ex[1]['cl_h4']],axis=2)
        ex[1]['cl_h'] = cluster_hits 
        ex[1].pop("cl_h0")
        ex[1].pop("cl_h1")
        ex[1].pop("cl_h2")
        ex[1].pop("cl_h4")
    
    return ex
  
def load_dataset(path, options):
    '''
    options = { "read_hits" }
    '''
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(path))
    dataset = dataset.map(lambda el: parse_window_all(el, options['read_hits']),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

# # Create datasets from TFRecord files.
# dataset = tf.data.TFRecordDataset(tf.io.gfile.glob('{}/training-*'.format(data_path)))
# dataset = dataset.map(_parse_tfr_element,num_parallel_calls=tf.data.experimental.AUTOTUNE)
# #dataset = dataset.shuffle(10000, reshuffle_each_iteration=True)

# ds_pos_train = dataset.filter(lambda X,Y: Y>0).take(args.ntrain//2)
# ds_neg_train = dataset.filter(lambda X,Y: Y==0).take(args.ntrain//2)
# ds_pos_test = dataset.filter(lambda X,Y: Y>0).skip(args.ntrain//2).take(args.nval//2)
# ds_neg_test = dataset.filter(lambda X,Y: Y==0).skip(args.ntrain//2).take(args.nval//2)

# ds_train = tf.data.experimental.sample_from_datasets([ds_pos_train, ds_neg_train], weights=[0.5, 0.5]).batch(args.batch_size).prefetch(10)
# ds_test = tf.data.experimental.sample_from_datasets([ds_pos_test, ds_neg_test], weights=[0.5, 0.5]).batch(args.batch_size).prefetch(10)

# ds_train_r = ds_train.repeat(args.nepochs)
# ds_test_r = ds_test.repeat(args.nepochs)