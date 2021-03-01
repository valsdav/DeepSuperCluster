import tensorflow as tf

def parse_window_all(element):
    context_features = {
        'Locale': tf.FixedLenFeature([], dtype=tf.string),
        'Age': tf.FixedLenFeature([], dtype=tf.int64),
        'Favorites': tf.VarLenFeature(dtype=tf.string)
    }
    sequence_features = {
        'Movie Names': tf.FixedLenSequenceFeature([], dtype=tf.string),
        'Movie Ratings': tf.FixedLenSequenceFeature([], dtype=tf.float32),
        'Movie Actors': tf.VarLenFeature(dtype=tf.string)
    }

     context_features = {
        's_f': tf.io.FixedLenFeature([], tf.string),
        's_l': tf.io.FixedLenFeature([], tf.int64),
        's_m': tf.io.FixedLenFeature([], tf.string),
        's_h': tf.io.VarLenFeature([], tf.string),
        # window class
        'w_cl' : tf.io.FixedLenFeature([], tf.int64),
        # number of clusters
        'n_cl' : tf.io.FixedLenFeature([], tf.int64),
        # flag (pdgid id)
        'f' :  tf.io.FixedLenFeature([], tf.int64)
    }
    
    clusters_features = {
            "cl_f" : tf.FixedLenSequenceFeature([], dtype=tf.string),
            "cl_m" : tf.FixedLenSequenceFeature([], dtype=tf.string),
            "cl_l" : tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "cl_h" : tf.VarLenFeature([], dtype=tf.string),
    }

    example = tf.io.parse_single_sequence_example(element, context_features=context_features, sequence_features=clusters_features)
    X = example_message['X']
    X_hits = example_message['X_hits']
    y = example_message['y']
    
    arr_X = tf.io.parse_tensor(X, out_type=tf.float32)
    arr_X_hits = tf.io.parse_tensor(X_hits, out_type=tf.float32)
    arr_y = y
    
    #https://github.com/tensorflow/tensorflow/issues/24520#issuecomment-577325475
    arr_X.set_shape( tf.TensorShape(args.nfeatures))
    arr_X_hits.set_shape(tf.TensorShape((None, 4))) #hits features
    
    #arr_y.set_shape( tf.TensorShape(1))
    arr_X = tf.boolean_mask(arr_X, [True,True,True,True,False,False,False,True,True,True,True,True,True,True])
    
    return arr_X, arr_y
  

# Create datasets from TFRecord files.
dataset = tf.data.TFRecordDataset(tf.io.gfile.glob('{}/training-*'.format(data_path)))
dataset = dataset.map(_parse_tfr_element,num_parallel_calls=tf.data.experimental.AUTOTUNE)
#dataset = dataset.shuffle(10000, reshuffle_each_iteration=True)

ds_pos_train = dataset.filter(lambda X,Y: Y>0).take(args.ntrain//2)
ds_neg_train = dataset.filter(lambda X,Y: Y==0).take(args.ntrain//2)
ds_pos_test = dataset.filter(lambda X,Y: Y>0).skip(args.ntrain//2).take(args.nval//2)
ds_neg_test = dataset.filter(lambda X,Y: Y==0).skip(args.ntrain//2).take(args.nval//2)

ds_train = tf.data.experimental.sample_from_datasets([ds_pos_train, ds_neg_train], weights=[0.5, 0.5]).batch(args.batch_size).prefetch(10)
ds_test = tf.data.experimental.sample_from_datasets([ds_pos_test, ds_neg_test], weights=[0.5, 0.5]).batch(args.batch_size).prefetch(10)

ds_train_r = ds_train.repeat(args.nepochs)
ds_test_r = ds_test.repeat(args.nepochs)