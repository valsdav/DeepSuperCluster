import tensorflow as tf

def clusters_classification_loss(y_true, y_pred):
    dense_clclass, dense_windclass, mask_cls, _  = y_pred
    y_clclass, y_windclass, cl_X, y_metadata = y_true
        
    class_loss = tf.keras.losses.binary_crossentropy(y_clclass, dense_clclass, from_logits=True) * mask_cls
    reduced_loss = tf.reduce_mean(tf.reduce_sum(class_loss, axis=-1))
    return reduced_loss 


def energy_weighted_classification_loss(y_true, y_pred):
    dense_clclass,dense_windclass, mask_cls, _  = y_pred
    y_clclass, y_windclass, cl_X, y_metadata = y_true
    cl_ets = cl_X[:,:,1]
    # compute the weighting mean of the loss based on the energy of each seed in the window
    cl_ets_weights = cl_ets / tf.reduce_sum(cl_ets, axis=-1)[:,tf.newaxis]
    class_loss = tf.keras.losses.binary_crossentropy(y_clclass, dense_clclass, from_logits=True) * mask_cls
    weighted_loss = class_loss * cl_ets_weights
    # mean over the batch
    reduced_loss = tf.reduce_mean(tf.reduce_sum(weighted_loss, axis=-1))
    return reduced_loss


def window_classification_loss(y_true, y_pred):
    dense_clclass,dense_windclass, mask_cls, _  = y_pred
    y_clclass, y_windclass, cl_X, y_metadata = y_true

    # Only window multi-class classification
    windclass_loss = tf.keras.losses.categorical_crossentropy(y_windclass, dense_windclass, from_logits=True)
    reduced_loss =   tf.reduce_mean(windclass_loss)
    return reduced_loss


def energy_loss(y_true, y_pred):
    dense_clclass,dense_windclass, mask_cls, _  = y_pred
    y_clclass, y_windclass, cl_X, y_metadata = y_true

    selected_cls =  tf.cast(tf.nn.sigmoid(dense_clclass) > 0.5, tf.int64)
    diff = y_clclass - selected_cls

    # 1 is et_cluster
    missing_en = tf.squeeze(tf.where(diff==1, cl_X[:,:,1][:,:,tf.newaxis], tf.zeros(tf.shape(diff))))
    spurious_en =  tf.squeeze(tf.where(diff==-1, cl_X[:,:,1][:,:,tf.newaxis], tf.zeros(tf.shape(diff))))

    reduced_loss_missing = tf.reduce_mean(tf.reduce_sum(missing_en, axis=-1)) 
    reduced_loss_spurious =  tf.reduce_mean(tf.reduce_sum(spurious_en, axis=-1))
    return reduced_loss_missing,reduced_loss_spurious
