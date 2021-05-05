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
    y_target = tf.cast(y_clclass, tf.float32) 

    pred_prob = tf.nn.sigmoid(dense_clclass)
    diff = tf.math.abs(y_target - pred_prob)
    Et = cl_X[:,:,1:2]
    missing_en = Et * diff * y_target
    spurious_en =  Et * diff * (1 - y_target)
    reduced_loss_missing = tf.reduce_mean(tf.reduce_sum(missing_en, axis=1)) 
    reduced_loss_spurious =  tf.reduce_mean(tf.reduce_sum(spurious_en, axis=1))
    return reduced_loss_missing,reduced_loss_spurious

def soft_f1_score(y_true, y_pred):
    dense_clclass,dense_windclass, mask_cls, _  = y_pred
    y_clclass, y_windclass, cl_X, y_metadata = y_true
    y_target = tf.cast(y_clclass, tf.float32) 

    pred_prob = tf.nn.sigmoid(dense_clclass)
    tp = tf.reduce_sum(pred_prob * y_target, axis=1)
    fn = tf.reduce_sum((1 - pred_prob) * y_target, axis=1)
    fp = tf.reduce_sum(pred_prob * (1 - y_target), axis=1)

    soft_f1_loss = 1 - (2 * tp)/ (2*tp + fn + fp + 1e-16) 
    reduced_f1 = tf.reduce_mean(soft_f1_loss)
    return reduced_f1