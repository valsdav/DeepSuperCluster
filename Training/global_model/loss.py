import tensorflow as tf

def simple_classification_loss(y_true, y_pred):
    dense_clclass, mask_cls, _  = y_pred
    y_class, y_metadata = y_true
        
    class_loss = tf.keras.losses.binary_crossentropy(y_class, dense_clclass, from_logits=True) * mask_cls
    reduced_loss = tf.reduce_mean(tf.reduce_sum(class_loss, axis=-1))
    return reduced_loss 

def energy_weighted_classification_loss(y_true, y_pred, X):
    dense_clclass, mask_cls, _  = y_pred
    y_class, y_metadata = y_true
    cl_X, cl_hits, is_seed,n_cl = X
    cl_ets = cl_X[:,:,1]
    # compute the weighting mean of the loss based on the energy of each seed in the window
    cl_ets_weights = cl_ets / tf.reduce_sum(cl_ets, axis=-1)[:,tf.newaxis]
    class_loss = tf.keras.losses.binary_crossentropy(y_class, dense_clclass, from_logits=True) * mask_cls
    weighted_loss = class_loss * cl_ets_weights
    # mean over the batch
    reduced_loss = tf.reduce_mean(tf.reduce_sum(weighted_loss, axis=-1))
    return reduced_loss


# def simenergy_weighted_classification_loss(y_true, y_pred, X):
#     dense_clclass, mask_cls, _  = y_pred
#     y_class, y_metadata = y_true
#     cl_X, cl_hits, is_seed,n_cl = X
#     # compute the weighting mean of the loss based on the energy fraction wrt of the calo simenergy
#     if y_metadata
#     cl_ets_weights = cl_X[:,:,1] / y[:,2]
#     class_loss = tf.keras.losses.binary_crossentropy(y_class, dense_clclass, from_logits=True) * mask_cls
#     weighted_loss = class_loss * cl_ets_weights
#     # mean over the batch
#     reduced_loss = tf.reduce_mean(tf.reduce_sum(weighted_loss, axis=-1))
#     return reduced_loss