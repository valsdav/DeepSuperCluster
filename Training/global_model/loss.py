import tensorflow as tf

def simple_classification_loss(y_true, y_pred):
    dense_clclass, mask_cls = y_pred
    y_class, y_metadata,_ = y_true
        
    class_loss = tf.keras.losses.binary_crossentropy(y_class, dense_clclass, from_logits=True) * mask_cls
    reduced_loss = tf.reduce_mean(tf.reduce_sum(class_loss, axis=-1))
    return reduced_loss 