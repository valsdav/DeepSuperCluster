## Graph Highway network
import tensorflow as tf
import numpy as np
from loss import *

#########################3
# Masking utils

def create_padding_masks(rechits):
    mask_rechits = tf.cast(tf.reduce_sum(rechits,-1) != 0, tf.float32)
    mask_cls = tf.cast(tf.reduce_sum(rechits,[-1,-2]) != 0, tf.float32)
    return mask_rechits, mask_cls

###########################

def get_dense(spec, act, last_act, dropout=0., L2=False, L1=False, name="dense"):
    layers = [] 
    for d in spec[:-1]:
        if not L1 and not L2:
            layers.append(tf.keras.layers.Dense(d, activation=act))
        if not L1 and L2:
            layers.append(tf.keras.layers.Dense(d, activation=act, kernel_regularizer='l2'))
        if not L2 and L1:
            layers.append(tf.keras.layers.Dense(d, activation=act, kernel_regularizer='l1'))
        if L1 and L2:
            layers.append(tf.keras.layers.Dense(d, activation=act, kernel_regularizer='l1_l2'))
        if dropout > 0.:
            layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(spec[-1], activation=last_act))
    return tf.keras.Sequential(layers, name=name)

###########################
#Distance

def dist(A,B):
    na = tf.reduce_sum(tf.square(A), -1)
    nb = tf.reduce_sum(tf.square(B), -1)
 
    na = tf.reshape(na, [tf.shape(na)[0], -1, 1])
    nb = tf.reshape(nb, [tf.shape(na)[0], 1, -1])
    Dsq = tf.clip_by_value(na - 2*tf.linalg.matmul(A, B, transpose_a=False, transpose_b=True) + nb, 1e-12, 1e12)
    D = tf.sqrt(Dsq)
    return D

def dist_batch2(A,B):
    na = tf.reduce_sum(tf.square(A), -1)
    nb = tf.reduce_sum(tf.square(B), -1)
 
    na = tf.reshape(na, [tf.shape(na)[0],tf.shape(na)[1], -1, 1])
    nb = tf.reshape(nb, [tf.shape(nb)[0],tf.shape(nb)[1], 1, -1])
    Dsq = tf.clip_by_value(na - 2*tf.linalg.matmul(A, B, transpose_a=False, transpose_b=True) + nb, 1e-12, 1e12)
    D = tf.sqrt(Dsq)
    return D


#Given a list of [Nbatch, Nelem, Nfeat] input nodes, computes the dense [Nbatch, Nelem, Nelem] adjacency matrices
class Distance(tf.keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        self.batch_dim = kwargs.pop('batch_dim',1)
        super(Distance, self).__init__(*args, **kwargs)
        
    def call(self, inputs1, inputs2):
        #compute the pairwise distance matrix between the vectors defined by the first two components of the input array
        #inputs1, inputs2: [Nbatch, Nelem, distance_dim] embedded coordinates used for element-to-element distance calculation
        if self.batch_dim == 1:
            D = dist(inputs1, inputs2)
        if self.batch_dim == 2:
            D = dist_batch2(inputs1, inputs2)
      
        #adjacency between two elements should be high if the distance is small.
        #this is equivalent to radial basis functions. 
        #self-loops adj_{i,i}=1 are included, as D_{i,i}=0 by construction
        adj = tf.math.exp(-1.0*D)

        #optionally set the adjacency matrix to 0 for low values in order to make the matrix sparse.
        #need to test if this improves the result.
        #adj = tf.keras.activations.relu(adj, threshold=0.01)
        return adj

###############################################
# https://arxiv.org/pdf/2004.04635.pdf
#https://github.com/gcucurull/jax-ghnet/blob/master/models.py 
class GHConvI(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim, n_iter, *args, **kwargs):
        self.activation = kwargs.pop("activation")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = n_iter

        super(GHConvI, self).__init__(*args, **kwargs)

        self.W_t = self.add_weight(shape=(self.input_dim, self.hidden_dim), name="w_t", initializer="random_normal")
        self.b_t = self.add_weight(shape=(self.hidden_dim, ), name="b_t", initializer="zeros")
        self.theta = self.add_weight(shape=(self.input_dim, self.hidden_dim), name="theta", initializer="random_normal")
    
    def call(self, x, adj):
        #compute the normalization of the adjacency matrix
        in_degrees = tf.reduce_sum(adj, axis=-1)
        #add epsilon to prevent numerical issues from 1/sqrt(x)
        norm = tf.expand_dims(tf.pow(in_degrees + 1e-6, -0.5), -1)
        norm_k = tf.pow(norm, self.k)
        adj_k = tf.pow(adj, self.k)

        f_het = tf.linalg.matmul(x, self.theta)  #inner infusion
        # Added activation to homogenous component
        f_hom = self.activation(tf.linalg.matmul(adj_k, f_het*norm_k)*norm_k)

        gate = tf.nn.sigmoid(tf.linalg.matmul(x, self.W_t) + self.b_t)
        #tf.print(tf.reduce_mean(f_hom), tf.reduce_mean(f_het), tf.reduce_mean(gate))

        out = gate*f_hom + (1-gate)*f_het
        return out

class GHConvO(tf.keras.layers.Layer):
    def __init__(self, k, *args, **kwargs):
        self.activation = kwargs.pop("activation")
        self.hidden_dim = args[0]
        self.k = k

        super(GHConvO, self).__init__(*args, **kwargs)

        self.W_t = self.add_weight(shape=(self.hidden_dim, self.hidden_dim), name="w_t", initializer="random_normal")
        self.b_t = self.add_weight(shape=(self.hidden_dim, ), name="b_t", initializer="zeros")
        self.W_h = self.add_weight(shape=(self.hidden_dim, self.hidden_dim), name="w_h", initializer="random_normal")
        self.theta = self.add_weight(shape=(self.hidden_dim, self.hidden_dim), name="theta", initializer="random_normal")
    
    def call(self, x, adj):
        #compute the normalization of the adjacency matrix
        in_degrees = tf.reduce_sum(adj, axis=-1)
        #add epsilon to prevent numerical issues from 1/sqrt(x)
        norm = tf.expand_dims(tf.pow(in_degrees + 1e-6, -0.5), -1)
        norm_k = tf.pow(norm, self.k)
        adj_k = tf.pow(adj, self.k)

        f_hom = tf.linalg.matmul(x, self.theta)
        # Added activation to homogenous component
        f_hom = self.activation(tf.linalg.matmul(adj_k, f_hom*norm_k)*norm_k)

        f_het = tf.linalg.matmul(x, self.W_h)  #outer infusion
        gate = tf.nn.sigmoid(tf.linalg.matmul(x, self.W_t) + self.b_t)
        #tf.print(tf.reduce_mean(f_hom), tf.reduce_mean(f_het), tf.reduce_mean(gate))

        out = gate*f_hom + (1-gate)*f_het
        return out

#######################
## Simple Graph Conv layer
class SGConv(tf.keras.layers.Dense):
    def __init__(self, k, *args, **kwargs):
        super(SGConv, self).__init__(*args, **kwargs)
        self.k = k
    
    def call(self, inputs, adj):
        W = self.weights[0]
        b = self.weights[1]

        #compute the normalization of the adjacency matrix
        in_degrees = tf.reduce_sum(adj, axis=-1)
        #add epsilon to prevent numerical issues from 1/sqrt(x)
        norm = tf.expand_dims(tf.pow(in_degrees + 1e-6, -0.5), -1)
        norm_k = tf.pow(norm, self.k)

        support = (tf.linalg.matmul(inputs, W))
     
        #k-th power of the normalized adjacency matrix is nearly equivalent to k consecutive GCN layers
        adj_k = tf.pow(adj, self.k)
        out = tf.linalg.matmul(adj_k, support*norm_k)*norm_k

        return self.activation(out + b)


############################
# From https://www.tensorflow.org/tutorials/text/transformer#multi-head_attention
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        # the mask signals the elements to keep so we have to invert it
        scaled_attention_logits += ( (1 - mask) * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights

########################################

class SelfAttention(tf.keras.layers.Layer):
    '''
    Generic self attention layer that can reduce or not the output feature vectors.
    Input : [Nbatch, Nclusters, input_dim]
    Output:  
            - reduce=none  [Nbatch, Nclusters, output_dim]  
            - reduce=sum or mean   [Nbatch, output_dim]  
    '''    
    def __init__(self, input_dim, output_dim, reduce=None, *args, **kwargs):
        self.activation = kwargs.pop("activation", "relu")
        self.dropout = kwargs.get("dropout", 0.)
        self.l2_reg = kwargs.get("l2_reg", False)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reduce = reduce # it can be None, sum, mean, max
        name = kwargs.pop("name", None)

        super(SelfAttention, self).__init__(name=name)
        
        # Self-attention matrices
        self.Q = self.add_weight(shape=(self.input_dim, self.output_dim), name="Q_sa", initializer="random_normal")
        self.K = self.add_weight(shape=(self.input_dim, self.output_dim), name="K_sa", initializer="random_normal")
        self.V = self.add_weight(shape=(self.input_dim, self.output_dim), name="V_sa", initializer="random_normal")
        self.inputW = self.add_weight(shape=(self.input_dim, self.output_dim), name="input_sa", initializer="random_normal")

        # Feed-forward output (1 hidden layer)
        self.dense_out = get_dense([self.output_dim, self.output_dim], self.activation, last_act=self.activation,
                                    L2=self.l2_reg, dropout=self.dropout, name="output_sa")
        # Layer normalizations
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-3, axis=-1)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-3, axis=-1)
        # Dropouts
        self.drop1 = tf.keras.layers.Dropout(self.dropout)
        self.drop2 = tf.keras.layers.Dropout(self.dropout)
        
    def call(self, x, mask, training):
        # x has structure  [Nbatch, Nclusters, Nfeatures]
        q = tf.matmul(x,self.Q)
        k = tf.matmul(x,self.K)
        v = tf.matmul(x,self.V)
        # Apply the dense_input to x to get transformed dimension
        transf_input = tf.matmul(x,self.inputW)
        # mask the padded clusters in the attention distance
        mask_for_attention = mask[:,tf.newaxis,:]
        # Mask for output
        mask_for_nodes = mask[:,:,tf.newaxis]
        # Get self-attention output and attention weights
        sa_output, attention_weights = scaled_dot_product_attention(q, k, v, mask_for_attention)
        # Dropout
        sa_output = self.drop1(sa_output, training=training) 
        # Add + layer norm  + mask
        output_sa = self.norm1(transf_input + sa_output) * mask_for_nodes
        # Apply dense
        output_dense = self.dense_out(output_sa, training=training) 
        # Dropout
        output_dense = self.drop2(output_dense, training=training)
        # Add and layer norm
        output_block = self.norm2(output_dense + output_sa) * mask_for_nodes

        # Now the aggregation 
        if self.reduce == "sum":
            return  tf.reduce_sum(output_block, -2), attention_weights
        if self.reduce == "mean":
            N_nodes = tf.reduce_sum(mask,-1)[:,tf.newaxis]
            return tf.math.divide_no_nan( tf.reduce_sum(output_block, -2), N_nodes), attention_weights
        if self.reduce == "max":
            return tf.reduce_max(output_block, axis=-2), attention_weights
        else:
            #just return all the nodes
            return output_block, attention_weights


############################
## GCN + Self-attention block for rechits feature extraction
# A single features vector of dimension output_dim is built from arbitrary list of rechits. 
# A GHconv block is applied and then a self-attention block 
class RechitsGCN(tf.keras.layers.Layer):

    def __init__(self, nconv, input_dim, output_dim,  *args, **kwargs):
        self.activation = kwargs.pop("activation", tf.keras.activations.relu)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.dropout = kwargs.pop("dropout", 0.01)
        self.l2_reg = kwargs.pop("l2_reg", False)
        self.nconv = nconv
    
        super(RechitsGCN, self).__init__(*args, **kwargs)
        
        self.dist = Distance(batch_dim=2)
        self.GCN = GHConvI(n_iter = self.nconv, input_dim=input_dim, hidden_dim=output_dim, activation=self.activation)
        
        # Self-attention matrices
        self.Q = self.add_weight(shape=(self.output_dim, self.output_dim), name="Q_sa", initializer="random_normal")
        self.K = self.add_weight(shape=(self.output_dim, self.output_dim), name="K_sa", initializer="random_normal")
        self.V = self.add_weight(shape=(self.output_dim, self.output_dim), name="V_sa", initializer="random_normal")

        #  Dense 
        # Feed-forward output (1 hidden layer)
        self.dense_out = get_dense([self.output_dim, self.output_dim], self.activation, last_act=self.activation,
                                    L2=self.l2_reg, dropout=self.dropout)
        # Dropouts
        self.drop1 = tf.keras.layers.Dropout(self.dropout)
        self.drop2 = tf.keras.layers.Dropout(self.dropout)
        #Layer normalizations
        self.sa_normalization = tf.keras.layers.LayerNormalization(epsilon=1e-3, axis=-1)
        self.out_normalization = tf.keras.layers.LayerNormalization(epsilon=1e-3, axis=-1)
        
    def call(self, x, mask, training):
        # x has structure  [Nbatch, Nclusters, Nrechits, 4]
        coord = x[:,:,:,0:2] #ieta and iphi as coordinated
        # create mask for adjacency matrix
        adj_mask = m =  mask[:,:,:,tf.newaxis] @ mask[:,:,tf.newaxis, :]
        # compute adjacency and mask it
        adj = self.dist(coord,coord) * adj_mask
        # apply GCN in fully batched style
        out_gcn = self.GCN(x,adj)
        # And now SA layer for aggregation
        q = tf.matmul(out_gcn,self.Q)
        k = tf.matmul(out_gcn,self.K)
        v = tf.matmul(out_gcn,self.V)
        mask_for_attention = mask[:,:,tf.newaxis,:]
        sa_output, attention_weights = scaled_dot_product_attention(q, k, v, mask_for_attention)
        # Mask to compute the output masking the correct rechits
        mask_for_output = mask[:,:,:,tf.newaxis]
        # Layer normalizationa and dropout on SA output
        sa_output = self.drop1(sa_output, training=training)
        sa_output = self.sa_normalization(sa_output) * mask_for_output
        # Apply dense layer on each rechit output before the final sum
        dense_output = self.dense_out(sa_output, training=training) 
        dense_output = self.drop2(dense_output, training=training)
        # Add + Norm
        dense_output = self.out_normalization(dense_output + sa_output) * mask_for_output
        # Sum the rechits vectors
        # output = tf.reduce_sum(convout * mask_for_output, -2)
        # Or doing the mean 
        N_rechits = tf.reduce_sum(mask,-1)[:,:,tf.newaxis]
        output = tf.math.divide_no_nan( tf.reduce_sum(dense_output, -2), N_rechits)
        return output, (sa_output,dense_output, attention_weights, adj)

################################
# Graph building part of the model
class GraphBuilding(tf.keras.layers.Layer):
    
    def __init__(self,  **kwargs):
        self.activation = kwargs.get("activation", tf.nn.selu)
        self.layers_input = kwargs.pop("layers_input",[64,64])
        self.output_dim_rechits = kwargs.pop("output_dim_rechits",16)
        self.output_dim_nodes = kwargs.pop("output_dim_nodes",32)
        self.coord_dim = kwargs.pop("coord_dim",3)
        self.nconv_rechits = kwargs.pop("nconv_rechits",3)
        self.dropout = kwargs.get("dropout", 0.)
        self.l2_reg = kwargs.get("l2_reg", False)
        name = kwargs.get("name", None)
            
        self.rechitsGCN = RechitsGCN(name="rechit_gcn", output_dim=self.output_dim_rechits, input_dim=4, 
                                nconv=self.nconv_rechits, activation=self.activation, dropout=self.dropout)
        
        #Self-attention for coordinations
        self.SA_coord = SelfAttention(name="rechit_SA", input_dim=self.output_dim_nodes, output_dim=self.coord_dim, dropout=0.)
        self.dist = Distance(batch_dim=1)
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        
        #append last layer dimension that is the output dimension of the node features
        self.dense_feats = get_dense(self.layers_input+[self.output_dim_nodes], self.activation,last_act=self.activation,
                                    L2=self.l2_reg, dropout=self.dropout)
        
        #Layer normalizations
        self.feat_layer_normalization = tf.keras.layers.LayerNormalization(epsilon=1e-3)

        super(GraphBuilding, self).__init__(name=name)


    def call(self, cl_features, rechits_features, training):
        # Conversion from RaggedTensor to dense tensor
        rechits = rechits_features.to_tensor()
        mask_rechits, mask_cls = create_padding_masks(rechits)
        # Cal the rechitGCN and get out 1 vector for each cluster 
        output_rechits, (debug) = self.rechitsGCN(rechits, mask_rechits, training=training)
        
        # Layer normalization on the two pieces
        # output_rechits_norm = self.rechit_layer_normalization(output_rechits)
        # cl_features_norm = self.input_layer_normalization(cl_features)

        # Concat the per-cluster feature with the rechits output
        cl_and_rechits = self.concat([cl_features, output_rechits])
        # apply dense layers for feature building 
        cl_and_rechits = self.dense_feats(cl_and_rechits, training=training) 
        # apply normalization and also applying the mask (to be sure)
        cl_and_rechits = self.feat_layer_normalization(cl_and_rechits) * mask_cls[:,:,tf.newaxis]

        #cl_and_rechits are now the baseline node features
        
        # Now apply the coordinate network with a layer normalization before
        coord_output, coord_att_ws = self.SA_coord(cl_and_rechits, mask_cls, training)
        # Build the adjacency matrix
        adj = self.dist(coord_output,coord_output)
        # mask the padded clusters      
        adj_mask = m =  mask_cls[:,:,tf.newaxis] @ mask_cls[:,tf.newaxis, :]
        adj = adj* adj_mask
        
        #return the nodes features, the coordinates , the adjacency matrix, the clusters mask
        return  cl_and_rechits, coord_output, adj, mask_cls, output_rechits, coord_att_ws



#############################################
# Putting all the pieces together

class DeepClusterGN(tf.keras.Model):
    '''
    Model parameters:
    - activation
    - output_dim_nodes: latent space dimension for clusters node built from rechits and cluster features
    - output_dim_rechits:  latent space dimension for the rechits per-cluster feature vector
    - output_dim_gconv: output of the graph convolution (default==output_dim_nodes)\
    - output_dim_sa_clclass: output of the self-attention layer for cluster classification (default==output_dim_gconv)
    - output_dim_sa_windclass: output of the self-attention layer for windows classification (default==output_dim_gconv)
    - coord_dim:  coordinated space dimension
    - nconv_rechits: number of convolutions for the rechits GCN
    - nconv: number of convolutions for the global model
    - layers_input:  list representing the DNN applied on the [rechit+cluster] concatened features to build the clusters latent space
    - layers_clclass:  list representing the DNN for cluster classification eg [64,64]
    - layers_windclass:  list representing the DNN for window classification eg [64,64]
    - n_windclasses: number of classes for window classification
    - dropout: dropout function to apply on classification DNN
    - l2_reg: activate l2 regularization in all the Dense layers
    - loss_weights:  dictionary "loss_clusters, loss_window, loss_etw, loss_et_miss, loss_et_spur"
    '''
    def __init__(self, **kwargs):
        self.activation = kwargs.get("activation", tf.nn.selu)
        self.output_dim_nodes = kwargs.get("output_dim_nodes",32)
        self.output_dim_gconv = kwargs.pop("output_dim_gconv",self.output_dim_nodes)
        self.output_dim_sa_clclass = kwargs.pop("output_dim_sa_clclass",self.output_dim_gconv)
        self.output_dim_sa_windclass = kwargs.pop("output_dim_sa_windclass",self.output_dim_gconv)
        self.nconv = kwargs.pop("nconv",3)
        self.layers_clclass = kwargs.pop("layers_clclass",[64,64])
        self.layers_windclass = kwargs.pop("layers_windclass",[64,64])
        self.n_windclasses = kwargs.pop("n_windclasses", 1)
        self.dropout = kwargs.get("dropout",0.)
        self.l2_reg = kwargs.get("l2_reg", False)
        self.loss_weights = kwargs.get("loss_weights", {"clusters":1., "window":1., "etw":1., "et_miss":1., "et_spur":1})
        
        super(DeepClusterGN, self).__init__()
        
        self.graphbuild = GraphBuilding(name="graph_builder", **kwargs)
        self.GCN = GHConvI(name="GHN_global", n_iter =self.nconv, input_dim=self.output_dim_nodes , 
                                        hidden_dim=self.output_dim_gconv , activation=self.activation)

        # Clusters classification head
        self.SA_clclass = SelfAttention(name="SA_clclass", input_dim=self.output_dim_gconv, output_dim=self.output_dim_sa_clclass, 
                                        reduce=None, **kwargs)
        self.dense_clclass = get_dense(name="dense_clclass", spec=self.layers_clclass+[1], act=self.activation,
                                     last_act=tf.keras.activations.linear, dropout=self.dropout, L2=self.l2_reg)
        # Window classification head
        # self.concat_gcn_SAcl = tf.keras.layers.Concatenate(axis=-1)
        # self-attention for windows classification with "mean" reduction
        self.SA_windclass = SelfAttention(name="SA_windclass", input_dim=self.output_dim_gconv, output_dim=self.output_dim_sa_windclass,
                                         reduce="sum", **kwargs)
        self.dense_windclass = get_dense(name="dense_windclass", spec=self.layers_windclass+[self.n_windclasses], act=self.activation,
                                     last_act=tf.keras.activations.linear, dropout=self.dropout, L2=self.l2_reg)

        #dropout layers
        self.gcn_dropout = tf.keras.layers.Dropout(self.dropout)
        # self.SA_windclass_input_dropout = tf.keras.layers.Dropout(self.dropout)
        self.SA_windclass_output_dropout = tf.keras.layers.Dropout(self.dropout)
        # Layer normalizations
        self.gcn_output_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-3)
        # self.input_SA_windclass_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-3)
        self.SA_windclass_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-3)

    def call(self, inputs, training):
        cl_X_initial, cl_hits, is_seed,n_cl = inputs 
        # Concatenate the seed label on clusters features
        cl_X_initial = tf.concat([tf.cast(is_seed, tf.float32), cl_X_initial], axis=-1)
        #cl_X now is the latent cluster+rechits representation
        cl_X, coord, adj, mask_cls, output_rechits,coord_att_ws = self.graphbuild(cl_X_initial, cl_hits, training)
        mask_cls_to_apply = mask_cls[:,:,tf.newaxis]
        out_gcn = self.GCN(cl_X, adj) 
        # Dropout + normalization
        out_gcn = self.gcn_dropout(out_gcn, training=training)
        out_gcn = self.gcn_output_layernorm(out_gcn) * mask_cls_to_apply
    
        # Apply self attention: output already masked internally
        out_SA_clclass, att_weights_clclass = self.SA_clclass(out_gcn, mask_cls, training)
        # No need to normalize since the self-attention layer
        # interanally has skip connections and add+norms
        clclass_out = self.dense_clclass(out_SA_clclass, training=training) * mask_cls_to_apply

        # Concatenate the GCN and SA_clusterclassification output
        # in_SA_windcl = self.concat_gcn_SAcl([out_gcn, out_SA_clclass])
        # Apply Self-attention for window classification
        # input_SA_windcl = self.input_SA_windclass_layernorm(out_gcn + out_SA_clclass)
        # input_SA_windcl = self.SA_windclass_input_dropout(input_SA_windcl, training=training)
        out_SA_windcl, att_weights_windclass = self.SA_windclass(out_gcn, mask_cls, training)
        # Norm before dense for wind classification because the sum is performed in the SA layer
        out_SA_windcl = self.SA_windclass_layernorm(out_SA_windcl)
        out_SA_windcl = self.SA_windclass_output_dropout(out_SA_windcl, training=training)
        windclass_out = self.dense_windclass(out_SA_windcl, training=training)
       
        return clclass_out, windclass_out, mask_cls, \
               (  cl_X, coord, adj, coord_att_ws, output_rechits, out_gcn, \
                  out_SA_clclass, out_SA_windcl, att_weights_clclass,att_weights_windclass)

    ########################
    # Training related methods
    def set_metrics(self):
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.loss1_tracker = tf.keras.metrics.Mean(name="loss_clusters")
        self.loss2_tracker = tf.keras.metrics.Mean(name="loss_windows")
        self.loss3_tracker = tf.keras.metrics.Mean(name="loss_etw")
        self.loss4_tracker = tf.keras.metrics.Mean(name="loss_et_miss")
        self.loss5_tracker = tf.keras.metrics.Mean(name="loss_et_spur")

    # Customized training loop
    # Based on https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit/
    def train_step(self, data):
        x, y = data 
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            loss_clusters = clusters_classification_loss(y, y_pred)
            loss_etweighted = energy_weighted_classification_loss(y,y_pred)
            loss_windows = window_classification_loss(y, y_pred)
            loss_et_miss, loss_et_spur = energy_loss(y, y_pred)
            # Total loss function
            loss =  self.loss_weights["clusters"] * loss_clusters +\
                    self.loss_weights["window"] * loss_windows + \
                    self.loss_weights["etw"] * loss_etweighted + \
                    self.loss_weights["et_miss"] * loss_et_miss + \
                    self.loss_weights["et_spur"] *  loss_et_spur + self.losses

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.loss1_tracker.update_state(loss_clusters)
        self.loss2_tracker.update_state(loss_windows)
        self.loss3_tracker.update_state(loss_etweighted)
        self.loss4_tracker.update_state(loss_et_miss)
        self.loss5_tracker.update_state(loss_et_spur)
        # mae_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(),
                "loss_clusters": self.loss1_tracker.result(),
                "loss_windows": self.loss2_tracker.result(),
                "loss_etw": self.loss3_tracker.result(),
                "loss_et_miss": self.loss4_tracker.result(),
                "loss_et_spur": self.loss5_tracker.result(),}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        loss_clusters = clusters_classification_loss(y, y_pred)
        loss_etweighted = energy_weighted_classification_loss(y,y_pred)
        loss_windows = window_classification_loss(y, y_pred)
        loss_et_miss, loss_et_spur = energy_loss(y, y_pred)
        # Total loss function
        loss =  self.loss_weights["clusters"] * loss_clusters +\
                self.loss_weights["window"] * loss_windows + \
                self.loss_weights["etw"] * loss_etweighted + \
                self.loss_weights["et_miss"] * loss_et_miss + \
                self.loss_weights["et_spur"] *  loss_et_spur + self.losses
        
        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.loss1_tracker.update_state(loss_clusters)
        self.loss2_tracker.update_state(loss_windows)
        self.loss3_tracker.update_state(loss_etweighted)
        self.loss4_tracker.update_state(loss_et_miss)
        self.loss5_tracker.update_state(loss_et_spur)
        # mae_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(),
                "loss_clusters": self.loss1_tracker.result(),
                "loss_windows": self.loss2_tracker.result(),
                "loss_etw": self.loss3_tracker.result(),
                "loss_et_miss": self.loss4_tracker.result(),
                "loss_et_spur": self.loss5_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.loss1_tracker, self.loss2_tracker, self.loss3_tracker,
                self.loss4_tracker, self.loss5_tracker]


    