## Graph Highway network
import tensorflow as tf
import numpy as np

#########################3
# Masking utils

def create_padding_masks(rechits):
    mask_rechits = tf.cast(tf.reduce_sum(rechits,-1) != 0, tf.float32)
    mask_cls = tf.cast(tf.reduce_sum(rechits,[-1,-2]) != 0, tf.float32)
    return mask_rechits, mask_cls


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
        f_hom = tf.linalg.matmul(adj_k, f_het*norm_k)*norm_k

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
        f_hom = tf.linalg.matmul(adj_k, f_hom*norm_k)*norm_k

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

###########################3
class SelfAttention(tf.keras.layers.Layer):
    '''
    Generic self attention layer that can reduce or not the output feature vectors.
    Input : [Nbatch, Nclusters, input_dim]
    Output:  
            - reduce=none  [Nbatch, Nclusters, output_dim]  
            - reduce=sum or mean   [Nbatch, output_dim]  
    '''    
    def __init__(self, input_dim, output_dim, reduce=None, *args, **kwargs):
        self.activation = kwargs.pop("activation")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reduce = reduce # it can be None, sum, mean
    
        super(SelfAttention, self).__init__(*args, **kwargs)
        
        # Self-attention matrices
        self.Q = self.add_weight(shape=(self.input_dim, self.output_dim), name="Q_sa", initializer="random_normal")
        self.K = self.add_weight(shape=(self.input_dim, self.output_dim), name="K_sa", initializer="random_normal")
        self.V = self.add_weight(shape=(self.input_dim, self.output_dim), name="V_sa", initializer="random_normal")

        # Output Conv1d / or Dense 
        #self.conv1d_out = tf.keras.layers.Conv1D(self.output_dim, 1, padding="Valid", activation=tf.nn.selu)
        self.dense_out = tf.keras.layers.Dense(self.output_dim, activation=self.activation)
        
    def call(self, x, mask):
        # x has structure  [Nbatch, Nclusters, Nfeatures]
        q = tf.matmul(x,self.Q)
        k = tf.matmul(x,self.K)
        v = tf.matmul(x,self.V)
        # mask the padded clusters in the attention distance
        mask_for_attention = mask[:,tf.newaxis,:]
        sa_output, attention_weights = scaled_dot_product_attention(q, k, v, mask_for_attention)
        # Mask for output
        mask_for_nodes = mask[:,:,tf.newaxis]
        # Apply dense layer on each rechit output before the aggregation
        out_nodes = self.dense_out(sa_output) * mask_for_nodes
   
        # Now the aggregation 
        if self.reduce == "sum":
            return  tf.reduce_sum(out_nodes, -2),attention_weights
        if self.reduce == "mean":
            N_nodes = tf.reduce_sum(mask,-1)[:,:,tf.newaxis]
            return tf.math.divide_no_nan( tf.reduce_sum(out_nodes, -2), N_nodes),attention_weights
        else:
            #just return all the nodes
            return out_nodes,attention_weights

############################
## GCN + Self-attention block for rechits feature extraction
# A single features vector of dimension output_dim is built from arbitrary list of rechits. 
# A GHconv block is applied and then a self-attention block 
class RechitsGCN(tf.keras.layers.Layer):
    
    def __init__(self, nconv, input_dim, output_dim, *args, **kwargs):
        self.activation = kwargs.pop("activation")
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.nconv = nconv
    
        super(RechitsGCN, self).__init__(*args, **kwargs)
        
        self.dist = Distance(batch_dim=2)
        self.GCN = GHConvI(n_iter = self.nconv, input_dim=input_dim, hidden_dim=output_dim, activation=self.activation)
        
        # Self-attention matrices
        self.Q = self.add_weight(shape=(self.output_dim, self.output_dim), name="Q_sa", initializer="random_normal")
        self.K = self.add_weight(shape=(self.output_dim, self.output_dim), name="K_sa", initializer="random_normal")
        self.V = self.add_weight(shape=(self.output_dim, self.output_dim), name="V_sa", initializer="random_normal")

        # Output Conv1d / or Dense 
        #self.conv1d_out = tf.keras.layers.Conv1D(self.output_dim, 1, padding="Valid", activation=tf.nn.selu)
        self.dense_out = tf.keras.layers.Dense(self.output_dim, activation=tf.nn.relu)
        
    def call(self, x, mask):
        # x has structure  [Nbatch, Nclusters, Nrechits, 4]
        coord = x[:,:,:,0:2] #ieta and iphi as coordinated
        adj = self.dist(coord,coord)
        # apply GCN in fully batched style
        out_gcn = self.GCN(x,adj)
        
        q = tf.matmul(out_gcn,self.Q)
        k = tf.matmul(out_gcn,self.K)
        v = tf.matmul(out_gcn,self.V)
        mask_for_attention = mask[:,:,tf.newaxis,:]
        sa_output, attention_weights = scaled_dot_product_attention(q, k, v, mask_for_attention)
        # Mask to compute the output masking the correct rechits
        mask_for_output = mask[:,:,:,tf.newaxis]
        # Apply dense layer on each rechit output before the final sum
        convout = self.dense_out(sa_output)
        # Sum the rechits vectors
        output = tf.reduce_sum(convout * mask_for_output, -2)
        # Or doing the mean 
        #N_rechits = tf.reduce_sum(mask,-1)[:,:,tf.newaxis]
        #output = tf.math.divide_no_nan( tf.reduce_sum(convout, -2), N_rechits)
        
        return output, (attention_weights, adj)


################################
# Graph building part of the model
class GraphBuilding(tf.keras.layers.Layer):
    
    def __init__(self,  **kwargs):
        self.activation = kwargs.pop("activation", tf.nn.selu)
        self.layers_input = kwargs.pop("layers_input",[64,64])
        self.layers_coord = kwargs.pop("layers_coord",[64,64])
        self.output_dim_rechits = kwargs.pop("output_dim_rechits",16)
        self.output_dim_nodes = kwargs.pop("output_dim_nodes",32)
        self.coord_dim = kwargs.pop("coord_dim",3)
        self.nconv_rechits = kwargs.pop("nconv_rechits",3)
    
        super(GraphBuilding, self).__init__( **kwargs)
        
        
        self.rechitsGCN = RechitsGCN(output_dim=self.output_dim_rechits, input_dim=4, nconv=self.nconv_rechits, activation=self.activation)
        
        self.dist = Distance(batch_dim=1)
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        
        self.dense_feats = []
        self.dense_coord = []
        #append last layer dimension that is the output dimension of the node features
        self.layers_input.append(self.output_dim_nodes)
        for N in self.layers_input:
            self.dense_feats.append(tf.keras.layers.Dense(N, activation=self.activation))
        
        for N in self.layers_coord:
            self.dense_coord.append(tf.keras.layers.Dense(N, activation=self.activation))
        # last layer fixes the dim of coordinatate space and the linear activation of it
        self.coord_dense_out = tf.keras.layers.Dense(self.coord_dim, activation=tf.keras.activations.linear)

    def call(self, inputs):
        rechits = inputs[1].to_tensor()
        mask_rechits, mask_cls = create_padding_masks(rechits)
        # Cal the rechitGCN and get out 1 vector for each cluster 
        output_rechits, (debug) = self.rechitsGCN(rechits, mask_rechits)
        # Concat the per-cluster feature with the rechits output
        cl_and_rechits = self.concat([inputs[0], output_rechits])
        # apply dense layers for feature building
        for dense in self.dense_feats:
            cl_and_rechits = dense(cl_and_rechits)
        #cl_and_rechits are now the baseline node features
        
        # Now apply the coordinate network 
        coord_x = cl_and_rechits
        for dense in self.dense_coord:
            coord_x = dense(coord_x)
        coord_output = self.coord_dense_out(coord_x)
        # Build the adjacency matrix
        adj = self.dist(coord_output,coord_output)
        # mask the padded clusters      
        adj = adj*mask_cls[:,tf.newaxis,:]
        
        #return the nodes features, the coordinates , the adjacency matrix, the clusters mask
        return  cl_and_rechits, coord_output, adj, mask_cls, debug

