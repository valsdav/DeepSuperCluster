import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

def plotM(*args,**kwargs):
    t = kwargs.get("t", True)
    f, axs = plt.subplots(1,len(args), figsize=(6*len(args), 6), dpi=100)
    if len(args)>1:
        for i in range(len(args)):
            if t:   c= axs[i].imshow(tf.transpose(args[i]))
            else:   c= axs[i].imshow(args[i])
            plt.colorbar(c, ax=axs[i])
    else:
        if t:  c= axs.imshow(tf.transpose(args[0]), **kwargs)
        else:  c= axs.imshow(args[0], **kwargs)
        plt.colorbar(c, ax=axs)

def plot3D(coord, mask):
    fig = plt.figure(figsize=(25, 25),dpi=150)
    gs = fig.add_gridspec(1,coord.shape[0] )
    for i in range(coord.shape[0]):
        ax = fig.add_subplot(gs[0, i], projection='3d')
        ncl =tf.cast(tf.reduce_sum(mask[i]),tf.int32).numpy()
        c = np.arange(ncl)
        im = ax.scatter(coord[i,0:ncl,0], coord[i,0:ncl,1], coord[i,0:ncl,2],c=c, s=80)
