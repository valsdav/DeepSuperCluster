# Import all the required Libraries
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Deconvolution2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam
from keras import callbacks
from keras import backend as K
from keras import regularizers



def get_model(tag, input_dim, optimizer):
    
    model = Sequential()
    
    if tag =="300x3":
        model.add(Dense(300, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(300, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(300, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))
        
        
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model