import numpy as np
import pandas as pd
import get_data
import seaborn as sns
import matplotlib.pyplot as plt
#SINCE I USE AN AMDGPU I CANT USE TENSORFLOW BACKEND -- IF YOU USE TENSORFLOW, JUST IMPORT KERAS THAT WAY INSTEAD
import plaidml.keras
plaidml.keras.install_backend()
#-----
import keras.backend as K
from keras.models import Sequential,Model
from keras.layers import Lambda,Conv2D,Dense,Input,Flatten,MaxPool2D,Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

def get_model_1(input_shape):

    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    model.add(Conv2D(64, (11,11), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D())
    model.add(Conv2D(128, (7,7), activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(128, (4,4), activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(256, (4,4), activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='sigmoid'))
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    #Difference between encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid')(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    return siamese_net

def get_model_2(input_shape):

    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    model.add(MaxPool2D())
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dropout(rate=0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='sigmoid'))
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    #Difference between encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid')(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    return siamese_net

def get_model_3(input_shape):

    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(256, (3,3), activation='relu'))
    model.add(Conv2D(256, (3,3), activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(512, (3,3), activation='relu'))
    model.add(Conv2D(512, (3,3), activation='relu'))
    model.add(Conv2D(512, (3,3), activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dropout(rate=0.4))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='sigmoid'))
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    #Difference between encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid')(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    return siamese_net
