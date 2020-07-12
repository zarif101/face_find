import numpy as np
import pandas as pd
import get_data
import seaborn as sns
import matplotlib.pyplot as plt
import models
#SINCE I USE AN AMDGPU I CANT USE TENSORFLOW BACKEND -- IF YOU USE TENSORFLOW, JUST IMPORT KERAS THAT WAY INSTEAD
import plaidml.keras
plaidml.keras.install_backend()
#-----
import keras.backend as K
from keras.models import Sequential,Model
from keras.layers import Lambda,Conv2D,Dense,Input,Flatten,MaxPool2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


def main(input_shape,model_name,epochs):
    same_train = pd.read_csv('data/same_pairs_train.csv')
    different_train = pd.read_csv('data/different_pairs_train.csv')
    same_test = pd.read_csv('data/same_pairs_test.csv')
    different_test = pd.read_csv('data/different_pairs_test.csv')

    data_loader = get_data.dataLoader(same_train,same_test,different_train,different_test)
    X_train,X_val,X_test,y_train,y_val,y_test = data_loader.load_data()
    #generator specific
    train_gen = get_data.image_generator(X_train,y_train,32)
    val_gen = get_data.image_generator(X_val,y_val,200)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(X_val.shape)
    print(y_val.shape)
    #Load model
    model = models.get_model_2(input_shape)
    model.summary()

    model.compile(optimizer=Adam(.00008),loss='binary_crossentropy',metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    #history = model.fit([X_train[:,0],X_train[:,1]],y_train,epochs=int(epochs),validation_split=0.15,
    #callbacks=[early_stopping]
    #)

    #Fit with generator instead
    history = model.fit_generator(train_gen,epochs=int(epochs),validation_data=val_gen)
    
    model.save(model_name)
    print('saved!')
    #PLOT loss
    sns.lineplot(x=[i for i in range(1,len(history.history['loss'])+1)],y=history.history['loss'])
    plt.show()
    #PLOT ACCURACY
    sns.lineplot(x=[i for i in range(1,len(history.history['val_loss'])+1)],y=history.history['val_loss'])
    plt.show()

main((128,128,3),'face_model_v5.h5',epochs=37)
