import numpy as np
import pandas as pd
import cv2
import os
import math
from sklearn.utils import shuffle
#SINCE I USE AN AMDGPU I CANT USE TENSORFLOW BACKEND -- IF YOU USE TENSORFLOW, JUST IMPORT KERAS THAT WAY INSTEAD
import plaidml.keras
plaidml.keras.install_backend()
#-----
from keras.utils import Sequence

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class dataLoader:
    def __init__(self,df_same_train,df_same_test,df_different_train,df_different_test):
        self.df_same_train = df_same_train
        self.df_same_test = df_same_test
        self.df_different_train = df_different_train
        self.df_different_test = df_different_test

        self.X_train_same = np.zeros([len(df_same_train),2,128,128,3])
        self.X_test_same = np.zeros([len(df_same_test),2,128,128,3])
        self.X_train_different = np.zeros([len(df_different_train),2,128,128,3])
        self.X_test_different = np.zeros([len(df_different_test),2,128,128,3])

    def load_same(self,df,add_array):
        global face_cascade
        for index,row in df.iterrows():
            files = os.listdir('data/images/images/'+row['name'])
            pic_1_index = int(row['imagenum1']) - 1
            pic_2_index = int(row['imagenum2']) - 1

            pic1 = cv2.imread('data/images/images/'+row['name']+'/'+files[pic_1_index])
            pic2 = cv2.imread('data/images/images/'+row['name']+'/'+files[pic_2_index])

            pic1_gray = cv2.cvtColor(pic1,cv2.COLOR_BGR2GRAY)
            pic2_gray = cv2.cvtColor(pic2,cv2.COLOR_BGR2GRAY)

            faces1 = face_cascade.detectMultiScale(pic1_gray,1.3,5)
            faces2 = face_cascade.detectMultiScale(pic2_gray,1.3,5)

            for x,y,w,h in faces1:
                pic1_use = pic1[y:y+h,x:x+w]
                break
            for x,y,w,h in faces2:
                pic2_use = pic2[y:y+h,x:x+w]
                break
            pic1_use = cv2.resize(pic1_use,(128,128))
            pic2_use = cv2.resize(pic2_use,(128,128))
            add_array[index,0,:,:,:] = pic1_use
            add_array[index,1,:,:,:] = pic2_use
            #print('done'+str(index))
        return add_array


    def load_different(self,df,to_arr):
        global face_cascade
        for index,row in df.iterrows():
            files1 = os.listdir('data/images/images/'+row['name'])
            files2 = os.listdir('data/images/images/'+row['name.1'])
            pic_1_index = int(row['imagenum1']) - 1
            pic_2_index = int(row['imagenum2']) - 1

            pic1 = cv2.imread('data/images/images/'+row['name']+'/'+files1[pic_1_index])
            pic2 = cv2.imread('data/images/images/'+row['name.1']+'/'+files2[pic_2_index])

            pic1_gray = cv2.cvtColor(pic1,cv2.COLOR_BGR2GRAY)
            pic2_gray = cv2.cvtColor(pic2,cv2.COLOR_BGR2GRAY)

            faces1 = face_cascade.detectMultiScale(pic1_gray,1.3,5)
            faces2 = face_cascade.detectMultiScale(pic2_gray,1.3,5)

            for x,y,w,h in faces1:
                pic1_use = pic1[y:y+h,x:x+w]
                break
            for x,y,w,h in faces2:
                pic2_use = pic2[y:y+h,x:x+w]
                break
            pic1_use = cv2.resize(pic1_use,(128,128))
            pic2_use = cv2.resize(pic2_use,(128,128))
            to_arr[index,0,:,:,:] = pic1_use
            to_arr[index,1,:,:,:] = pic2_use
            #print('done'+str(index))
        return to_arr

    def load_data(self):
        X_train_same = self.load_same(self.df_same_train,self.X_train_same)
        X_test_same = self.load_same(self.df_same_test,self.X_test_same)
        X_train_different = self.load_different(self.df_different_train,self.X_train_different)
        X_test_different = self.load_different(self.df_different_test,self.X_test_different)


        #note that there is the same # of same and different, within train and test sets respectively
        y_train_same = []
        y_train_different = []
        for i in range(len(X_train_same)):
            y_train_same.append(1)
            y_train_different.append(0)
        y_test_same = []
        y_test_different = []
        for i in range(len(X_test_different)):
            y_test_same.append(1)
            y_test_different.append(0)

        y_train_same = np.array(y_train_same)
        y_train_different = np.array(y_train_different)
        y_test_same = np.array(y_test_same)
        y_test_different = np.array(y_test_different)

        X_train = np.concatenate([X_train_same,X_train_different])
        X_test = np.concatenate([X_test_same,X_test_different])

        y_train = np.concatenate([y_train_same,y_train_different])
        y_test = np.concatenate([y_test_same,y_test_different])
        y_train = y_train.reshape((len(y_train),1))
        y_test = y_test.reshape((len(y_test),1))

        X_train = X_train/255
        X_test = X_test/255

        X_train_shuffled,y_train_shuffled = shuffle(X_train,y_train,random_state=5)
        X_test_shuffled,y_test_shuffled = shuffle(X_test,y_test,random_state=5)

        X_val = X_train_shuffled[:200,:]
        y_val = y_train_shuffled[:200,:]

        X_train_shuffled = X_train_shuffled[201:,:]
        y_train_shuffled = y_train_shuffled[201:,:]

        self.X_train_shuffled = X_train_shuffled
        self.X_test_shuffled = X_test_shuffled
        self.X_val = X_val
        self.y_train_shuffled = y_train_shuffled
        self.y_val = y_val
        self.y_test_shuffled = y_test_shuffled

        return X_train_shuffled,X_val,X_test_shuffled,y_train_shuffled,y_val,y_test_shuffled

class image_generator(Sequence):
    def __init__(self,X_set,y_set,batch_size):
        self.X_set = X_set
        self.y_set = y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.X_set) / self.batch_size)

    def __getitem__(self,idx):
        batch_x_1 = self.X_set[idx * self.batch_size:(idx + 1) *
        self.batch_size,0,:]
        batch_x_2 = self.X_set[idx * self.batch_size:(idx + 1) *
        self.batch_size,1,:]

        batch_y = self.y_set[idx * self.batch_size:(idx + 1) *
        self.batch_size,:]

        return [batch_x_1,batch_x_2],batch_y
