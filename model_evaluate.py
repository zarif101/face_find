import numpy as np
import pandas as pd
import get_data
import seaborn as sns
import matplotlib.pyplot as plt
import get_data
from sklearn.metrics import accuracy_score
#SINCE I USE AN AMDGPU I CANT USE TENSORFLOW BACKEND -- IF YOU USE TENSORFLOW, JUST IMPORT KERAS THAT WAY INSTEAD
import plaidml.keras
plaidml.keras.install_backend()
#-----
from keras.models import load_model

def main(model_name):
    model = load_model(model_name)

    same_train = pd.read_csv('data/same_pairs_train.csv')
    different_train = pd.read_csv('data/different_pairs_train.csv')
    same_test = pd.read_csv('data/same_pairs_test.csv')
    different_test = pd.read_csv('data/different_pairs_test.csv')

    data_loader = get_data.dataLoader(same_train,same_test,different_train,different_test)
    X_train,X_test,y_train,y_test = data_loader.load_data()

    threshold = 0.65

    preds = []
    results = model.evaluate([X_test[:,0],X_test[:,1]],y_test)
    print(results)
    for i in range(0,len(y_test)):
        pred = model.predict([X_test[i,0,:].reshape(1,128,128,3),X_test[i,1,:].reshape(1,128,128,3)])

        if pred[0][0] > threshold:
            preds.append(1)
        else:
            preds.append(0)
        #print(pred[0][0])

    preds = np.array(preds)
    accuracy = accuracy_score(y_test,preds)

    print('ACCURACY IS {}%'.format(str(accuracy)))

main('face_model.h5')
