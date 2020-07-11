import plaidml.keras
plaidml.keras.install_backend()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from cv2.data import haarcascades
from keras.models import load_model
import time

model = load_model('../face_model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
people = os.listdir('people')
#print(people)

def prep(pic):
    global face_cascade
    gray = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray)
    #plt.show()
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x,y,w,h in faces:
        #jeff = cv2.resize(pic[y:y+h,x:x+w],(128,128))
        #jeff = jeff.reshape((1,128,128,3))
        gray_face = cv2.resize(gray[y:y+h,x:x+w],(128,128))
        #plt.imshow(gray_face)
        #plt.show()
        gray_face = gray_face.reshape((1,128,128,1))
        color_face = cv2.resize(pic[y:y+h,x:x+w],(128,128))
        color_face = color_face.reshape((1,128,128,3))
        #print(gray_face.shape)
        #return gray_face, (x,y,w,h)
        return color_face, (x,y,w,h)

def check_and_find(prepped_ref_image):
    global people
    global model
    for person in people:
        #print(person)
        pics = os.listdir('people/'+person)
        matches = []
        for pic in pics:

            img = cv2.imread('people/'+person+'/'+pic)
            face,dumdum = prep(img)
            pred = model.predict([face,prepped_ref_image])
            #print(pred[0][0])
            if pred[0][0] > 0.8:
                #print('MATCH FOUND'+person)
                matches.append(1)
        if len(matches) == len(pics):
            print('MATCH FOUND'+person)
            return person
        else:
            continue

def main():
    global face_cascade
    #ref_face,dumm = prep(ref_image)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        try:
            test_face,coords = prep(frame)
            person = check_and_find(test_face)
            x,y,w,h = coords
            #print(coords)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
            cv2.putText(frame,person,(100,100),FONT_HERSHEY_SIMPLEX,1,(255,0,0),4)
            #plt.imshow(frame)
            #plt.show()
            cv2.imshow('frame',frame)

        except Exception as e:

            #print(e)
            cv2.imshow('frame',frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

main()
