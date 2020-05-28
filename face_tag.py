# -*- coding: utf-8 -*-
"""
Created on Sat May 23 21:08:56 2020

@author: rashmibh
"""
MODEL_FILE='model.facedetect_family'
INDICES_FILE='class_indices_saved_family.npy'


from keras.models import load_model
import cv2
import numpy as np
import os

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
list_indices=[]
if os.path.isfile(INDICES_FILE):
    class_indices = np.load(INDICES_FILE,allow_pickle=True).item()
    [list_indices.extend([[k,v]]) for k,v in class_indices.items()]

model = load_model(MODEL_FILE)

DIR_path="validation_full/"
   
filelist=os.listdir(DIR_path) 

#Setup face detection part 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
    if not(fichier.endswith(".JPG")) and not(fichier.endswith(".jpg")) and not(fichier.endswith(".PNG")) and not(fichier.endswith(".png")):
        filelist.remove(fichier)

for img_name in filelist:
    # Read the input image
    img = cv2.imread(DIR_path+img_name)
        
    print(img_name)
    (h, w, d) = img.shape    
    #Detection not working well on large images
    r = 600.0 / w
    dim = (600, int(h * r))
    resized = cv2.resize(img, dim)

    # Convert into grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    #Remove Nosise for better detection
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    # Detect faces
    #minNeighbors , set to 7
    faces = face_cascade.detectMultiScale(gray, 1.1, 9)
    # Draw rectangle around the faces

    for (x, y, w, h) in faces:
        cv2.rectangle(resized, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi = resized[y:(y+h), (x):(x+w)]
        roi = cv2.resize(roi,(64,64))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        #Get the prection value for the current face
        pred = model.predict(roi[np.newaxis, :, :, np.newaxis]/255)
        print(pred)

        for item in list_indices:
            if item[1] == np.argmax(pred):
                name=item[0]
                break
            
       
        cv2.putText(resized, name, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(resized, str(pred.max()), (x, y+h+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
  
    cv2.imshow(img_name, resized)
    cv2.waitKey(0)

