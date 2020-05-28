# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:11:42 2020

@author: rashmibh
"""

import cv2
from keras.models import load_model
import numpy as np
import os
  
#Video to use for face tagging
Video_File_Name='VID_20200518_191839.mp4'
MODEL_FILE='model.facedetect_family'
INDICES_FILE='class_indices_saved_family.npy'
list_indices=[]
if os.path.isfile(INDICES_FILE):
    class_indices = np.load(INDICES_FILE,allow_pickle=True).item()
    [list_indices.extend([[k,v]]) for k,v in class_indices.items()]

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

#Load the saved model
model = load_model(MODEL_FILE)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Setup face detection part   
cascPath =  'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)


video_capture = cv2.VideoCapture(Video_File_Name)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    #print(frame)
    (h, w,d)= frame.shape    
    #Detection not working well on large images
    r = 600.0 / w
    dim = (600, int(h * r))
    resized = cv2.resize(frame, dim)
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
  
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
        
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
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
    # Display the resulting frame
    cv2.imshow(Video_File_Name, resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()