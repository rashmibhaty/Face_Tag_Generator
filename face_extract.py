# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:27:03 2020

@author: rashmibh
"""

import cv2
import os

DIR_path='Photos_to_extract/'
filelist=os.listdir(DIR_path) 

for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
    if not(fichier.endswith(".JPG")) and not(fichier.endswith(".jpg")) and not(fichier.endswith(".PNG")) and not(fichier.endswith(".png")):
        filelist.remove(fichier)
print(filelist)

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for img_name in filelist:
    # Read the input image
    img = cv2.imread(DIR_path+img_name)

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
    faces = face_cascade.detectMultiScale(gray, 1.1, 7)
    # Draw rectangle around the faces
    i=0
    for (x, y, w, h) in faces:
        cv2.rectangle(resized, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi = resized[y:(y+h), (x):(x+w)]
        #cv2.imwrite("saved_face/"+str(i)+ img_name,roi)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("saved_face_gray/"+str(i)+img_name,gray)
        i=i+1
  