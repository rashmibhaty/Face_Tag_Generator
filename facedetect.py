# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:27:03 2020

@author: rashmibh
"""

import cv2
import os

filelist=os.listdir('.')
for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
    if not(fichier.endswith(".JPG")) and not(fichier.endswith(".jpg")) and not(fichier.endswith(".PNG")) and not(fichier.endswith(".png")):
        filelist.remove(fichier)
print(filelist)

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for img_name in filelist:
    # Read the input image
    img = cv2.imread(img_name)
    
    #Detection not working well on large images
    r = 600.0 / w
    dim = (600, int(h * r))
    resized = cv2.resize(img, dim)
    
    # Convert into grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    #Remove Nosise for better detection
    gray = cv2.GaussianBlur(gray, (25, 25), 0)
    
    # Detect faces
    #minNeighbors , set to 7
    faces = face_cascade.detectMultiScale(gray, 1.1, 7)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(resized, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    cv2.imshow(img_name, resized)
    cv2.waitKey()